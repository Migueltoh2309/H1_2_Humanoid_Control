"""Capa 2 del control bimanual: coordinación por prioridad (sin ROS).

La capa 1 (bimanual_avoidance.py: QP + velocity dampers) garantiza que los
brazos NO choquen, pero es local: cuando los dos quieren el mismo espacio
se frenan mutuamente a d_s y ninguno llega (mínimo local / bloqueo). Esta
capa decide QUIÉN pasa primero, igual que en:

  * Ju et al., Biomimetics 2026 — esquema maestro/esclavo con cambio de
    rol según el estado de los brazos;
  * Shin & Zheng — "path-velocity decomposition": un brazo se retrasa
    para dejar pasar al otro.

Máquina de estados (por par de brazos):

    LIBRE ──(bloqueo detectado)──> CEDE[esclavo]
      ^                               │  el maestro llega a su objetivo
      │                               │  o se aleja (d > d_libre)
      └───────(esclavo reanuda)───────┘
                                       │  el maestro TAMBIÉN se bloquea
                                       └──> se invierten los roles
                                            (hasta max_switches; luego
                                             ESCALAR: hace falta la capa 3)

Bloqueo = hay una restricción brazo-brazo activa (d < d_i) Y ningún brazo
avanza hacia SU objetivo (el de la tarea, no el de retirada) durante
`stall_window` s Y al menos uno no llegó. Si no hay restricción activa y no
avanzan, el problema no es de coordinación (objetivo fuera de alcance) y
esta capa no interviene.

Quién es maestro: el que tiene MENOS distancia por recorrer (el que
libera antes el espacio compartido: "shortest job first"). Empate -> el
brazo `default_master`.

El esclavo no solo se detiene: va a una pose de RETIRADA de su lado. Si
solo se detuviera, su propio cuerpo podría seguir tapando el camino del
maestro (el bloqueo es mutuo).
"""
from collections import deque

import numpy as np

from h1_2_algoritms.bimanual.collision_model import arm_capsules, points_to_segment
from h1_2_algoritms.movimiento.fk_functions import fkine_arm_left_unitree, fkine_arm_right_unitree

FK = {"left": fkine_arm_left_unitree, "right": fkine_arm_right_unitree}
OTHER = {"left": "right", "right": "left"}

# Poses de retirada: la mano atrás y hacia su propio lado, a la altura
# de trabajo (con la mano horizontal, z < ~0.08 no es alcanzable: lo
# limita el pitch de la muñeca, ±0.46 rad).
DEFAULT_RETREAT = {
    "left": np.array([0.22, 0.26, 0.10, 1.0, 0.0, 0.0, 0.0]),
    "right": np.array([0.22, -0.26, 0.10, 1.0, 0.0, 0.0, 0.0]),
}


class BimanualCoordinator:

    def __init__(self, dt, d_influence=0.15, tol_pos=0.02, stall_window=1.0,
                 stall_progress=0.01, d_free=None, max_switches=2,
                 max_assignments=4, default_master="right", retreat=None):
        self.dt = dt
        self.d_i = d_influence
        self.d_free = d_free if d_free is not None else d_influence
        self.tol = tol_pos
        self.window = max(2, int(round(stall_window / dt)))
        self.stall_progress = stall_progress
        self.max_switches = max_switches
        self.max_assignments = max_assignments
        self.default_master = default_master
        self.retreat = retreat if retreat is not None else DEFAULT_RETREAT
        self.reset()

    def reset(self):
        self.state = "free"          # free | yield | escalate
        self.master = None
        self.slave = None
        self.switches = 0
        self.assignments = 0
        self.master_goal_at_yield = None
        self.hist = {s: deque(maxlen=self.window) for s in ("left", "right")}
        self.last_goal = {s: None for s in ("left", "right")}
        self.t_roles = None
        self.events = []             # (t, texto) para registro

    # ------------------------------------------------------------ util
    @staticmethod
    def task_error(q, goals):
        """Error de posición de cada muñeca a su objetivo de TAREA [m]."""
        e = {}
        for i, side in enumerate(("left", "right")):
            p = FK[side](q[7 * i:7 * i + 7])[0:3, 3]
            e[side] = float(np.linalg.norm(np.asarray(goals[side])[:3] - p))
        return e

    def _stalled(self, side):
        h = self.hist[side]
        return len(h) == h.maxlen and (h[0] - h[-1]) < self.stall_progress

    def _clear_of(self, q, master, slave_goal):
        """True si el objetivo del esclavo queda a más de d_i de todas las
        cápsulas del maestro (su mano no está 'sentada' encima)."""
        i = 0 if master == "left" else 1
        p = np.asarray(slave_goal, dtype=float)[:3][None]
        for c in arm_capsules(master, q[7 * i:7 * i + 7]):
            d, _ = points_to_segment(p, c["a"], c["b"])
            if d[0] - c["r"] < self.d_i:
                return False
        return True

    def _log(self, t, msg):
        self.events.append((t, msg))

    # ------------------------------------------------------------ paso
    def step(self, t, q, goals, d_arms):
        """goals: objetivos de la TAREA {"left": xd, "right": xd}.
        d_arms: distancia brazo-brazo mínima del último ciclo
        (diag["d_arms_min"] de la capa 1; SIN los pares con el torso, que
        no son conflicto entre brazos). Devuelve (objetivos para la capa 1,
        error de cada brazo a su objetivo de tarea)."""
        e = self.task_error(q, goals)
        for s in ("left", "right"):
            g = np.asarray(goals[s])[:3]
            if self.last_goal[s] is not None and np.linalg.norm(g - self.last_goal[s]) > 1e-3:
                # Objetivo nuevo: el error salta y no dice nada del avance.
                self.hist[s].clear()
            self.last_goal[s] = g.copy()
            self.hist[s].append(e[s])
        done = {s: e[s] < self.tol for s in e}
        conflict = d_arms < self.d_i

        if self.state == "free":
            if conflict and not all(done.values()) and \
                    all(self._stalled(s) or done[s] for s in e):
                self._assign(t, e, done, goals)

        elif self.state == "yield":
            m, s = self.master, self.slave
            master_moved_on = np.linalg.norm(np.asarray(goals[m])[:3] - self.master_goal_at_yield) > 1e-3
            if master_moved_on and d_arms > self.d_free:
                # La tarea mandó al maestro a otro lado y ya se alejó.
                self._log(t, f"{m} ya pasó; {s} reanuda")
                self._free()
            elif done[m] and self._clear_of(q, m, goals[s]):
                # El maestro llegó y se queda, pero no ocupa el objetivo del
                # esclavo (caminos que se cruzan, no un punto compartido).
                # Sin esta comprobación el esclavo volvería mientras el
                # maestro sigue "agarrando" en el punto compartido y los
                # dos se turnarían en falso.
                self._log(t, f"{m} llegó y deja libre el objetivo de {s}; {s} reanuda")
                self._free()
            elif not done[m] and self._stalled(m) and not conflict and \
                    t - self.t_roles >= 2 * self.window * self.dt:
                # El maestro no avanza y el otro brazo NI SIQUIERA está cerca:
                # la causa no es entre brazos (alcance, entorno). Dar
                # prioridad no lo va a arreglar: pasa a la capa 3.
                self.state = "escalate"
                self._log(t, f"{m} no avanza y no es por el otro brazo (alcance o entorno): "
                             "hace falta la capa 3")
            elif not done[m] and self._stalled(m) and conflict and \
                    t - self.t_roles >= 2 * self.window * self.dt:
                # (el margen de 2 ventanas deja que el esclavo termine de
                # retirarse antes de juzgar al maestro)
                # El maestro tampoco avanza aunque el otro se retiró:
                # probar el orden inverso.
                if self.switches < self.max_switches:
                    self.switches += 1
                    self._log(t, f"{m} bloqueado aun con prioridad: se invierten los roles")
                    self._set_roles(t, master=s, goals=goals)
                else:
                    self.state = "escalate"
                    self._log(t, "sin salida con prioridades: hace falta la capa 3 (planificador)")

        return self._targets(goals), e

    def _assign(self, t, e, done, goals):
        self.assignments += 1
        if self.assignments > self.max_assignments:
            # Turnarse sin fin: los objetivos son incompatibles a la vez
            # (la configuración final chocaría) y la tarea no los libera.
            self.state = "escalate"
            self._log(t, "los brazos se turnan sin terminar: objetivos incompatibles, "
                         "hace falta la capa 3 (planificador/tarea)")
            return
        pending = [s for s in e if not done[s]]
        if len(pending) == 1:
            master = pending[0]      # el que ya llegó cede el espacio
        elif abs(e["left"] - e["right"]) < 1e-3:
            master = self.default_master
        else:
            master = min(pending, key=lambda s: e[s])
        self._log(t, f"bloqueo: pasa {master}, cede {OTHER[master]} "
                     f"(err izq {1000 * e['left']:.0f} mm, der {1000 * e['right']:.0f} mm)")
        self._set_roles(t, master=master, goals=goals)

    def _set_roles(self, t, master, goals):
        self.state = "yield"
        self.t_roles = t
        self.master, self.slave = master, OTHER[master]
        # Para saber si después la tarea mandó al maestro a otro lado.
        self.master_goal_at_yield = np.asarray(goals[master])[:3].copy()
        for h in self.hist.values():
            h.clear()

    def _free(self):
        self.state = "free"
        self.master = self.slave = None
        self.master_goal_at_yield = None
        for h in self.hist.values():
            h.clear()

    def _targets(self, goals):
        out = {s: np.asarray(goals[s], dtype=float) for s in ("left", "right")}
        if self.state == "yield":
            out[self.slave] = self.retreat[self.slave]
        return out


class TaskSequence:
    """Tarea de un brazo como lista de (pose, dwell): ir a la pose, quedarse
    `dwell` s dentro de tolerancia (p.ej. "agarrar") y pasar a la siguiente.
    Avanza con el error a SU objetivo, así que mientras el brazo cede no
    avanza: la tarea espera, no se salta pasos."""

    def __init__(self, steps, tol_pos=0.02):
        self.steps = [(np.asarray(p, dtype=float), float(d)) for p, d in steps]
        self.tol = tol_pos
        self.i = 0
        self.t_in = None
        self.t_done = None

    @property
    def done(self):
        return self.i >= len(self.steps)

    def goal(self):
        return self.steps[min(self.i, len(self.steps) - 1)][0]

    def update(self, t, err):
        if self.done:
            return
        if err < self.tol:
            if self.t_in is None:
                self.t_in = t
            if t - self.t_in >= self.steps[self.i][1]:
                self.i += 1
                self.t_in = None
                if self.done:
                    self.t_done = t
        else:
            self.t_in = None
