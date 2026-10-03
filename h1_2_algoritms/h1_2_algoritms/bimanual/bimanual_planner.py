"""Capa 3 del control bimanual: planificador global (sin ROS).

Las capas 1 (QP + velocity dampers) y 2 (prioridad entre brazos) son
locales: miran el instante actual. Cuando el camino directo a un objetivo
queda tapado —un obstáculo delante de la mano, o el otro brazo quieto en
medio—, el QP se detiene a distancia segura (mínimo local) y ninguna
prioridad lo arregla. Esta capa busca un CAMINO libre en el espacio
articular y se lo entrega a la capa 1 como puntos de paso. Es la
arquitectura "global + reactiva" de GGDRC (Geng et al., Biomimetics 2024) y
la que recomienda NEO (Haviland & Corke, RA-L 2021).

Piezas:

  ArmCollisionChecker  ¿es válida una postura q (7) de un brazo? Mismas
                       cápsulas que la capa 1, contra el otro brazo (fijo),
                       el torso y la nube de la cámara, con márgenes MAYORES
                       que los d_s de la capa 1 (40 vs 30 mm entre
                       brazos, 60 vs 50 mm con el entorno) y límites
                       articulares.
  goal_configurations  IK a la pose objetivo desde varias semillas; se
                       quedan solo las soluciones sin colisión.
  rrt_connect          RRT-Connect (Kuffner & LaValle, ICRA 2000) en 7 GDL.
  shortcut / densify   atajos aleatorios + remuestreo a pasos cortos.
  GlobalPlannerLayer   decide CUÁNDO planificar (un brazo no avanza), a qué
                       brazo, detecta objetivos imposibles y guía la
                       ejecución punto a punto.

Planificación PRIORIZADA (Erdmann & Lozano-Pérez, 1987): se planifica un
brazo con el otro congelado en su postura actual (tratado como obstáculo).
Encaja con la capa 2: el que no planifica espera quieto.
"""
import threading
import time
from collections import deque

import numpy as np

from h1_2_algoritms.bimanual.collision_model import (
    arm_capsules, closest_points_segments, points_to_segment,
    TORSO_CAPSULE, TORSO_PAIRS)
from h1_2_algoritms.movimiento.fk_functions import TF2xyzquat, fkine_arm_left_unitree, fkine_arm_right_unitree
from h1_2_algoritms.movimiento.ik_functions import ik_solve_limited

FK = {"left": fkine_arm_left_unitree, "right": fkine_arm_right_unitree}
OTHER = {"left": "right", "right": "left"}
IDX = {"left": slice(0, 7), "right": slice(7, 14)}

# Hombro de cada brazo en torso_link (origen del frame 1 con q = 0): para
# descartar de entrada los puntos de obstáculo fuera del alcance del brazo.
SHOULDER = {"left": np.array([0.0, 0.2095, 0.4336]), "right": np.array([0.0, -0.2095, 0.4336])}
ARM_REACH = 0.85   # [m] hombro -> punta de la mano, con holgura


# =====================================================================
# Verificador de colisiones de un brazo
# =====================================================================

class ArmCollisionChecker:
    """Validez de posturas de UN brazo con el otro brazo fijo.

    clearance(q) = holgura mínima (distancia − margen) sobre todos los pares
    vigilados; la postura es válida si clearance ≥ 0 y está dentro de los
    límites articulares."""

    def __init__(self, side, q_other, obstacles, q_min, q_max,
                 margin_self=0.04, margin_env=0.06):
        self.side = side
        self.q_min = np.asarray(q_min, dtype=float)
        self.q_max = np.asarray(q_max, dtype=float)
        self.m_self = margin_self
        self.m_env = margin_env
        self.caps_other = arm_capsules(OTHER[side], np.asarray(q_other, dtype=float))
        P = np.zeros((0, 3)) if obstacles is None else np.asarray(obstacles, dtype=float)
        if len(P):
            P = P[np.linalg.norm(P - SHOULDER[side], axis=1) < ARM_REACH]
        self.obstacles = P
        self.n_checks = 0
        self.relax = 0.0   # ver relax_for_start

    def clearance(self, q):
        self.n_checks += 1
        caps = arm_capsules(self.side, q)
        worst = np.inf
        for c in caps:
            for o in self.caps_other:
                pa, pb = closest_points_segments(c["a"], c["b"], o["a"], o["b"])
                worst = min(worst, np.linalg.norm(pa - pb) - c["r"] - o["r"] - self.m_self)
            if c["name"] in TORSO_PAIRS:
                _, ta, tb, tr = TORSO_CAPSULE
                pa, pb = closest_points_segments(c["a"], c["b"], ta, tb)
                worst = min(worst, np.linalg.norm(pa - pb) - c["r"] - tr - self.m_self)
            if len(self.obstacles):
                d, _ = points_to_segment(self.obstacles, c["a"], c["b"])
                worst = min(worst, float(d.min()) - c["r"] - self.m_env)
        return worst + self.relax

    def in_limits(self, q):
        return bool(np.all(q >= self.q_min - 1e-9) and np.all(q <= self.q_max + 1e-9))

    def valid(self, q):
        return self.in_limits(q) and self.clearance(q) >= 0.0

    def edge_valid(self, qa, qb, step=0.04):
        """Todas las posturas intermedias (cada `step` rad en la
        articulación que más se mueve) son válidas. qa se supone válida."""
        n = int(np.ceil(np.max(np.abs(qb - qa)) / step))
        for k in range(1, n + 1):
            if not self.valid(qa + (qb - qa) * (k / n)):
                return False
        return True

    def relax_for_start(self, q_start):
        """Si el brazo ya arranca por dentro del margen del planificador
        (la capa 1 lo deja llegar hasta d_s, que es menor), relaja los
        márgenes lo justo para que el inicio cuente como válido. Si no, el
        RRT no podría ni salir del punto de partida."""
        c = self.clearance(q_start)
        self.relax = max(0.0, -c + 0.005)
        return self.relax


# =====================================================================
# Configuraciones objetivo sin colisión
# =====================================================================

def goal_configurations(checker, xd, q_seed, n_seeds=8, max_goals=3, rng=None):
    """Soluciones de IK de la pose xd (7) que además son válidas para el
    verificador. Ordenadas de la más cercana a la más lejana de q_seed."""
    rng = np.random.default_rng(0) if rng is None else rng
    fk = FK[checker.side]
    sols = []
    for k in range(n_seeds):
        q0 = q_seed if k == 0 else rng.uniform(checker.q_min, checker.q_max)
        s = ik_solve_limited(fk, TF2xyzquat, q0, xd, checker.q_min, checker.q_max,
                             max_iters=150, restarts=1, seed=int(rng.integers(1 << 30)))
        if s["ok"] and checker.valid(s["q"]):
            if all(np.max(np.abs(s["q"] - o)) > 0.05 for o in sols):
                sols.append(s["q"])
                if len(sols) >= max_goals:
                    break
    sols.sort(key=lambda g: np.linalg.norm(g - q_seed))
    return sols


def pair_compatible(goals_l, goals_r, margin):
    """¿Existe algún par (objetivo izq, objetivo der) cuyas cápsulas no
    choquen entre sí? Si no, los objetivos son INCOMPATIBLES a la vez."""
    for gl in goals_l:
        cl = arm_capsules("left", gl)
        for gr in goals_r:
            cr = arm_capsules("right", gr)
            if all(np.linalg.norm(np.subtract(*closest_points_segments(a["a"], a["b"], b["a"], b["b"])))
                   - a["r"] - b["r"] >= margin for a in cl for b in cr):
                return gl, gr
    return None


# =====================================================================
# RRT-Connect
# =====================================================================

class _Tree:
    def __init__(self, root, cap=20000):
        self.q = np.zeros((cap, len(root)))
        self.parent = np.full(cap, -1, dtype=int)
        self.q[0] = root
        self.n = 1

    def nearest(self, q):
        return int(np.argmin(np.sum((self.q[:self.n] - q) ** 2, axis=1)))

    def add(self, q, parent):
        self.q[self.n] = q
        self.parent[self.n] = parent
        self.n += 1
        return self.n - 1

    def branch(self, i):
        out = []
        while i >= 0:
            out.append(self.q[i].copy())
            i = self.parent[i]
        return out


def rrt_connect(checker, q_start, q_goal, step=0.15, time_limit=5.0, rng=None):
    """Camino [q_start, ..., q_goal] o None si no lo encuentra a tiempo.

    Dos árboles, uno desde cada extremo. En cada iteración uno crece un paso
    hacia una muestra aleatoria (extend) y el otro intenta alcanzar el nodo
    nuevo en línea recta tantos pasos como pueda (connect). Cuando se tocan,
    hay camino. Después se alternan los papeles."""
    rng = np.random.default_rng(0) if rng is None else rng
    ta, tb = _Tree(q_start), _Tree(q_goal)
    t_end = time.perf_counter() + time_limit

    def extend(tree, target):
        i = tree.nearest(target)
        d = target - tree.q[i]
        dist = np.linalg.norm(d)
        reached = dist <= step
        q_new = target if reached else tree.q[i] + d * (step / dist)
        if checker.edge_valid(tree.q[i], q_new):
            return ("reached" if reached else "advanced"), tree.add(q_new, i)
        return "trapped", -1

    while time.perf_counter() < t_end and ta.n < ta.q.shape[0] - 1:
        q_rand = rng.uniform(checker.q_min, checker.q_max)
        status, ia = extend(ta, q_rand)
        if status != "trapped":
            target = ta.q[ia]
            while True:
                status_b, ib = extend(tb, target)
                if status_b != "advanced":
                    break
            if status_b == "reached":
                pa, pb = ta.branch(ia), tb.branch(ib)
                path = pa[::-1] + pb[1:]
                if np.allclose(path[0], q_goal):   # los árboles estaban intercambiados
                    path = path[::-1]
                return path
        ta, tb = tb, ta
    return None


def shortcut(checker, path, iters=150, rng=None):
    """Atajos aleatorios: si dos puntos del camino se ven en línea recta
    libre, se borra lo que hay entre ellos. Quita el zigzag típico del RRT."""
    rng = np.random.default_rng(0) if rng is None else rng
    path = list(path)
    for _ in range(iters):
        if len(path) < 3:
            break
        i, j = sorted(rng.choice(len(path), 2, replace=False))
        if j - i < 2:
            continue
        if checker.edge_valid(path[i], path[j]):
            path = path[:i + 1] + path[j:]
    return path


def densify(path, max_step=0.2):
    """Remuestrea para que entre puntos de paso consecutivos ninguna
    articulación se mueva más de max_step rad."""
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(np.ceil(np.max(np.abs(b - a)) / max_step)))
        out += [a + (b - a) * (k / n) for k in range(1, n + 1)]
    return out


def plan_arm(side, q, xd, obstacles, q_min, q_max, margin_self=0.04, margin_env=0.06,
             time_limit=5.0, seed=0):
    """Planifica el brazo `side` hasta la pose xd con el otro brazo fijo en
    su postura actual. Devuelve un dict con:
        status  "ok" | "goal_invalid" (no hay postura objetivo sin colisión:
                fuera de alcance o dentro de un obstáculo) | "goal_occupied"
                (habría postura válida, pero el otro brazo, quieto donde está,
                la ocupa) | "no_path" (hay objetivo, pero no se encontró camino)
        path    lista de q (7) o None
        info    tiempos, nodos, chequeos de colisión
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    q = np.asarray(q, dtype=float)
    checker = ArmCollisionChecker(side, q[IDX[OTHER[side]]], obstacles,
                                  q_min[IDX[side]], q_max[IDX[side]],
                                  margin_self, margin_env)
    q_start = q[IDX[side]]
    relax = checker.relax_for_start(q_start)
    goals = goal_configurations(checker, np.asarray(xd, dtype=float), q_start, rng=rng)
    info = {"relax": relax, "n_goals": len(goals)}
    if not goals:
        # ¿Es el otro brazo el que estorba? Se repite sin él: si así hay
        # postura válida, el objetivo es alcanzable pero está OCUPADO.
        caps_other, checker.caps_other = checker.caps_other, []
        free = goal_configurations(checker, np.asarray(xd, dtype=float), q_start,
                                   n_seeds=4, max_goals=1, rng=rng)
        checker.caps_other = caps_other
        info["t"] = time.perf_counter() - t0
        return {"status": "goal_occupied" if free else "goal_invalid", "path": None, "info": info}
    path = None
    budget = time_limit - (time.perf_counter() - t0)
    for g in goals[:3]:
        path = rrt_connect(checker, q_start, g, time_limit=max(0.5, budget / 3), rng=rng)
        if path is not None:
            break
    info["t_rrt"] = time.perf_counter() - t0
    if path is None:
        info["t"] = time.perf_counter() - t0
        info["checks"] = checker.n_checks
        return {"status": "no_path", "path": None, "info": info}
    raw = len(path)
    path = densify(shortcut(checker, path, rng=rng))
    info.update(t=time.perf_counter() - t0, checks=checker.n_checks, raw_nodes=raw,
                waypoints=len(path))
    return {"status": "ok", "path": path, "info": info}


# =====================================================================
# Capa 3: cuándo planificar y cómo ejecutar
# =====================================================================

class GlobalPlannerLayer:
    """Supervisa el avance de cada brazo hacia su objetivo de TAREA. Si uno
    no avanza (y no es porque la capa 2 lo mandó a ceder), planifica un
    camino para él con el otro congelado, y lo guía punto a punto.

    step(...) devuelve joint_targets = {"left": q7|None, "right": q7|None}
    para la capa 1 (None = la capa 1 sigue la pose de la mano normalmente).

    Con asynchronous=True la planificación corre en un hilo (nodo ROS: el
    lazo de control sigue a 20 Hz manteniendo al robot quieto y seguro
    mientras tanto). Con False es bloqueante (evaluación offline)."""

    def __init__(self, dt, q_min, q_max, tol_pos=0.02, stall_window=1.5,
                 stall_progress=0.01, wp_tol=0.05, margin_self=0.04, margin_env=0.06,
                 planning_time=5.0, follow_timeout=8.0, asynchronous=False, seed=0):
        self.dt = dt
        self.q_min, self.q_max = np.asarray(q_min), np.asarray(q_max)
        self.tol = tol_pos
        self.window = max(2, int(round(stall_window / dt)))
        self.stall_progress = stall_progress
        self.wp_tol = wp_tol
        self.m_self, self.m_env = margin_self, margin_env
        self.planning_time = planning_time
        self.follow_timeout = follow_timeout
        self.asynchronous = asynchronous
        self.seed = seed
        self.events = []
        self.plans = []            # info de cada planificación (para registro)
        self.reset()

    def reset(self):
        self.hist = {s: deque(maxlen=self.window) for s in ("left", "right")}
        self.last_goal = {s: None for s in ("left", "right")}
        self.failed = {s: None for s in ("left", "right")}     # objetivo donde falló
        self.plan = None           # {"side", "path", "i", "goal", "frozen", "t_wp"}
        self._job = None           # planificación en curso (hilo)

    @property
    def active(self):
        """True mientras planifica o sigue un camino (la capa 2 debe esperar)."""
        return self.plan is not None or self._job is not None

    def _log(self, t, msg):
        self.events.append((t, msg))

    def _stalled(self, s):
        h = self.hist[s]
        return len(h) == h.maxlen and (h[0] - h[-1]) < self.stall_progress

    # ------------------------------------------------------------ paso
    def step(self, t, q, goals, task_err, obstacles, l2_busy=False):
        """goals/task_err: objetivos de la tarea y error de cada muñeca a
        ellos. l2_busy: la capa 2 está arbitrando (un brazo cede a
        propósito). Mientras tanto la capa 3 no dispara y descarta su
        historial: un brazo que espera en retirada "no avanza" porque así se
        le pidió, y planificar en ese momento (con el otro brazo a medio
        camino) da diagnósticos falsos. Solo actúa con la capa 2 libre o
        cuando ésta ya escaló ("hace falta la capa 3")."""
        q = np.asarray(q, dtype=float)
        for s in ("left", "right"):
            g = np.asarray(goals[s])[:3]
            if self.last_goal[s] is not None and np.linalg.norm(g - self.last_goal[s]) > 1e-3:
                self.hist[s].clear()
                self.failed[s] = None
                if self.plan is not None and self.plan["side"] == s:
                    self._log(t, f"{s}: la tarea cambió de objetivo, se descarta el camino")
                    self.plan = None
            self.last_goal[s] = g.copy()
            self.hist[s].append(task_err[s])

        # ---- planificación en curso (asíncrona) ----
        if self._job is not None:
            if not self._job["done"].is_set():
                return self._hold(q, self._job["side"])
            self._install(t, q, self._job)
            self._job = None

        # ---- siguiendo un camino ----
        if self.plan is not None:
            return self._follow(t, q)

        # ---- ¿hace falta planificar? ----
        if l2_busy:
            for h in self.hist.values():
                h.clear()
            return {"left": None, "right": None}
        pending = [s for s in ("left", "right")
                   if task_err[s] > self.tol and self._stalled(s)
                   and not self._failed_here(s, goals)]
        if not pending:
            return {"left": None, "right": None}
        # Si los dos están trabados, primero: ¿son compatibles sus objetivos?
        if len(pending) == 2 and not self._pair_ok(t, q, goals):
            return {"left": None, "right": None}
        side = min(pending, key=lambda s: task_err[s])     # como la capa 2
        self._start(t, q, side, goals[side], obstacles)
        if self._job is not None:
            return self._hold(q, side)
        return self._follow(t, q) if self.plan is not None else {"left": None, "right": None}

    # ------------------------------------------------------------ internos
    def _failed_here(self, s, goals):
        f = self.failed[s]
        return f is not None and np.linalg.norm(np.asarray(goals[s])[:3] - f) < 1e-3

    def _pair_ok(self, t, q, goals):
        rng = np.random.default_rng(self.seed)
        sols = {}
        for s in ("left", "right"):
            ch = ArmCollisionChecker(s, q[IDX[OTHER[s]]], None, self.q_min[IDX[s]],
                                     self.q_max[IDX[s]], self.m_self, self.m_env)
            ch.caps_other = []          # el otro brazo NO cuenta: se prueba el par después
            sols[s] = goal_configurations(ch, np.asarray(goals[s], dtype=float), q[IDX[s]], rng=rng)
        if sols["left"] and sols["right"] and pair_compatible(sols["left"], sols["right"], self.m_self):
            return True
        for s in ("left", "right"):
            self.failed[s] = np.asarray(goals[s])[:3].copy()
        alone = [s for s in ("left", "right") if not sols[s]]
        if alone:
            self._log(t, "objetivo INALCANZABLE para " + " y ".join(alone) + ": ni solo, sin el "
                         "otro brazo, tiene una postura final válida (fuera de alcance o dentro "
                         "de un obstáculo); se mantienen seguros donde están")
        else:
            self._log(t, "objetivos INCOMPATIBLES: cada uno es alcanzable solo, pero no existe "
                         "un par de posturas finales sin choque entre brazos; se mantienen "
                         "seguros donde están")
        return False

    def _start(self, t, q, side, xd, obstacles):
        self._log(t, f"{side} no avanza: planificando camino global (RRT-Connect)")
        job = {"side": side, "goal": np.asarray(xd)[:3].copy(), "frozen": q[IDX[OTHER[side]]].copy(),
               "t0": t, "done": threading.Event(), "result": None}
        args = (side, q.copy(), np.asarray(xd, dtype=float),
                None if obstacles is None else np.array(obstacles, copy=True),
                self.q_min, self.q_max, self.m_self, self.m_env, self.planning_time,
                self.seed + len(self.plans))

        def work():
            job["result"] = plan_arm(*args)
            job["done"].set()

        if self.asynchronous:
            self._job = job
            threading.Thread(target=work, daemon=True).start()
        else:
            work()
            self._install(t, q, job)

    def _install(self, t, q, job):
        r = job["result"]
        side = job["side"]
        self.plans.append({"t": t, "side": side, "status": r["status"], **r["info"]})
        if r["status"] != "ok":
            self.failed[side] = job["goal"]
            why = {"goal_invalid": "no hay postura objetivo sin colisión (objetivo dentro de "
                                   "un obstáculo o fuera de alcance)",
                   "goal_occupied": "el objetivo es alcanzable, pero lo ocupa el otro brazo "
                                    "en su postura actual (objetivos incompatibles a la vez)",
                   "no_path": "no se encontró camino en el tiempo disponible"}[r["status"]]
            self._log(t, f"{side}: planificación fallida — {why}")
            return
        self._log(t, f"{side}: camino de {r['info']['waypoints']} puntos en "
                     f"{1000 * r['info']['t']:.0f} ms ({r['info']['checks']} chequeos de colisión)")
        self.plan = {"side": side, "path": r["path"], "i": 1, "goal": job["goal"],
                     "frozen": job["frozen"], "t_wp": t}

    def _hold(self, q, side):
        """Mientras se planifica: los dos brazos quietos en su postura."""
        return {"left": q[IDX["left"]].copy(), "right": q[IDX["right"]].copy()}

    def _follow(self, t, q):
        p = self.plan
        side = p["side"]
        qs = q[IDX[side]]
        path = p["path"]
        # avanzar mientras el punto actual ya esté alcanzado
        while p["i"] < len(path) - 1 and np.max(np.abs(qs - path[p["i"]])) < self.wp_tol:
            p["i"] += 1
            p["t_wp"] = t
        if p["i"] == len(path) - 1 and np.max(np.abs(qs - path[-1])) < self.wp_tol:
            self._log(t, f"{side}: camino completado; vuelve el control por la pose de la mano")
            self.plan = None
            for h in self.hist.values():
                h.clear()
            return {"left": None, "right": None}
        if t - p["t_wp"] > self.follow_timeout:
            self._log(t, f"{side}: el seguimiento del camino no avanza; se abandona")
            self.failed[side] = p["goal"]
            self.plan = None
            return {"left": None, "right": None}
        out = {"left": None, "right": None}
        out[side] = path[p["i"]]
        out[OTHER[side]] = p["frozen"]
        return out
