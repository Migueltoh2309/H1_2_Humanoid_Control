"""Lógica de visual servoing + agarre, sin ROS ni MuJoCo.

La usan igual el nodo en vivo (`QP_bimanual_avoidance.py`, `scenario:=grasp`)
y la evaluación en lazo cerrado (`demos/visual_servoing_grasp.py`), para que
lo que se mide en la batería sea exactamente lo que corre en el robot.
Diseño y resultados: VISUAL_SERVOING_PLAN.md.

    medida del centro de la fruta (torso_link, con su instante de captura)
      -> FruitTracker: Kalman de velocidad constante -> p(t), v
      -> GraspServoing: máquina de estados que, con la meta predicha, da
         la pose objetivo de cada muñeca, el feedforward de velocidad y el
         comando de la mano
      -> BimanualAvoidanceController.step(..., x_dot_ff=...)  (capa 1)

Es PBVS con la cámara fija en el torso [Chaumette & Hutchinson 2006]: el
error "muñeca deseada (función de la fruta vista) − muñeca actual (FK)" se
cierra en cada ciclo con la estimación nueva. La máquina de estados es la
parte hecha a mano que el LfD debería reemplazar (plan §7).
"""
import numpy as np

from h1_2_algoritms.vision import grasp_geometry as GG

SIDES = ("left", "right")


class FruitTracker:
    """Kalman de velocidad constante en 3D (estado [p, v]), con compuerta
    de innovación. Con la fruta quieta converge a v = 0; sobre la faja
    estima la velocidad de la faja sin conocerla, y `predict(t)` da la
    posición futura: la meta g = p(t_grasp) del manuscrito (§3.3).

    Compuerta ADAPTATIVA: acepta una innovación de hasta `gate` + 3σ de la
    predicción, y σ crece con el tiempo sin medidas. Con una compuerta fija
    (8 cm), una oclusión de ~4 s (el brazo pasando delante de la cámara con
    la faja a 0.05 m/s: 20 cm) dejaba el filtro rechazando todas las medidas
    nuevas para siempre (medido en vivo). Y si aun así se rechazan
    `max_rejections` medidas seguidas, se reinicia la pista con la última:
    la fruta está donde la cámara la ve, no donde el filtro la esperaba."""

    def __init__(self, sigma_meas=0.004, sigma_acc=0.05, gate=0.08, max_rejections=3):
        self.x = None
        self.P = None
        self.t = None
        self.r = sigma_meas ** 2
        self.q = sigma_acc ** 2
        self.gate = gate
        self.max_rejections = max_rejections
        self.n_updates = 0
        self.n_rejected = 0
        self.n_consecutive_rejected = 0
        self.n_reinit = 0

    def reset(self):
        self.x = self.P = self.t = None
        self.n_updates = self.n_rejected = self.n_consecutive_rejected = 0

    def _F(self, dt):
        F = np.eye(6)
        F[0:3, 3:6] = dt * np.eye(3)
        return F

    def _Q(self, dt):
        q = self.q
        Q = np.zeros((6, 6))
        Q[0:3, 0:3] = q * dt ** 4 / 4 * np.eye(3)
        Q[0:3, 3:6] = Q[3:6, 0:3] = q * dt ** 3 / 2 * np.eye(3)
        Q[3:6, 3:6] = q * dt ** 2 * np.eye(3)
        return Q

    def update(self, z, t):
        """z: centro medido (3,), t: instante de CAPTURA de la imagen [s]."""
        z = np.asarray(z, dtype=float)
        if self.x is None:
            self.x = np.hstack((z, np.zeros(3)))
            self.P = np.diag([self.r] * 3 + [0.1 ** 2] * 3)
            self.t = t
            self.n_updates = 1
            return True
        dt = max(t - self.t, 1e-4)
        F = self._F(dt)
        x = F @ self.x
        P = F @ self.P @ F.T + self._Q(dt)
        y = z - x[0:3]
        S = P[0:3, 0:3] + self.r * np.eye(3)
        gate = self.gate + 3.0 * np.sqrt(np.max(np.diag(S)))
        if np.linalg.norm(y) > gate:            # salto imposible -> outlier
            self.n_rejected += 1
            self.n_consecutive_rejected += 1
            if self.n_consecutive_rejected >= self.max_rejections:
                n_reinit = self.n_reinit + 1
                self.reset()
                self.n_reinit = n_reinit
                return self.update(z, t)
            return False
        self.n_consecutive_rejected = 0
        K = P[:, 0:3] @ np.linalg.inv(S)
        self.x = x + K @ y
        self.P = (np.eye(6) - K[:, 0:3] @ np.eye(3, 6)) @ P
        self.t = t
        self.n_updates += 1
        return True

    def predict(self, t):
        if self.x is None:
            return None, None
        dt = t - self.t
        return self.x[0:3] + dt * self.x[3:6], self.x[3:6].copy()


def grasp_point_err(side, q, xd):
    """Distancia entre el punto de agarre (centro de fruta en la mano) con
    la muñeca REAL (FK de q) y con la muñeca OBJETIVO xd = [p, quat wxyz].
    Combina error de posición y de orientación en la unidad que importa:
    con el punto a 0.18 m de la muñeca, 10° de orientación son ~3 cm aunque
    la muñeca esté en su sitio."""
    T = GG.FK[side](q)
    actual = T[0:3, 3] + T[0:3, 0:3] @ GG.GRASP_OFFSET[side]
    desired = np.asarray(xd[0:3]) + _quat_to_R(xd[3:7]) @ GG.GRASP_OFFSET[side]
    return float(np.linalg.norm(actual - desired))


def _quat_to_R(q):
    w, x, y, z = np.asarray(q, dtype=float) / (np.linalg.norm(q) + 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class GraspServoing:
    """Máquina de estados de un agarre con un brazo:

        WAIT      espera a que la fruta (predicha a `lookahead` s) entre en
                  la ventana alcanzable de un brazo -> elige ese brazo
        APPROACH  pre-agarre (`approach` m por detrás de la palma), siguiendo
                  la fruta predicha + feedforward de su velocidad
        DESCEND   baja a la pose de agarre; sigue corrigiendo con la visión
                  mientras la fruta se ve (cuando la mano la tapa, el Kalman
                  predice)
        CLOSE     cierra la mano en `close_time` s; la meta queda
                  comprometida (la fruta ya no se ve) y avanza con la
                  velocidad de la faja congelada en el compromiso
        LIFT      sube `lift_height` m en `lift_time` s
        RETREAT   lleva la mano (cerrada, con la fruta) a `home`: saca el
                  brazo del campo de la cámara para poder verificar
        DONE      agarre verificado: la cámara VE la fruta en la mano
        FAILED    la cámara ve la fruta donde quedó en la faja, o no la ve ni
                  en la mano ni en la faja al terminar el retiro (se cayó):
                  la mano se abre y vuelve a home
        MISSED    la fruta salió de la ventana del brazo antes de cerrar, o
                  la bajada no llegó al punto de agarre (no se cierra a
                  ciegas: cerrar lejos de la fruta la golpea)

    Verificación (`observe`), con evidencia POSITIVA: la mano levantada
    todavía tapa la fruta desde la cámara del torso (medido), así que "no
    verla en la faja" no prueba nada (en vivo, una fruta golpeada al piso
    pasaba como éxito). Durante el retiro:
      * medida a < `verify_tol` de donde estaría la fruta en la faja
        (posición al cerrar + velocidad × tiempo)            -> FAILED
      * `verify_min_hits` medidas a < `verify_hand_tol` del punto de agarre
        de la mano (FK de la muñeca)                          -> DONE
        (tolerancia del tamaño de la mano, no de la precisión del agarre:
        la mano sostiene la fruta donde la atrapó; medido en vivo: una
        fruta bien sujeta quedó a ~8 cm del punto nominal. Tras el retiro
        la mano está a > 20 cm de la faja, así que no hay ambigüedad.)
      * ninguna de las dos al pasar `verify_timeout` s       -> FAILED

    El brazo que no agarra se queda en `home`. Una fruta por ciclo:
    `reset()` rearma la máquina para la siguiente.
    """

    PHASES = ("WAIT", "APPROACH", "DESCEND", "CLOSE", "LIFT", "RETREAT", "DONE", "FAILED", "MISSED")

    def __init__(self, home, windows=None, belt_box=None, lookahead=1.5, exit_margin=0.03,
                 approach=0.08, tol_pregrasp=0.015, tol_grasp=0.008, min_approach_time=0.5,
                 descend_timeout=3.0, close_time=0.6, close_hold=0.8,
                 lift_height=0.12, lift_time=1.0, lift_hold=0.3, retreat_time=2.0,
                 verify_tol=0.04, verify_hand_tol=0.10, verify_min_hits=2, verify_timeout=3.0, close_max_err=0.02,
                 min_updates=3):
        self.home = {s: np.asarray(home[s], dtype=float) for s in SIDES}
        # Ventana alcanzable en y (torso_link) de cada brazo sobre la faja
        # (barrido de IK del plan §4).
        self.windows = windows or {"right": (-0.30, 0.05), "left": (-0.05, 0.30)}
        # Superficie útil de la faja en torso_link (x, z): una fruta fuera de
        # aquí (p.ej. empujada a la mesa, medido en vivo) no es alcanzable
        # y no se persigue, aunque su y caiga en la ventana de un brazo.
        self.belt_box = belt_box or {"x": (0.26, 0.47), "z": (-0.20, -0.12)}
        self.lookahead = lookahead
        self.exit_margin = exit_margin
        self.approach = approach
        self.tol_pregrasp = tol_pregrasp
        self.tol_grasp = tol_grasp
        self.min_approach_time = min_approach_time
        self.descend_timeout = descend_timeout
        self.close_time = close_time
        self.close_hold = close_hold
        self.lift_height = lift_height
        self.lift_time = lift_time
        self.lift_hold = lift_hold
        self.retreat_time = retreat_time
        self.verify_tol = verify_tol
        self.verify_hand_tol = verify_hand_tol
        self.verify_min_hits = verify_min_hits
        self.verify_timeout = verify_timeout
        self.close_max_err = close_max_err
        self.min_updates = min_updates
        self.reset()

    def reset(self):
        self.phase = "WAIT"
        self.side = None
        self.t_phase = 0.0
        self.fruit_ref = None      # meta comprometida al cerrar
        self.v_ref = np.zeros(3)
        self.events = {}           # t_start, t_close, q_close, fruit_close, err_close, t_done
        self._in_hand = None       # punto de agarre de la mano (FK), para verificar
        self._hits = 0

    def _enter(self, phase, t):
        self.phase, self.t_phase = phase, t

    def tracking(self):
        """¿Las medidas de la cámara son de la fruta en la faja (y deben ir
        al Kalman)? Deja de serlo desde que la mano la agarra."""
        return self.phase in ("WAIT", "APPROACH", "DESCEND", "MISSED")

    def observe(self, t, z):
        """Toda medida cruda de la fruta (torso_link, instante de captura).
        Solo se usa para verificar el agarre durante el retiro."""
        if self.phase != "RETREAT" or "fruit_close" not in self.events:
            return
        z = np.asarray(z)
        on_belt = self.events["fruit_close"] + (t - self.events["t_close"]) * self.v_ref
        if np.linalg.norm(z - on_belt) < self.verify_tol:
            self._fail(t, "la fruta sigue en la faja")
        elif self._in_hand is not None and np.linalg.norm(z - self._in_hand) < self.verify_hand_tol:
            self._hits += 1

    def _fail(self, t, why):
        self._enter("FAILED", t)
        self.events.update(t_failed=t, why=why)

    def step(self, t, q_meas, est, vel, n_updates):
        """q_meas: 14 (izq + der) MEDIDAS. est, vel: predicción del tracker
        para el instante t (o None). Devuelve un dict con:
            targets   {side: [x y z qw qx qy qz]} de las muñecas
            x_dot_ff  {side: v3 o None}
            hand      {side: 6 aperturas (HAND_*)}
            phase, side
        """
        q_meas = np.asarray(q_meas, dtype=float)
        targets = {s: self.home[s].copy() for s in SIDES}
        ff = {s: None for s in SIDES}
        hand = {s: GG.HAND_PRESHAPE.copy() for s in SIDES}
        seen = est is not None and n_updates >= self.min_updates

        on_belt = seen and self.belt_box["x"][0] <= est[0] <= self.belt_box["x"][1] and \
            self.belt_box["z"][0] <= est[2] <= self.belt_box["z"][1]
        if self.phase == "WAIT" and on_belt:
            y_future = est[1] + self.lookahead * vel[1]
            # Las ventanas se solapan en el centro: primero el brazo del lado
            # donde estará la fruta.
            for s in (("right", "left") if y_future < 0 else ("left", "right")):
                lo, hi = self.windows[s]
                if lo <= y_future <= hi:
                    self.side = s
                    self._enter("APPROACH", t)
                    self.events["t_start"] = t
                    break

        s = self.side
        if s is not None:
            q_side = q_meas[:7] if s == "left" else q_meas[7:]
            if self.phase in ("APPROACH", "DESCEND") and seen and not on_belt:
                self._enter("MISSED", t)
                self.events["why"] = "la fruta ya no está sobre la faja"
            if self.phase in ("APPROACH", "DESCEND") and seen:
                # Solo el borde hacia el que avanza la fruta: una fruta que
                # todavía está llegando (del otro lado) no se ha perdido.
                lo, hi = self.windows[s]
                if (vel[1] >= 0 and est[1] > hi + self.exit_margin) or \
                        (vel[1] < 0 and est[1] < lo - self.exit_margin):
                    self._enter("MISSED", t)

            if self.phase == "APPROACH":
                if seen:
                    targets[s] = GG.pregrasp_wrist_target(s, est, approach=self.approach)
                    ff[s] = vel
                    if (grasp_point_err(s, q_side, targets[s]) < self.tol_pregrasp
                            and t - self.t_phase > self.min_approach_time):
                        self._enter("DESCEND", t)
            elif self.phase == "DESCEND":
                if seen:
                    targets[s] = GG.grasp_wrist_target(s, est)
                    ff[s] = vel
                    err = grasp_point_err(s, q_side, targets[s])
                    if err < self.tol_grasp or t - self.t_phase > self.descend_timeout:
                        if err > self.close_max_err:
                            self._enter("MISSED", t)
                            self.events["why"] = f"la bajada no llegó al agarre ({1e3 * err:.0f} mm)"
                        else:
                            self._enter("CLOSE", t)
                            self.fruit_ref = est.copy()
                            self.v_ref = vel.copy()
                            self.events.update(t_close=t, q_close=q_side.copy(), fruit_close=est.copy(),
                                               err_close=err)
            if self.phase == "CLOSE":
                # La fruta sigue en la faja hasta que la mano la levante.
                fr = self.fruit_ref + (t - self.t_phase) * self.v_ref
                targets[s] = GG.grasp_wrist_target(s, fr)
                ff[s] = self.v_ref
                a = min(1.0, (t - self.t_phase) / self.close_time)
                hand[s] = GG.HAND_PRESHAPE + a * (GG.HAND_CLOSED - GG.HAND_PRESHAPE)
                if t - self.t_phase > self.close_hold:
                    self.fruit_ref = fr
                    self._enter("LIFT", t)
            elif self.phase == "LIFT":
                xg = GG.grasp_wrist_target(s, self.fruit_ref)
                xg[2] += min(self.lift_height, self.lift_height * (t - self.t_phase) / self.lift_time)
                targets[s] = xg
                hand[s] = GG.HAND_CLOSED.copy()
                if t - self.t_phase > self.lift_time + self.lift_hold:
                    self._enter("RETREAT", t)
            elif self.phase in ("RETREAT", "DONE"):
                hand[s] = GG.HAND_CLOSED.copy()     # targets[s] = home
                self._in_hand = GG.wrist_to_fruit(s, q_side)
                if self.phase == "RETREAT":
                    if self._hits >= self.verify_min_hits:
                        self._enter("DONE", t)
                        self.events["t_done"] = t
                    elif t - self.t_phase > self.retreat_time + self.verify_timeout:
                        self._fail(t, "no se ve la fruta ni en la mano ni en la faja (¿se cayó?)")
            elif self.phase == "FAILED":
                hand[s] = GG.HAND_OPEN.copy()
        return {"targets": targets, "x_dot_ff": ff, "hand": hand,
                "phase": self.phase, "side": self.side}
