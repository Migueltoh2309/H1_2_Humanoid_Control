"""Política de ALTO NIVEL sobre la biblioteca de habilidades (fase B del
LfD, docs/LFD_BC.md): cuando un brazo queda libre, ¿qué fruta agarra, o
espera?

Estado simbólico (lo arma el lazo, demos/bc_faja.py):

    {"t": t, "tau": duración del alcance,
     "cands": {pista: (p, v)}  frutas sobre la faja, sin asignar,
     "arms": {side: {"busy", "remaining", "target"}}}

Acción para el brazo libre `side`: una pista de `cands` o None (esperar).

Tres políticas con la misma interfaz `act(state, side)`:

  * FifoPolicy    la regla voraz de GraspServoing (línea base de la tesis):
                  la fruta más antigua cuya posición dentro de `lookahead` s
                  cae en la ventana del brazo.
  * ExpertPolicy  el DEMOSTRADOR: planifica sobre las frutas visibles con el
                  modelo de tiempos de las habilidades (cuándo entra/sale cada
                  fruta de la ventana de cada brazo, cuánto dura un ciclo) y
                  elige la asignación fruta→brazo que agarra más frutas, sin
                  que los dos brazos agarren cerca a la vez. Más adelante lo
                  reemplazan las demostraciones humanas.
  * BCPolicy      behavior cloning del experto: una red puntúa cada fruta
                  candidata (y la opción esperar) a partir de características
                  RELATIVAS AL BRAZO; softmax sobre las opciones.
"""
import itertools

import numpy as np

from h1_2_algoritms.aprendizaje.demostraciones import to_right
from h1_2_algoritms.aprendizaje.habilidades import TIMING, cycle_time

# Ventanas alcanzables en y (torso_link) — las de GraspServoing.
WINDOWS = {"right": (-0.30, 0.05), "left": (-0.05, 0.30)}
OTHER = {"left": "right", "right": "left"}

# Mapa de alcance dependiente de x (IK: error < 8 mm y < 5°, intervalo
# contiguo más cercano al centro de la ventana). Para x >= 0.40 no hay
# solución: con el robot donde está, el lado lejano de la faja no se
# alcanza. APAGADO por omisión (USE_REACH_MAP); el derecho, y en la
# convención del torso; el izquierdo es su reflejo.
REACH_MAP_RIGHT = [
    (0.26, (-0.35, 0.12)), (0.28, (-0.34, 0.11)), (0.30, (-0.32, 0.09)),
    (0.32, (-0.30, 0.07)), (0.34, (-0.28, 0.05)), (0.36, (-0.25, 0.02)),
    (0.38, (-0.20, -0.03)),
]
REACH_MAP_XMAX = 0.39          # más allá: inalcanzable
USE_REACH_MAP = False


def window(side, x=None):
    """Ventana en y del brazo para una fruta en x: la nominal (WINDOWS, que
    separa a los dos brazos en el centro) intersecada, si USE_REACH_MAP,
    con el mapa de alcance interpolado en x. None si no hay alcance."""
    lo0, hi0 = WINDOWS[side]
    if not USE_REACH_MAP or x is None:
        return lo0, hi0
    if x > REACH_MAP_XMAX:
        return None
    xs = [r[0] for r in REACH_MAP_RIGHT]
    xc = min(max(x, xs[0]), xs[-1])
    lo = float(np.interp(xc, xs, [r[1][0] for r in REACH_MAP_RIGHT]))
    hi = float(np.interp(xc, xs, [r[1][1] for r in REACH_MAP_RIGHT]))
    if side == "left":
        lo, hi = -hi, -lo
    lo, hi = max(lo, lo0), min(hi, hi0)
    return (lo, hi) if lo < hi else None


# =====================================================================
# Modelo de factibilidad temporal
# =====================================================================
def reach_time(tau, timing=TIMING):
    return tau + timing["reach_extra"]


def start_interval(side, p, v, t, tau, margin=0.02, track_band=0.08, timing=TIMING):
    """[s_min, s_max]: instantes de inicio del alcance para los que

      * la fruta, al agarrarla (inicio + duración del alcance), está dentro
        de la ventana del brazo (con `margin`), y
      * al EMPEZAR ya está a < `track_band` de la ventana: la DMP sigue a la
        fruta (meta móvil), y empezar con la fruta fuera del alcance del
        brazo lo lleva a los límites articulares (medido: abortos a
        260-290 mm en los episodios).

    Vacío si s_min > s_max."""
    w = window(side, float(p[0]))
    if w is None:
        return np.inf, -np.inf
    lo0, hi0 = w
    lo, hi = lo0 + margin, hi0 - margin
    if lo > hi:
        return np.inf, -np.inf
    TR = reach_time(tau, timing)
    vy = float(v[1])
    if abs(vy) < 1e-3:
        return (-np.inf, np.inf) if lo <= p[1] <= hi else (np.inf, -np.inf)
    if vy > 0:
        a = max(t + (lo - p[1]) / vy - TR, t + (lo0 - track_band - p[1]) / vy)
        b = t + (hi - p[1]) / vy - TR
    else:
        a = max(t + (hi - p[1]) / vy - TR, t + (hi0 + track_band - p[1]) / vy)
        b = t + (lo - p[1]) / vy - TR
    return a, b


# =====================================================================
# Políticas
# =====================================================================
class FifoPolicy:
    name = "fifo"

    def __init__(self, lookahead=1.5):
        self.lookahead = lookahead

    def act(self, state, side):
        lo, hi = WINDOWS[side]
        best, best_key = None, None
        for tid, (p, v) in state["cands"].items():
            y_future = p[1] + self.lookahead * v[1]
            if lo <= y_future <= hi:
                key = p[1] * np.sign(v[1] if abs(v[1]) > 1e-3 else 1.0)    # más abajo en la faja = más antigua
                if best_key is None or key > best_key:
                    best, best_key = tid, key
        return best


class ExpertPolicy:
    """Planificación por enumeración de asignaciones (≤ 3^K) sobre las K
    frutas más urgentes, con el modelo de tiempos:

      * cada brazo atiende sus frutas por fecha límite (EDF);
      * una fruta se empieza en max(brazo libre, s_min) y es factible si
        eso es ≤ s_max;
      * dos agarres de brazos distintos a < `sep_y` en y y a < `sep_t` en el
        tiempo son incompatibles (se estorbarían en el centro: el QP lo
        evita, pero frenando a los dos);
      * objetivo: más frutas, después empezar antes.

    La acción del brazo libre es su primera fruta del mejor plan si se
    puede empezar ya (`now_tol`), si no, esperar."""
    name = "experto"

    def __init__(self, K=6, sep_y=0.20, sep_t=2.0, now_tol=0.15, timing=TIMING):
        self.K, self.sep_y, self.sep_t, self.now_tol, self.timing = K, sep_y, sep_t, now_tol, timing

    def plan(self, state):
        t, tau = state["t"], state["tau"]
        cyc = cycle_time(tau, self.timing)
        TR = reach_time(tau, self.timing)
        iv = {(s, tid): start_interval(s, p, v, t, tau, timing=self.timing)
              for tid, (p, v) in state["cands"].items() for s in ("left", "right")}
        # Frutas factibles para algún brazo, las más urgentes primero.
        cands = [tid for tid in state["cands"]
                 if any(iv[(s, tid)][0] <= iv[(s, tid)][1] and iv[(s, tid)][1] >= t for s in ("left", "right"))]
        cands.sort(key=lambda k: max(iv[("left", k)][1], iv[("right", k)][1]))
        cands = cands[:self.K]
        free0 = {s: t + state["arms"][s]["remaining"] for s in ("left", "right")}
        fixed = []   # agarres ya comprometidos de un brazo ocupado: (t_agarre, y)
        for s in ("left", "right"):
            a = state["arms"][s]
            if a["busy"] and a.get("target") is not None and a.get("t_grasp") is not None:
                fixed.append((s, a["t_grasp"], a["target"][1]))
        best, best_score = {"left": [], "right": []}, (-1, 0.0)
        for assign in itertools.product((None, "left", "right"), repeat=len(cands)):
            sched = {"left": [], "right": []}
            for tid, s in zip(cands, assign):
                if s is not None:
                    sched[s].append(tid)
            grasps = list(fixed)
            ok, total_start = True, 0.0
            plan = {}
            for s in ("left", "right"):
                free = free0[s]
                seq = sorted(sched[s], key=lambda k: iv[(s, k)][1])
                plan[s] = []
                for tid in seq:
                    a, b = iv[(s, tid)]
                    st = max(free, a)
                    if st > b:
                        ok = False
                        break
                    p, v = state["cands"][tid]
                    grasps.append((s, st + TR, p[1] + v[1] * (st - t + TR)))
                    plan[s].append((tid, st))
                    total_start += st - t
                    free = st + cyc
                if not ok:
                    break
            if not ok:
                continue
            for (s1, g1, y1), (s2, g2, y2) in itertools.combinations(grasps, 2):
                if s1 != s2 and abs(g1 - g2) < self.sep_t and abs(y1 - y2) < self.sep_y:
                    ok = False
                    break
            if not ok:
                continue
            n = len(plan["left"]) + len(plan["right"])
            score = (n, -total_start)
            if score > best_score:
                best, best_score = plan, score
        return best

    def act(self, state, side):
        plan = self.plan(state)
        if plan[side]:
            tid, st = plan[side][0]
            if st <= state["t"] + self.now_tol:
                return tid
        return None


# =====================================================================
# Características (relativas al brazo) y política aprendida
# =====================================================================
def _clip(x, lo=-5.0, hi=15.0):
    return float(np.clip(x, lo, hi))


def features(state, side):
    """(ids, X_cand (n×d), ctx (dc)) en la convención del brazo derecho
    (la izquierda se refleja en y, como la DMP): la misma red sirve para
    los dos brazos."""
    t, tau = state["t"], state["tau"]
    TR = reach_time(tau)
    o = OTHER[side]
    arm, oth = state["arms"][side], state["arms"][o]
    ids = list(state["cands"].keys())
    rows = []
    iv_me = {k: start_interval(side, *state["cands"][k], t, tau) for k in ids}
    iv_ot = {k: start_interval(o, *state["cands"][k], t, tau) for k in ids}
    feas_me = [iv_me[k][0] <= t + 0.15 and t <= iv_me[k][1] for k in ids]
    feas_ot_later = [iv_ot[k][0] <= iv_ot[k][1] and iv_ot[k][1] >= t + oth["remaining"] for k in ids]
    urg = np.argsort(np.argsort([iv_me[k][1] for k in ids])) if ids else []
    o_target = oth.get("target")
    for j, k in enumerate(ids):
        p, v = state["cands"][k]
        pr, vr = to_right(side, p), to_right(side, v)
        yg = pr[1] + vr[1] * TR
        dy_other = 1.0 if o_target is None else abs(p[1] + v[1] * TR - o_target[1])
        rows.append([
            pr[0] - 0.35, pr[1], 10 * vr[1], yg,
            _clip(iv_me[k][0] - t), _clip(iv_me[k][1] - t),
            float(feas_me[j]),
            _clip(iv_ot[k][0] - t), _clip(iv_ot[k][1] - t),
            float(feas_ot_later[j]),
            min(dy_other, 1.0),
            urg[j] / max(1, len(ids) - 1) if len(ids) > 1 else 0.0,
        ])
    ctx = [
        len(ids) / 6.0,
        float(oth["busy"]), _clip(oth["remaining"], 0, 15) / 10,
        float(sum(feas_me)) / 3.0, float(sum(feas_ot_later)) / 3.0,
        min([_clip(iv_me[k][1] - t) for k in ids], default=15.0) / 10,
        float(arm["busy"]),
    ]
    return ids, np.array(rows, dtype=np.float32).reshape(len(ids), N_CAND_FEAT), np.array(ctx, dtype=np.float32)


N_CAND_FEAT = 12
N_CTX_FEAT = 7


def build_net(hidden=64):
    import torch.nn as nn

    class Scorer(nn.Module):
        """s(fruta) = MLP([φ_fruta, φ_ctx]),  s(esperar) = MLP(φ_ctx).
        Invariante a permutaciones y a la cantidad de frutas."""

        def __init__(self):
            super().__init__()
            self.cand = nn.Sequential(nn.Linear(N_CAND_FEAT + N_CTX_FEAT, hidden), nn.ReLU(),
                                      nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            self.wait = nn.Sequential(nn.Linear(N_CTX_FEAT, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        def forward(self, X, ctx):
            """X: (n, d), ctx: (dc,) -> logits (n + 1,), el último = esperar."""
            import torch
            c = ctx.unsqueeze(0).expand(X.shape[0], -1)
            s = self.cand(torch.cat([X, c], dim=1)).squeeze(1)
            return torch.cat([s, self.wait(ctx.unsqueeze(0)).squeeze(1)])

        def forward_batch(self, X, mask, ctx):
            """X: (B, n, d) con relleno, mask: (B, n) True = fruta real,
            ctx: (B, dc) -> logits (B, n + 1); el relleno queda en −inf."""
            import torch
            c = ctx.unsqueeze(1).expand(-1, X.shape[1], -1)
            s = self.cand(torch.cat([X, c], dim=2)).squeeze(2)
            s = s.masked_fill(~mask, float("-inf"))
            return torch.cat([s, self.wait(ctx)], dim=1)

    return Scorer()


class BCPolicy:
    name = "bc"

    def __init__(self, net):
        self.net = net.eval()

    @classmethod
    def load(cls, path):
        import torch
        net = build_net()
        net.load_state_dict(torch.load(path, map_location="cpu"))
        return cls(net)

    def act(self, state, side):
        import torch
        ids, X, ctx = features(state, side)
        with torch.no_grad():
            logits = self.net(torch.from_numpy(X), torch.from_numpy(ctx))
        k = int(torch.argmax(logits))
        return ids[k] if k < len(ids) else None
