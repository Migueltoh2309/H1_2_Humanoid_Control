"""DMP discreta cartesiana (posición de la muñeca), relativa a la meta.

Es la primitiva de bajo nivel del LfD (docs/LFD_DMP.md). Se aprende en
coordenadas RELATIVAS a la meta, y = x − g:

    x_ref(t) = g(t) + y(t)        ẋ_ref(t) = ġ(t) + ẏ(t)

con g la pose de muñeca de agarre que da la percepción (grasp_geometry
sobre la fruta predicha por el Kalman). Así la misma primitiva sirve con la
fruta quieta (ġ = 0) y sobre la faja (ġ = velocidad estimada): lo que se
aprende es CÓMO se acerca la mano a la fruta, no a qué punto del espacio.
Es la variante "meta móvil" de [Koutras & Doulgeri 2020] escrita en el
frame de la meta.

Sistema de transformación (Pastor et al., ICRA 2009), por eje, con la meta
del sistema relativo en 0:

    τ ż = K (0 − y) − D z − K (0 − y0) s + K f(s)
    τ ẏ = z

f se suma en metros, SIN escalar por (g − y0): con el escalado clásico de
Ijspeert, un eje con desplazamiento demostrado casi nulo amplifica la forma
sin límite (o la invierte si cambia de signo); aquí la forma se conserva y
el atractor absorbe el cambio de inicio.

Fase LINEAL s = 1 − t/τ (con freno opcional por error de seguimiento,
"phase stopping" [Ijspeert et al. 2013]): con la exponencial la última
parte del movimiento llega tarde (lección del DMP del UR5, proyecto
cafetero). f(s) = s · Σ ψ_i w_i / Σ ψ_i, que se anula al final.

Paramétrica (opcional): los pesos dependen linealmente de los parámetros
de la tarea θ (aquí, el inicio relativo y0):  w(θ) = W0 + W1 θ. Sigue
siendo un problema de mínimos cuadrados lineal (DMP paramétrica,
[Matsubara, Hyon & Morimoto, Neural Netw. 2011]).
"""
import numpy as np

DIM = 3


class DMP:

    def __init__(self, n_basis=20, alpha=25.0, beta=None, parametric=False, ridge=1e-4):
        self.n = n_basis
        self.D = alpha
        self.K = alpha * (alpha / 4.0 if beta is None else beta)   # crítico: β = α/4
        self.parametric = parametric
        self.ridge = ridge
        # Centros uniformes en s (fase lineal) y ancho tal que vecinas se
        # crucen a ~0.5.
        self.c = np.linspace(0.0, 1.0, n_basis)
        self.h = 1.0 / (2 * (np.diff(self.c).mean() * 0.65) ** 2)
        self.W = None           # (n_feat, DIM)
        self.tau_demo = None
        self.theta_mean = np.zeros(DIM)
        self.theta_std = np.ones(DIM)

    # ------------------------------------------------------------------
    def _psi(self, s):
        s = np.atleast_1d(s)
        psi = np.exp(-self.h * (s[:, None] - self.c[None, :]) ** 2)
        return s[:, None] * psi / (psi.sum(axis=1, keepdims=True) + 1e-10)

    def _theta(self, y0):
        return (np.asarray(y0) - self.theta_mean) / self.theta_std

    def _features(self, s, y0):
        phi = self._psi(s)
        if not self.parametric:
            return phi
        th = self._theta(y0)
        # [φ, φ θ1, φ θ2, φ θ3]
        return np.hstack([phi] + [phi * th[k] for k in range(DIM)])

    def forcing(self, s, y0):
        return self._features(s, y0) @ self.W

    # ------------------------------------------------------------------
    def fit(self, demos):
        """demos: lista de (t, y, yd, ydd), con t desde 0 y y relativo a la
        meta (y[-1] ≈ 0). Una sola regresión con todas las demos juntas."""
        y0s = np.array([d[1][0] for d in demos])
        self.tau_demo = float(np.mean([d[0][-1] for d in demos]))
        if self.parametric:
            self.theta_mean = y0s.mean(axis=0)
            # Un eje con inicio casi constante entre demos no se normaliza
            # (amplificaría ruido: lección del UR5).
            sd = y0s.std(axis=0)
            self.theta_std = np.where(sd > 0.01, sd, 1.0)
        A, B = [], []
        for t, y, yd, ydd in demos:
            tau = t[-1]
            s = 1.0 - t / tau
            f = (tau ** 2 * ydd + self.D * tau * yd) / self.K + y - y[0] * s[:, None]
            A.append(self._features(s, y[0]))
            B.append(f)
        A, B = np.vstack(A), np.vstack(B)
        reg = self.ridge * np.eye(A.shape[1])
        self.W = np.linalg.solve(A.T @ A + reg, A.T @ B)
        return self

    # ------------------------------------------------------------------
    def rollout(self, y0, tau=None, dt=0.01, t_extra=0.0):
        """Reproducción abierta desde y0 (meta fija en 0)."""
        r = DMPRunner(self, y0, tau=tau)
        n = int(round((r.tau + t_extra) / dt))
        Y, Yd = [r.y.copy()], [r.z / r.tau]
        for _ in range(n):
            y, yd = r.step(dt)
            Y.append(y.copy())
            Yd.append(yd.copy())
        return np.arange(n + 1) * dt, np.array(Y), np.array(Yd)

    def save(self, path):
        np.savez(path, W=self.W, n=self.n, D=self.D, K=self.K, parametric=self.parametric,
                 tau_demo=self.tau_demo, theta_mean=self.theta_mean, theta_std=self.theta_std,
                 c=self.c, h=self.h)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        m = cls(n_basis=int(z["n"]), parametric=bool(z["parametric"]))
        m.D, m.K = float(z["D"]), float(z["K"])
        m.c, m.h = z["c"], float(z["h"])
        m.W, m.tau_demo = z["W"], float(z["tau_demo"])
        m.theta_mean, m.theta_std = z["theta_mean"], z["theta_std"]
        return m


class DMPRunner:
    """Integración en línea de una DMP, para el lazo de control.

    `step(dt, err)` avanza la fase y devuelve (y, ẏ) relativos a la meta.
    `err` (opcional) es el error de seguimiento |x_ref − x| de la muñeca:
    con `alpha_err` > 0 la fase se frena mientras el robot no sigue a la
    referencia (dampers activos, límite de velocidad), en vez de dejar que
    la referencia se escape [Ijspeert et al. 2013, §3.2]."""

    def __init__(self, dmp, y0, tau=None, alpha_err=0.0):
        self.dmp = dmp
        self.y0 = np.asarray(y0, dtype=float).copy()
        self.y = self.y0.copy()
        self.z = np.zeros(DIM)
        self.tau = float(tau if tau is not None else dmp.tau_demo)
        self.alpha_err = alpha_err
        self.s = 1.0

    def step(self, dt, err=0.0):
        m, tau = self.dmp, self.tau
        f = m.forcing(self.s, self.y0)[0]
        zd = (m.K * (0.0 - self.y) - m.D * self.z - m.K * (0.0 - self.y0) * self.s + m.K * f) / tau
        self.z = self.z + zd * dt
        self.y = self.y + self.z / tau * dt
        self.s = max(0.0, self.s - dt / tau / (1.0 + self.alpha_err * err))
        return self.y, self.z / tau

    @property
    def done(self):
        return self.s <= 0.0
