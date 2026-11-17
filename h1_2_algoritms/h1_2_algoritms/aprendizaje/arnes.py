"""[COPIA para la tesis, 2026-11-16, de tareas/manipulacion/sim/arnes.py (tarea de manipulación, sin commit allá); la lógica es la
misma: si allá se corrige, copiarla de nuevo. La usa aprendizaje/cuerpo_pie.py.]

El arnés del laboratorio: la grúa de la que cuelga el H1-2 en las pruebas (solo simulador).

Portado de la referencia h1_2_sim_mujoco (scripts/sim2real/sim_robot_mujoco.py, SimRobot.gantry):
una polea fija encima del robot, el cable del cabrestante hasta el gancho y dos correas que
abrazan el cuello (justo encima de la placa superior del torso, una a cada lado) y suben junto a
la cabeza hasta el gancho, en V. Es una cuerda: SOLO TIRA, y se queda floja en cuanto las piernas
cargan el peso. Lo que hace:

  * de lado el robot cuelga del gancho (0.95 m por encima del centro de masas) y de frente/atrás
    oscila alrededor de la línea entre las dos correas (cuelga del cuello, 0.55 m por encima);
  * el cable tira del gancho hacia la polea: con el cable tenso el robot es un péndulo bajo la
    polea, que lo endereza y lo devuelve debajo de ella;
  * rozamiento de las correas en el cuello y del cable en la polea (amortiguamiento).

La fuerza va por xfrc_applied de torso_link. Solo se quita la que puso el propio arnés (el visor
también empuja por xfrc_applied, y en cada sync() lo pone a 0: entonces no hay nada que quitar). `bajar(m)` suelta cable (negativo: lo recoge), como la grúa.
"""
import numpy as np

RIGIDEZ, AMORTIGUAMIENTO = 50000.0, 2000.0   # cable + correas [N/m, N s/m]: 1.5 cm de estirado con 77 kg
CUELLO = np.array([0.0, 0.0, 0.49])           # donde abrazan las correas [m, marco de torso_link]
SEPARACION = 0.05                             # cada correa a +-y del cuello [m]
LARGO_CORREA, LARGO_CABLE = 0.43, 0.5         # correa cuello-gancho y cable gancho-polea al enganchar [m]
ALTO_GANCHO = float(np.sqrt(LARGO_CORREA ** 2 - SEPARACION ** 2))
AMORT_BALANCEO, AMORT_GIRO = 20.0, 5.0        # [N s/m] en el cuello y [N m s/rad] del giro, con el cable tenso
VELOCIDAD = 0.04                              # bajar / subir la grúa [m/s]


class Arnes:
    def __init__(self, mujoco, model, data, cuerpo="torso_link"):
        self.mj, self.m, self.d = mujoco, model, data
        self.torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cuerpo)
        self.peso = float(np.sum(model.body_mass[1:])) * float(-model.opt.gravity[2])
        self.enganchado = False
        self.tension = 0.0
        self._f = None
        self._total = None
        self._estirado_ant = None
        self._vel = np.zeros(6)
        self.pulley = None
        self.largo = LARGO_CABLE

    def fijar_peso(self, kg):
        self.peso = kg * float(-self.m.opt.gravity[2])

    # ------------------------------------------------------------ geometría
    def cuello(self, lado=0.0):
        """Donde abraza una correa (lado +1 izq, -1 der; 0: entre las dos), marco mundo."""
        R = self.d.xmat[self.torso].reshape(3, 3)
        return self.d.xpos[self.torso] + R @ (CUELLO + [0.0, lado * SEPARACION, 0.0])

    def gancho(self):
        R = self.d.xmat[self.torso].reshape(3, 3)
        c, eje = self.cuello(), R[:, 1]
        arriba = self.pulley - c
        arriba = arriba - (arriba @ eje) * eje       # el gancho solo gira alrededor de la línea de las correas
        n = np.linalg.norm(arriba)
        return c + ALTO_GANCHO * (arriba / n if n > 1e-6 else R[:, 2])

    # ------------------------------------------------------------ grúa
    def enganchar(self, estirado=0.0):
        """La polea encima del cuello y el cable justo tenso (+ `estirado`)."""
        self.enganchado = True
        self.pulley = self.cuello() + [0.0, 0.0, ALTO_GANCHO + LARGO_CABLE]
        self.largo = LARGO_CABLE - estirado
        self._estirado_ant = None

    def colgar(self):
        """Enganchado con el cable cargando todo el peso (al encender: colgado de la grúa)."""
        self.enganchar(self.peso / RIGIDEZ)

    def bajar(self, metros):
        self.largo = max(self.largo + metros, 0.05)

    def soltar(self):
        self.enganchado = False

    def carga(self):
        """Fracción del peso que lleva el cable."""
        return self.tension / self.peso if self.peso > 0 else 0.0

    def holgura(self):
        """Cable que sobra [m] (> 0 flojo; < 0 estirado)."""
        if not self.enganchado:
            return float("inf")
        return float(self.largo - np.linalg.norm(self.pulley - self.gancho()))

    # ------------------------------------------------------------ física (antes de cada mj_step)
    def aplicar(self, dt):
        d, m = self.d, self.m
        if self._f is not None:
            # solo si sigue ahí: el visor pasivo pone xfrc_applied a 0 en cada sync() (y añade el
            # empujón del ratón); restarla entonces dejaba la fuerza del cable AL REVÉS
            if np.allclose(d.xfrc_applied[self.torso], self._total):
                d.xfrc_applied[self.torso] -= self._f
            self._f = None
        self.tension = 0.0
        if not self.enganchado:
            return
        g = self.gancho()
        cable = self.pulley - g
        u = cable / np.linalg.norm(cable)
        estirado = float(np.linalg.norm(cable) - self.largo)
        ritmo = 0.0 if self._estirado_ant is None else (estirado - self._estirado_ant) / dt
        self._estirado_ant = estirado
        tension = max(0.0, RIGIDEZ * estirado + AMORTIGUAMIENTO * ritmo) if estirado > 0.0 else 0.0
        self.tension = tension
        if tension <= 0.0:
            return
        self.mj.mj_objectVelocity(m, d, self.mj.mjtObj.mjOBJ_BODY, self.torso, self._vel, 0)
        c, com = self.cuello(), d.xipos[self.torso]
        v_c = self._vel[3:] + np.cross(self._vel[:3], c - com)
        balanceo = -AMORT_BALANCEO * (v_c - (u @ v_c) * u)
        f = np.zeros(6)
        f[:3] = tension * u + balanceo
        f[3:] = np.cross(g - com, tension * u) + np.cross(c - com, balanceo) - AMORT_GIRO * self._vel[:3]
        d.xfrc_applied[self.torso] += f
        self._f = f
        self._total = d.xfrc_applied[self.torso].copy()

    # ------------------------------------------------------------ visor
    def dibujar(self, scn):
        """Polea, cable, gancho y las dos correas (naranja con el cable tenso, gris flojo)."""
        mj = self.mj
        if not self.enganchado or scn.ngeom + 8 > scn.maxgeom:
            return
        R = self.d.xmat[self.torso].reshape(3, 3)
        g = self.gancho()
        tenso = np.linalg.norm(self.pulley - g) >= self.largo
        if not tenso:                       # el gancho cuelga del cable, por encima de la cabeza
            g = self.pulley - [0.0, 0.0, self.largo]
            g[2] = max(g[2], self.cuello()[2] + 0.32)
        color = np.array((1.0, 0.5, 0.0, 1.0) if tenso else (0.6, 0.6, 0.6, 1.0), dtype=np.float32)

        def poner(tipo, tam, a, b=None, rgba=color):
            geom = scn.geoms[scn.ngeom]
            mj.mjv_initGeom(geom, tipo, np.full(3, tam), a if b is None else np.zeros(3), np.eye(3).ravel(),
                            np.asarray(rgba, dtype=np.float32))
            if b is not None:
                mj.mjv_connector(geom, tipo, tam, a, b)
            scn.ngeom += 1

        cap, esf = mj.mjtGeom.mjGEOM_CAPSULE, mj.mjtGeom.mjGEOM_SPHERE
        poner(esf, 0.04, self.pulley, rgba=(0.2, 0.2, 0.2, 1.0))
        poner(cap, 0.004, self.pulley, g)
        poner(esf, 0.02, g, rgba=(0.8, 0.1, 0.1, 1.0))
        for lado in (1.0, -1.0):            # cada correa sube junto a la cabeza
            junto = self.d.xpos[self.torso] + R @ (CUELLO + [0.0, 0.085 * lado, 0.17])
            poner(cap, 0.01, self.cuello(lado), junto)
            poner(cap, 0.01, junto, g)
