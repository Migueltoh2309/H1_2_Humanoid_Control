#!/usr/bin/env python3
"""Hito 2 — curva de pasos y giro del cuerpo frente a la duración de un movimiento del brazo
(preguntas 6.3.1 y 6.3.2). Como prueba_brazos_armsdk.py de Codigos, pero barriendo duraciones.

El MISMO movimiento (de la pose de observación a una pose de alcance sobre la mesa y vuelta,
con perfil de mínima sacudida) se repite con duraciones de MAYOR a MENOR (30, 20, 10, 8 s; 5 s
solo con --permitir-5s: la sec. 10 prohíbe bajar de 8 s fuera de esta medida). Por cada una:
  * giro máximo del yaw de la IMU, roll y pitch máximos;
  * pasos: en el simulador, desplazamiento de los pies (verdad de terreno); en el robot, los
    cuenta el operador con las huellas de cinta (lo pregunta el programa);
  * lo que lo provoca: velocidad y aceleración pico de las juntas (del perfil) y
    desplazamiento del centro de masas del robot entero (modelo MuJoCo con las juntas medidas).
Para en la primera duración que da un paso o gira más de 2 grados (o la corta el vigilante).

    python3 herramientas/curva_pasos.py --sim --lado izq
    python3 herramientas/curva_pasos.py --iface eth0 --lado izq      # robot

En el simulador el robot está de pie con las piernas rígidas (sin la política de Unitree): el
cuerpo recibe el empuje real del brazo pero no puede dar pasos. La curva de pasos que vale es
la del robot. El simulador da la parte física: cuánto se mueve el centro de masas y con qué
aceleraciones, que es la respuesta a "¿influye más la velocidad, la aceleración o el
desplazamiento del centro de masas?" cuando se cruza con la curva del robot.
"""
import argparse
import json
import os
import sys
import threading
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
from manip import cinematica as C  # noqa: E402
from manip.config import _leer, cargar  # noqa: E402

DATOS = os.path.join(RAIZ, "datos", "curva_pasos")


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


class PiesSim:
    """Verdad de terreno de los pies (solo simulador): /sim/ground_truth/*_ankle_roll_link."""

    def __init__(self):
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from rclpy.executors import SingleThreadedExecutor
        if not rclpy.ok():
            rclpy.init()
        self.nodo = rclpy.create_node("manip_pies")
        self.p = {}
        for lado in ("left", "right"):
            self.nodo.create_subscription(
                PoseStamped, f"/sim/ground_truth/{lado}_ankle_roll_link",
                lambda m, k=lado: self.p.__setitem__(k, np.array([m.pose.position.x, m.pose.position.y])), 5)
        self.ex = SingleThreadedExecutor()
        self.ex.add_node(self.nodo)
        self.hilo = threading.Thread(target=self.ex.spin, daemon=True)
        self.hilo.start()

    def cerrar(self):
        import rclpy
        self.ex.shutdown()
        self.hilo.join(timeout=2.0)
        self.nodo.destroy_node()
        rclpy.try_shutdown()

    def leer(self):
        return {k: v.copy() for k, v in self.p.items()}


class CentroMasas:
    """Centro de masas del robot entero con las juntas medidas (modelo MuJoCo de la escena)."""

    def __init__(self, xml):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.d = mujoco.MjData(self.m)
        k = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_KEY, "de_pie")
        if k >= 0:
            mujoco.mj_resetDataKeyframe(self.m, self.d, k)
        self.adr = {lado: [int(self.m.jnt_qposadr[self.m.joint(n).id]) for n in C.NOMBRES_JUNTAS[lado]]
                    for lado in C.LADOS}
        self.pelvis = self.m.body("pelvis").id

    def com(self, q27):
        from manip.robot_io import INDICES
        for lado in C.LADOS:
            self.d.qpos[self.adr[lado]] = q27[INDICES[lado]]
        self.mj.mj_kinematics(self.m, self.d)
        self.mj.mj_comPos(self.m, self.d)
        return self.d.subtree_com[self.pelvis].copy() - self.d.xpos[self.pelvis]


def pose_alcance(lado, cfg):
    """Pose de alcance típica: palma abajo sobre la zona, a 15 cm de la mesa, lo más lejos."""
    from manip.estacion import Estacion
    est = Estacion(cfg)
    s = 1.0 if lado == "izq" else -1.0
    p = [est.zona.cx + 0.05, s * 0.12, est.z_mesa + 0.15 - est.torso_z]
    R = C.rotacion_agarre(lado, "arriba", 0.0, -30.0 * s)
    q, ep, eo, ok = C.ik(lado, p, R)
    if not ok:
        raise SystemExit("no hay IK para la pose de alcance")
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--lado", choices=C.LADOS, default="izq")
    ap.add_argument("--duraciones", type=float, nargs="+", default=[30.0, 20.0, 10.0, 8.0])
    ap.add_argument("--permitir-5s", action="store_true", help="añade 5 s (solo para esta medida)")
    ap.add_argument("--escena", default=os.path.join(RAIZ, "sim", "estaciones", "estacion_n0_s0.xml"),
                    help="MJCF para el centro de masas (el de la estación)")
    a = ap.parse_args()
    from manip import robot_io as IO
    from manip.brazo import Brazo, Parada
    from manip.seguridad import Vigilante
    cfg = cargar()
    cfg_brazo = dict(_leer(os.path.join(RAIZ, "config", "brazo.yaml")))
    poses = _leer(os.path.join(RAIZ, "config", "poses.yaml"))
    duraciones = sorted(set(a.duraciones + ([5.0] if a.permitir_5s else [])), reverse=True)
    cfg_brazo["duracion_min"] = min(duraciones)       # esta herramienta mide justo eso
    IO.iniciar_dds(a.sim, a.iface)
    estado = IO.Estado()
    if not estado.esperar():
        sys.exit("no llega rt/lowstate")
    fsm = IO.LectorFsm()
    if fsm.leer() != 201:
        sys.exit("el robot tiene que estar en FSM 201")
    pies = PiesSim() if a.sim else None
    cm = CentroMasas(a.escena)
    obs = {lado: np.array(poses["observacion"][lado]) for lado in C.LADOS}
    q_alc = pose_alcance(a.lado, cfg)
    dq = float(np.max(np.abs(q_alc - obs[a.lado])))
    if not IO.confirmar("CURVA", a.sim, log):
        sys.exit("cancelado")
    q_ini = {lado: estado.brazo(lado) for lado in C.LADOS}
    brazo = Brazo(estado, cfg_brazo)
    vig = Vigilante(estado, fsm, brazo)
    filas = []
    try:
        log("rampa de entrada (12 s) y pose de observación")
        brazo.rampa_entrada(comprobar=vig)
        vig.mirar_seguimiento = True
        brazo.mover(obs, comprobar=vig)
        time.sleep(3.0)
        for T in duraciones:
            if 1.875 * dq / T > 0.5 + 1e-9:
                log(f"{T:.0f} s: pasaría de 0.5 rad/s ({1.875 * dq / T:.2f}); se salta")
                continue
            reg = []
            yaw0 = estado.rpy()[2]
            p0 = pies.leer() if pies else None
            vig.yaw0, vig.giro_obs, vig.incl_obs = yaw0, 0.0, 0.0
            parar = threading.Event()

            def registrar():
                while not parar.is_set():
                    q = estado.q()
                    reg.append((time.monotonic(), *estado.rpy(), *cm.com(q)))
                    time.sleep(0.02)
            h = threading.Thread(target=registrar, daemon=True)
            h.start()
            motivo = None
            try:
                log(f"duración {T:.0f} s: ida")
                brazo.mover({a.lado: q_alc}, duracion=T, comprobar=vig)
                time.sleep(3.0)
                log(f"duración {T:.0f} s: vuelta")
                brazo.mover({a.lado: obs[a.lado]}, duracion=T, comprobar=vig)
                time.sleep(3.0)
            except Parada as e:
                motivo = str(e)
            parar.set()
            h.join()
            r = np.array(reg)
            giro = float(np.degrees(np.max(np.abs(np.unwrap(r[:, 3]) - yaw0)))) if len(r) else float("nan")
            incl = float(np.degrees(np.max(np.abs(r[:, 1:3])))) if len(r) else float("nan")
            dcom = float(1000 * np.max(np.linalg.norm(r[:, 4:6] - r[0, 4:6], axis=1))) if len(r) else float("nan")
            if pies:
                p1 = pies.leer()
                desp = max(float(1000 * np.linalg.norm(p1[k] - p0[k])) for k in p0) if p0 else float("nan")
                pasos = int(desp > 50.0)
            else:
                desp = float("nan")
                try:
                    pasos = int(input(f"  ¿Cuántos pasos dio con {T:.0f} s? (pies fuera de la huella > 5 cm): ") or 0)
                except ValueError:
                    pasos = 0
            fila = {"duracion_s": T, "giro_max_deg": giro, "incl_max_deg": incl, "pasos": pasos,
                    "pies_mm": desp, "com_desplaz_mm": dcom, "dq_rad": dq,
                    "v_pico_rad_s": 1.875 * dq / T, "a_pico_rad_s2": 5.7735 * dq / T ** 2, "cortado": motivo}
            filas.append(fila)
            log(f"  {T:.0f} s: giro {giro:.2f} grados, inclinación {incl:.2f}, pasos {pasos}, pies {desp:.1f} mm, "
                f"centro de masas {dcom:.1f} mm, v pico {fila['v_pico_rad_s']:.2f} rad/s, "
                f"a pico {fila['a_pico_rad_s2']:.3f} rad/s2" + (f"  CORTADO: {motivo}" if motivo else ""))
            if motivo or pasos or giro > 2.0:
                log("se para aquí: esta duración ya mueve el cuerpo")
                break
        brazo.mover(q_ini, comprobar=vig)
    except (Parada, KeyboardInterrupt) as e:
        log(f"PARADA: {e}")
    finally:
        vig.mirar_seguimiento = False
        brazo.rampa_salida()
        log(f"peso de arm_sdk = {brazo.peso:.2f}")
        brazo.parar()
        if pies:
            pies.cerrar()
    seguras = [f["duracion_s"] for f in filas if not f["cortado"] and not f["pasos"] and f["giro_max_deg"] <= 2.0]
    t_min = min(seguras) if seguras else None
    res = {"fecha": time.strftime("%Y%m%d_%H%M%S"), "sim": a.sim, "lado": a.lado, "filas": filas,
           "duracion_min_medida": t_min,
           "duracion_min_recomendada": None if t_min is None else max(8.0, t_min),
           "primeras_tiradas": None if t_min is None else 2 * max(8.0, t_min),
           "nota": "en el simulador las piernas son rígidas: no puede dar pasos; vale la curva del robot"}
    os.makedirs(DATOS, exist_ok=True)
    ruta = os.path.join(DATOS, f"curva_{res['fecha']}.json")
    json.dump(res, open(ruta, "w"), indent=1)
    if filas:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        T = [f["duracion_s"] for f in filas]
        ax[0].plot(T, [f["giro_max_deg"] for f in filas], "o-", label="giro máx (grados)")
        ax[0].plot(T, [f["incl_max_deg"] for f in filas], "s-", label="inclinación máx (grados)")
        ax[0].plot(T, [f["pasos"] for f in filas], "^-", label="pasos")
        ax[0].set_xlabel("duración del movimiento (s)")
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3)
        ax[1].plot(T, [f["com_desplaz_mm"] for f in filas], "o-", label="centro de masas (mm)")
        ax[1].plot(T, [100 * f["v_pico_rad_s"] for f in filas], "s-", label="v pico (x100 rad/s)")
        ax[1].plot(T, [1000 * f["a_pico_rad_s2"] for f in filas], "^-", label="a pico (x1000 rad/s2)")
        ax[1].set_xlabel("duración del movimiento (s)")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3)
        fig.suptitle(f"Curva de pasos ({'simulador' if a.sim else 'robot'}, brazo {a.lado})", fontsize=10)
        fig.tight_layout()
        fig.savefig(ruta.replace(".json", ".png"), dpi=110)
    log(f"duración mínima medida: {t_min} s -> config/brazo.yaml duracion_min = {res['duracion_min_recomendada']}; "
        f"primeras tiradas al doble: {res['primeras_tiradas']} s. Guardado {ruta}")


if __name__ == "__main__":
    main()
