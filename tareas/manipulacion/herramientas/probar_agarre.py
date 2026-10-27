#!/usr/bin/env python3
"""Banco de pruebas del AGARRE con la física del simulador, sin ROS y más rápido que el tiempo
real (preguntas 6.4.1, 6.4.3, 6.4.5). Para ajustar config/agarres.yaml antes de las tiradas.

Con la pose VERDADERA del objeto (aquí se prueba la mano, no la percepción) y opcionalmente
un error de posición (tolerancia de la preforma, 6.4.1), el planificador hace el plan y el
robot simulado (RobotSimulado de h1_2_sim2real: la misma física, la mano con FORCE_SET real,
sin asistencia de agarre) lo ejecuta por rt/arm_sdk "en memoria" (las órdenes que publicaría
el bloque del brazo) y la mano por sus registros Modbus. Mide:
  * agarre detectado por la mano (dedos parados lejos de 0 con fuerza, 6.4.3);
  * objeto levantado (verdad de terreno) y si se cae durante el transporte;
  * objeto quieto dentro de su bandeja al final, y si se tumbó (6.4.5).

    python3 herramientas/probar_agarre.py --objeto cubo
    python3 herramientas/probar_agarre.py --todos --error 0.01      # con 1 cm de error de posición
"""
import argparse
import json
import math
import os
import sys
import tempfile
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "sim"))
from construir_estacion import S2R, construir  # noqa: E402

sys.path.insert(0, S2R)
from h1_2_sim2real.comun import J  # noqa: E402
from h1_2_sim2real.robot_simulado import RobotSimulado  # noqa: E402

from manip import cinematica as C  # noqa: E402
from manip.brazo import minima_sacudida  # noqa: E402
from manip.compensacion import Compensador, por_cuerpo  # noqa: E402
from manip.config import _leer, cargar, cargar_objetos  # noqa: E402
from manip.estacion import Estacion  # noqa: E402
from manip.modelo import Modelo  # noqa: E402
from manip.planificacion import Planificador  # noqa: E402
from manip.robot_io import INDICES  # noqa: E402
from manip.tipos import ObjetoDetectado  # noqa: E402

DATOS = os.path.join(RAIZ, "datos", "agarre")


class Ejecutor:
    """RobotSimulado + órdenes de rt/arm_sdk y de la mano, paso a paso."""

    def __init__(self, xml, cfg_brazo, compensar=True):
        self.sim = RobotSimulado(xml, log=lambda s: None)
        n = J.N
        self.c = {"lc_t": np.array([-1e9]), "lc_q": np.zeros(n), "lc_dq": np.zeros(n), "lc_kp": np.zeros(n),
                  "lc_kd": np.zeros(n), "lc_tau": np.zeros(n), "lc_mode": np.zeros(n), "arm_t": np.zeros(1),
                  "arm_q": np.zeros(n), "arm_dq": np.zeros(n), "arm_kp": np.zeros(n), "arm_kd": np.zeros(n),
                  "arm_tau": np.zeros(n), "arm_w": np.ones(1), "loco_v": np.zeros(3), "loco_t_fin": np.zeros(1),
                  "fsm": np.array([201.0])}
        self.c["arm_kp"][12:27] = cfg_brazo["kp"]
        self.c["arm_kd"][12:27] = cfg_brazo["kd"]
        self.c["arm_kp"][12], self.c["arm_kd"][12] = cfg_brazo["kp_cintura"], cfg_brazo["kd_cintura"]
        q0 = self.sim.data.qpos[self.sim.qadr].copy()
        self.c["arm_q"][:] = q0
        self.t = 0.0
        self.cfg = cfg_brazo
        # consigna DESEADA (la del plan); la publicada lleva el desvío de la compensación
        self.deseada = {lado: q0[INDICES[lado]].copy() for lado in C.LADOS}
        self.comp = Compensador(xml, cfg_brazo["kp"]) if compensar else None

    def pasos(self, segundos, cada=None):
        for k in range(int(segundos / self.sim.dt)):
            self.c["arm_t"][0] = self.t
            self.sim.paso(self.t, self.c)
            self.t += self.sim.dt
            if cada is not None:
                cada()

    def brazo(self, lado):
        return self.deseada[lado].copy()

    def medida(self, lado):
        return self.sim.data.qpos[[self.sim.qadr[i] for i in INDICES[lado]]].copy()

    def poner(self, lado, q):
        self.deseada[lado] = np.asarray(q, float).copy()
        self.c["arm_q"][INDICES[lado]] = q + (self.comp.desvio(lado, q) if self.comp else 0.0)

    def asentar(self, lado, segundos=0.6, veces=2):
        """Parada: espera quieto y corrige con los encoders (lazo de la compensación).
        Devuelve el error articular máximo (rad) que quedaba antes de la última corrección."""
        err = 0.0
        for _ in range(veces):
            self.pasos(segundos)
            if self.comp is None:
                return float(np.max(np.abs(self.deseada[lado] - self.medida(lado))))
            err = float(np.max(np.abs(self.comp.corregir(lado, self.deseada[lado], self.medida(lado)))))
            self.poner(lado, self.deseada[lado])
        self.pasos(segundos)
        return err

    def seguir(self, lado, ruta, cada=None, duracion=None):
        """Como Brazo.seguir_ruta: mínima sacudida sobre la longitud de arco."""
        ruta = np.vstack([self.brazo(lado), np.asarray(ruta)])
        seg = np.linalg.norm(np.diff(ruta, axis=0), axis=1)
        largo = np.concatenate([[0.0], np.cumsum(seg)])
        T = max(self.cfg["duracion_min"], 1.875 * largo[-1] / self.cfg["v_max"], duracion or 0.0)
        n = int(T / self.sim.dt)
        for k in range(1, n + 1):
            s = minima_sacudida(k / n) * largo[-1]
            i = min(int(np.searchsorted(largo, s, side="right")) - 1, len(seg) - 1)
            a = (s - largo[i]) / max(seg[i], 1e-12)
            self.poner(lado, ruta[i] + a * (ruta[i + 1] - ruta[i]))
            self.pasos(self.sim.dt, cada)
        return T

    def cuerpo(self, nombre):
        p, q = self.sim.pose_cuerpo(nombre)
        return p, q


def foto(sim, lookat, azimut, elev=-10, dist=0.35):
    """Imagen (320x240) del simulador desde una cámara libre: cerrar, elevar y final."""
    import mujoco
    r = mujoco.Renderer(sim.model, 240, 320)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = dist, azimut, elev
    r.update_scene(sim.data, cam)
    img = r.render().copy()
    r.close()
    return img


def mano_estado(mano):
    a = mano.leer(J.ANGLE_ACT, 6)
    f = mano.leer(J.FORCE_ACT, 6)
    return np.array(a, float), np.array(f, float)


def probar(obj_nombre, nivel, semilla, error, cfg, cat, agarres, cfg_plan, cfg_brazo, poses, mod, rng, guardar,
           corregir_cuerpo=True):
    xml, geo = construir(nivel, semilla, cfg, cat, salida=tempfile.mkdtemp())
    est = Estacion(cfg)
    ej = Ejecutor(xml, cfg_brazo)
    sim = ej.sim
    tor = sim.model.body("torso_link").id
    # 2 s de pie para que se asiente; luego a la observación (como en la tirada)
    ej.pasos(2.0)
    obs = {k: np.asarray(v) for k, v in poses["observacion"].items()}
    for lado in C.LADOS:
        ej.seguir(lado, [obs[lado]])
    o = geo["objetos"][obj_nombre]
    Rt, pt = sim.data.xmat[tor].reshape(3, 3).copy(), sim.data.xpos[tor].copy()
    pw, _ = ej.cuerpo(obj_nombre)
    p_t = Rt.T @ (pw - pt)
    if error > 0:
        ang = rng.uniform(0, 2 * math.pi)
        p_t = p_t + error * np.array([math.cos(ang), math.sin(ang), 0.0])
    agr = agarres[obj_nombre]
    sim_g = math.radians(agr.get("simetria_deg", 0))
    objeto = ObjetoDetectado(obj_nombre, p_t, o["guinada"], sim_g, np.zeros(3), 1.0, 1.0, 500)
    plan = Planificador(est, agarres, cat, cfg_plan, mod, poses["observacion"])
    R0 = Rt.copy()                                # IMU cuando se midió el objeto
    plano = np.array([0, 0, 1.0, -(est.z_mesa - pt[2])])
    # DESTINO="x,y" (marco del torso): dejarlo en ese punto de la mesa, como el cambio de mano del
    # nivel 3, en vez de en su bandeja
    destino = tuple(float(v) for v in os.environ["DESTINO"].split(",")) if os.environ.get("DESTINO") else None
    # los demás objetos, como los ve el supervisor (sin ellos el plan puede pasar por encima de uno)
    otros = []
    for nm, og in geo["objetos"].items():
        if nm != obj_nombre:
            po = Rt.T @ (ej.cuerpo(nm)[0] - pt)
            c = cat[nm]
            med = {"caja": lambda: c["medidas"], "cilindro": lambda: [2 * c["radio"], 2 * c["radio"], c["alto"]],
                   "esfera": lambda: [2 * c["radio"]] * 3}.get(c["forma"], lambda: [0.06] * 3)()
            otros.append(ObjetoDetectado(nm, po, og["guinada"], math.radians(agarres[nm].get("simetria_deg", 0)),
                                         np.asarray(med, float), 1.0, 1.0, 500))
    # MANO=izq|der: la mano que lo coge (por defecto la de su lado)
    p = plan.planificar(objeto, obs, bandeja=destino, plano=plano, otros=otros, mano=os.environ.get("MANO") or None)
    res = {"objeto": obj_nombre, "nivel": nivel, "semilla": semilla, "error_m": error, "plan": p.ok,
           "motivo": p.motivo}
    if not p.ok:
        return res
    lado = p.mano
    mano = sim.manos[lado]
    ag = p.agarre
    mano.escribir(J.SPEED_SET, [150] * 6)
    mano.escribir(J.ANGLE_SET, ag.preforma)
    t = {tr.nombre: tr for tr in p.tramos}
    p_antes = ej.cuerpo(obj_nombre)[0].copy()
    ej.seguir(lado, t["aproximar"].q_ruta)
    res["error_preagarre_deg"] = math.degrees(ej.asentar(lado))      # parada: lazo con los encoders
    baja, sube = t["descender"], t["elevar"]
    if corregir_cuerpo:
        # IMU: el cuerpo se ha inclinado al extender el brazo; el objeto, medido en el marco del
        # torso desde la observación, está en otro sitio de ese marco
        R1 = sim.data.xmat[tor].reshape(3, 3).copy()
        p_ag = (C.fk(lado, baja.q_ruta[-1])[:3, :3] @ plan.off(lado, obj_nombre)
                + C.fk(lado, baja.q_ruta[-1])[:3, 3])
        p_nuevo = por_cuerpo(p_ag, R0, R1, cfg_brazo["pivote_cuerpo"])
        res["inclinacion_cuerpo_deg"] = math.degrees(math.acos(min(1.0, (np.trace(R1.T @ R0) - 1) / 2)))
        res["correccion_cuerpo_mm"] = float(1000 * np.linalg.norm(p_nuevo - p_ag))
        nuevo = plan.rehacer_descenso(p, p_nuevo, ej.brazo(lado))
        if nuevo is not None:
            baja, sube = nuevo
    ej.seguir(lado, baja.q_ruta)
    ej.pasos(0.5)
    res["error_agarre_mm"] = float(1000 * np.linalg.norm(C.fk(lado, ej.medida(lado))[:3, 3]
                                                         - C.fk(lado, ej.brazo(lado))[:3, 3]))
    z0 = ej.cuerpo(obj_nombre)[0][2]
    res["empujado_mm"] = float(1000 * np.linalg.norm(ej.cuerpo(obj_nombre)[0][:2] - p_antes[:2]))
    # cerrar: FORCE_SET primero, consigna cerrada
    mano.escribir(J.FORCE_SET, ag.force_set)
    if ag.previo:                                 # lateral: primero oponer el pulgar
        mano.escribir(J.ANGLE_SET, ag.previo)
        ej.pasos(2.0)
    mano.escribir(J.ANGLE_SET, ag.cierre)
    ej.pasos(2.5)
    fotos = []
    if guardar:
        fotos.append(foto(sim, ej.cuerpo(obj_nombre)[0], 90))
        fotos.append(foto(sim, ej.cuerpo(obj_nombre)[0], 180))
    if os.environ.get("CONTACTOS"):
        import mujoco
        m_, d_ = sim.model, sim.data
        oid = m_.body(obj_nombre).id
        f6 = np.zeros(6)
        wb = m_.body(("left" if lado == "izq" else "right") + "_wrist_yaw_link").id
        Rw = d_.xmat[wb].reshape(3, 3)
        for i in range(d_.ncon):
            c_ = d_.contact[i]
            b1, b2 = m_.geom_bodyid[c_.geom1], m_.geom_bodyid[c_.geom2]
            if oid in (b1, b2):
                mujoco.mj_contactForce(m_, d_, i, f6)
                print(f"   contacto {m_.body(b2 if b1 == oid else b1).name:24s} "
                      f"normal(muñeca) "
                      f"{np.round(Rw.T @ c_.frame[:3], 2)} fn {f6[0]:.2f} N  "
                      f"pos(muñeca) {np.round(Rw.T @ (c_.pos - d_.xpos[wb]), 3)}")
    a, f = mano_estado(mano)
    tocan = [k for k in range(5) if a[k] > 60 and f[k] >= 0.5 * ag.force_set[k]]
    res["agarre_mano"] = bool(4 in tocan and any(k < 4 for k in tocan))
    res["angulos"], res["fuerzas"] = a.astype(int).tolist(), f.astype(int).tolist()
    if ej.comp is not None:                       # el objeto pesa en la mano: al modelo del error
        ej.comp.poner_carga(lado, cat[obj_nombre]["masa"], plan.off(lado, obj_nombre))
    zm0 = C.fk(lado, ej.medida(lado))[2, 3]
    ej.seguir(lado, sube.q_ruta)
    res["mano_subio_mm"] = float(1000 * (C.fk(lado, ej.medida(lado))[2, 3] - zm0))
    ej.pasos(float(os.environ.get("ESPERA_ELEVADO", 1.0)))     # >1: ¿se escurre quieto?
    if guardar:
        fotos.append(foto(sim, ej.cuerpo(obj_nombre)[0], 90))
    res["levantado_mm"] = float(1000 * (ej.cuerpo(obj_nombre)[0][2] - z0))
    # dónde está el objeto en la mano (marco de la muñeca, como `suelta_off`; y de la mano izquierda)
    wb = sim.model.body(("left" if lado == "izq" else "right") + "_wrist_yaw_link").id
    Rw = sim.data.xmat[wb].reshape(3, 3)
    om = Rw.T @ (ej.cuerpo(obj_nombre)[0] - sim.data.xpos[wb])
    res["en_mano_off"] = [round(float(om[0]), 4), round(float(om[1] if lado == "izq" else -om[1]), 4),
                          round(float(om[2]), 4)]
    res["off_plan"] = [round(float(v), 4) for v in plan.off(lado, obj_nombre) * (1, 1 if lado == "izq" else -1, 1)]
    a2, f2 = mano_estado(mano)                    # la segunda comprobación del supervisor
    res["dedos_elevado"] = len([k for k in range(4) if a2[k] > 60 and f2[k] >= 0.5 * ag.force_set[k]])
    ej.seguir(lado, t["transportar"].q_ruta, duracion=plan.duracion_transporte.get(obj_nombre))
    baja_s, retira = t["bajar"], t["retirar"]
    if corregir_cuerpo:                           # como el supervisor: el cuerpo también antes de soltar
        ej.asentar(lado)
        R2 = sim.data.xmat[tor].reshape(3, 3).copy()
        p_su = plan.punto_suelta(p)
        p_su_n = por_cuerpo(p_su, R0, R2, cfg_brazo["pivote_cuerpo"])
        res["correccion_suelta_mm"] = float(1000 * np.linalg.norm(p_su_n - p_su))
        res["correccion_suelta_vec_mm"] = (1000 * (p_su_n - p_su)).round(1).tolist()
        nv = plan.rehacer_suelta(p, p_su_n, ej.brazo(lado))
        if nv is not None:
            baja_s, retira = nv
    ej.seguir(lado, baja_s.q_ruta)
    muneca = pt + Rt @ C.fk(lado, ej.brazo(lado))[:3, 3]
    res["en_mano_al_bajar"] = bool(np.linalg.norm(ej.cuerpo(obj_nombre)[0] - muneca) < 0.30)
    mano.escribir(J.ANGLE_SET, ag.suelta or ag.preforma)   # soltar: como el supervisor
    ej.pasos(1.5)
    ej.seguir(lado, retira.q_ruta)
    ej.seguir(lado, t["volver"].q_ruta)
    ej.pasos(2.0)
    pf, qf = ej.cuerpo(obj_nombre)
    b = est.bandejas[cat[obj_nombre]["bandeja"]]
    res["en_bandeja"] = bool(b.contiene(pf[0], pf[1], 0.0))
    if destino is not None:                       # en la mesa: a menos de 3 cm del punto
        pf_t = Rt.T @ (pf - pt)
        res["error_destino_mm"] = float(1000 * np.hypot(pf_t[0] - destino[0], pf_t[1] - destino[1]))
        res["en_bandeja"] = res["error_destino_mm"] < 30
    # ¿tumbado? eje z del objeto respecto de la vertical (para lo que va de pie)
    R = np.zeros(9)
    import mujoco
    mujoco.mju_quat2Mat(R, qf)
    zz = R.reshape(3, 3)[:, 2]
    res["inclinacion_deg"] = float(math.degrees(math.acos(min(1.0, abs(zz[2])))))
    res["quieto"] = True
    res["final_xyz"] = pf.round(3).tolist()
    res["ok"] = bool(res["agarre_mano"] and res["levantado_mm"] > 20 and res["en_bandeja"])
    if guardar:
        fotos.append(foto(sim, np.array([b.cx, b.cy, est.z_mesa]), 180, elev=-45, dist=0.5))
        import cv2
        os.makedirs(DATOS, exist_ok=True)
        cv2.imwrite(os.path.join(DATOS, f"{obj_nombre}_n{nivel}_s{semilla}.png"), np.hstack(fotos)[..., ::-1])
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objeto", default="cubo")
    ap.add_argument("--todos", action="store_true")
    ap.add_argument("--nivel", type=int, default=2)
    ap.add_argument("--semillas", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--error", type=float, default=0.0, help="error de posición del objeto (m) para 6.4.1")
    ap.add_argument("--guardar", action="store_true", help="imagen final de cada prueba")
    ap.add_argument("--fijar", nargs="*", default=[], help="clave=valor para el agarre del objeto (barridos)")
    a = ap.parse_args()
    cfg, cat = cargar(), cargar_objetos()
    agarres = _leer(os.path.join(RAIZ, "config", "agarres.yaml"))
    import yaml
    for kv in a.fijar:
        k, v = kv.split("=", 1)
        for ob in agarres:
            if a.todos or ob == a.objeto:
                agarres[ob][k] = yaml.safe_load(v)
    cfg_plan = _leer(os.path.join(RAIZ, "config", "planificacion.yaml"))
    cfg_brazo = _leer(os.path.join(RAIZ, "config", "brazo.yaml"))
    poses = _leer(os.path.join(RAIZ, "config", "poses.yaml"))
    mod = Modelo(construir(0, 0, cfg, cat, salida=tempfile.mkdtemp())[0])
    rng = np.random.default_rng(0)
    objetos = list(cat) if a.todos else [a.objeto]
    filas = []
    for ob in objetos:
        for s in a.semillas:
            t0 = time.time()
            r = probar(ob, a.nivel, s, a.error, cfg, cat, agarres, cfg_plan, cfg_brazo, poses, mod, rng, a.guardar)
            r["t_s"] = round(time.time() - t0, 1)
            filas.append(r)
            if not r["plan"]:
                print(f"{ob:10s} n{a.nivel} s{s}: sin plan ({r['motivo'][:90]})", flush=True)
                continue
            print(f"{ob:10s} n{a.nivel} s{s}: {'OK ' if r['ok'] else 'MAL'} agarre mano {r['agarre_mano']} "
                  f"(ángulos {r['angulos'][:5]}, fuerzas {r['fuerzas'][:5]}), levantado {r['levantado_mm']:.0f} mm "
                  f"(mano {r.get('mano_subio_mm', 0):.0f}), error agarre {r['error_agarre_mm']:.1f} mm, "
                  f"empujado {r['empujado_mm']:.0f} mm, "
                  f"en bandeja {r['en_bandeja']}, inclinado {r['inclinacion_deg']:.0f} grados  [{r['t_s']} s]",
                  flush=True)
    os.makedirs(DATOS, exist_ok=True)
    json.dump(filas, open(os.path.join(DATOS, f"agarre_e{int(1000 * a.error)}mm.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
