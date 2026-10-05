"""Procesos hijos del simulador (ver compartido.py). Se lanzan con multiprocessing
'spawn': cada uno arranca limpio, sin heredar el estado de rclpy ni el contexto GL."""
import faulthandler
import signal
import sys
import time

import numpy as np

from .compartido import Compartido

MARCA_INTERNO = 0x51A     # reserve[0] de los rt/lowcmd del control interno simulado
FUENTES = {0: "interno", 1: "interno+arm_sdk", 2: "lowcmd", 3: "debug_sin_ordenes"}


def _preparar_hijo():
    """Salida sin bufer (si no, lo que imprime un hijo se pierde en el log del
    launch) y traza de todos los hilos con SIGUSR1: el nodo principal la pide si
    el hijo deja de avanzar, antes de reiniciarlo."""
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    faulthandler.register(signal.SIGUSR1, all_threads=True)


def _ritmo(periodo):
    """Generador de esperas a periodo fijo, sin acumular deuda si se retrasa."""
    t_sig = time.monotonic()
    while True:
        t_sig += periodo
        espera = t_sig - time.monotonic()
        if espera > 0:
            time.sleep(espera)
        else:
            t_sig = time.monotonic()
        yield


def proceso_dds(nombre_shm, nq, cerrojo, dominio, interfaz, hz_ls, hz_int, crc):
    """SDK de Unitree: rt/lowstate (pub), rt/lowcmd del control interno (pub),
    rt/lowcmd y rt/arm_sdk externos (sub) -> memoria compartida.

    UN SOLO HILO y SIN listeners de Python: los lectores se sondean con take() en
    el propio bucle. Con los ChannelSubscriber(handler) del SDK este proceso se
    INTERBLOQUEABA (2026-10-04, traza con faulthandler): escribia rt/lowcmd (control
    interno) teniendo un lector con listener en rt/lowcmd; cyclone entrega lo local
    de forma sincrona en el hilo que escribe (con el GIL), y si a la vez llegaba el
    rt/lowcmd de un script externo, su hilo de recepcion tomaba el cerrojo del lector
    y esperaba el GIL -> los dos parados para siempre. Ademas el lector ignora lo
    publicado por este mismo participante (IgnoreLocal) y los escritores son
    best-effort KeepLast(1): un flujo de estado no espera a nadie (los lectores del
    SDK son best-effort por defecto, asi que emparejan igual)."""
    from cyclonedds.pub import DataWriter
    from cyclonedds.qos import Policy, Qos
    from cyclonedds.sub import DataReader
    from cyclonedds.topic import Topic
    from unitree_sdk2py.core.channel import ChannelFactory, ChannelFactoryInitialize
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC

    _preparar_hijo()
    sh = Compartido(nombre_shm, nq)
    v = sh.v
    N = 27
    ChannelFactoryInitialize(dominio, interfaz)
    part = ChannelFactory()._ChannelFactory__participant     # el participante ya configurado (dominio + interfaz)
    t_ls = Topic(part, "rt/lowstate", LowState_)
    t_lc = Topic(part, "rt/lowcmd", LowCmd_)
    t_arm = Topic(part, "rt/arm_sdk", LowCmd_)
    q_pub = Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(1))
    q_sub = Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(1), Policy.IgnoreLocal.Participant)
    pub_ls = DataWriter(part, t_ls, q_pub)
    pub_int = DataWriter(part, t_lc, q_pub)
    sub_lc = DataReader(part, t_lc, q_sub)
    sub_arm = DataReader(part, t_arm, q_sub)

    def ultimo(lector):
        m = None
        for x in lector.take(N=16):                # vacia la cola y se queda con el mas nuevo
            if x is not None and getattr(x, "motor_cmd", None) is not None:
                m = x
        return m

    def a_arrays(m):
        mc = m.motor_cmd
        return (np.array([mc[i].q for i in range(N)]), np.array([mc[i].dq for i in range(N)]),
                np.array([mc[i].kp for i in range(N)]), np.array([mc[i].kd for i in range(N)]),
                np.array([mc[i].tau for i in range(N)]), np.array([mc[i].mode for i in range(N)], dtype=float))

    ls = unitree_hg_msg_dds__LowState_()
    ls.mode_machine = 0                        # como unitree_mujoco (el robot real da 4 o 6)
    for i in range(N):
        ls.motor_state[i].mode = 1
        ls.motor_state[i].temperature = [35, 35]
    cint = unitree_hg_msg_dds__LowCmd_()
    cint.reserve[0] = MARCA_INTERNO
    for i in range(N):
        cint.motor_cmd[i].mode = 1
    calc = CRC() if crc else None
    k_int = max(1, int(round(hz_ls / hz_int)))
    n = 0
    for _ in _ritmo(1.0 / hz_ls):
        lc, arm = ultimo(sub_lc), ultimo(sub_arm)
        if lc is not None and lc.reserve[0] == MARCA_INTERNO:      # el control interno de otra instancia
            lc = None
        lc_a = a_arrays(lc) if lc is not None else None
        arm_a = a_arrays(arm) if arm is not None else None
        ahora = time.monotonic()
        with cerrojo:
            if v["parar"][0]:
                break
            if lc_a is not None:
                q, dq, kp, kd, tau, mode = lc_a
                v["lc_q"][:], v["lc_dq"][:], v["lc_kp"][:], v["lc_kd"][:] = q, dq, kp, kd
                v["lc_tau"][:], v["lc_mode"][:] = tau, mode
                v["lc_t"][0] = ahora
            if arm_a is not None:
                q, dq, kp, kd, tau, _ = arm_a
                v["arm_q"][:], v["arm_dq"][:], v["arm_kp"][:], v["arm_kd"][:], v["arm_tau"][:] = q, dq, kp, kd, tau
                v["arm_w"][0] = arm.motor_cmd[27].q
                v["arm_t"][0] = ahora
            tick, q, dq, tau = int(v["tick"][0]), v["q"].copy(), v["dq"].copy(), v["tau"].copy()
            quat, rpy, gyro, acc = v["quat"].tolist(), v["rpy"].tolist(), v["gyro"].tolist(), v["acc"].tolist()
            fuente = int(v["fuente"][0])
            if n % k_int == 0:
                iq, ikp, ikd = v["int_q"].copy(), v["int_kp"].copy(), v["int_kd"].copy()
        ls.tick = tick
        for i in range(N):
            s = ls.motor_state[i]
            s.q, s.dq, s.tau_est = float(q[i]), float(dq[i]), float(tau[i])
        imu = ls.imu_state
        imu.quaternion, imu.rpy, imu.gyroscope, imu.accelerometer = quat, rpy, gyro, acc
        if calc is not None:
            ls.crc = calc.Crc(ls)
        pub_ls.write(ls)
        v["ls_n"][0] += 1                      # unico escritor: sin cerrojo
        # el control interno solo "existe" mientras manda (FSM 201); en Debug no publica
        if n % k_int == 0 and fuente in (0, 1):
            for i in range(N):
                c = cint.motor_cmd[i]
                c.q, c.kp, c.kd = float(iq[i]), float(ikp[i]), float(ikd[i])
            pub_int.write(cint)
        n += 1
    sh.cerrar()


def patron_emisor(h, w, semilla, n=6000):
    """Patron de puntos del proyector IR de la D435 (aprox.): puntos dispersos fijos
    en la imagen, algo borrosos. Se aplica MULTIPLICANDO la reflectancia: la luz del
    proyector que vuelve es proporcional al albedo, asi que los puntos se ven fuertes
    en el suelo claro y casi nada en la cinta mate oscura. Aproximaciones: el patron
    no se desplaza con la profundidad y no se separa la luz ambiente del albedo."""
    rng = np.random.default_rng(semilla)
    p = np.zeros((h, w), np.float32)
    ys, xs = rng.integers(0, h, n), rng.integers(0, w, n)
    p[ys, xs] = rng.uniform(0.5, 1.0, n)
    # borroso con un nucleo 3x3 sin depender de cv2
    q = p.copy()
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        q += 0.35 * np.roll(np.roll(p, dy, 0), dx, 1)
    return np.clip(q, 0.0, 1.0)


def proceso_camaras(nombre_shm, nq, cerrojo, modelo, con_ir, hz, linea_base, emisor=True):
    """D435 simulada: RGB + depth (alineada al RGB) + par IR, con su propio MjData
    al que se copia el qpos de la fisica. Topicos como realsense2_camera."""
    import os
    # Offscreen por EGL: en este portatil (prime on-demand) usa la GTX 1650 (~7 ms
    # por imagen 848x480); con el GLFW por defecto cae en la Intel integrada (~65 ms).
    os.environ.setdefault("MUJOCO_GL", "egl")
    _preparar_hijo()
    import mujoco
    import rclpy
    from geometry_msgs.msg import TransformStamped
    from sensor_msgs.msg import CameraInfo, Image
    from tf2_ros import StaticTransformBroadcaster

    from . import mensajes
    from .robot_simulado import info_camara

    sh = Compartido(nombre_shm, nq)
    v = sh.v
    m = mujoco.MjModel.from_xml_path(modelo)
    d = mujoco.MjData(m)
    rclpy.init()
    nodo = rclpy.create_node("h1_2_sim_camaras")
    log = nodo.get_logger()

    defs = [("robot_rgbd_camera", "color", "/camera/color/image_raw", "/camera/color/camera_info",
             "camera_color_optical_frame", 0.0)]
    if con_ir:
        defs += [("robot_ir_left", "ir", "/camera/infra1/image_rect_raw", "/camera/infra1/camera_info",
                  "camera_infra1_optical_frame", 0.0),
                 ("robot_ir_right", "ir", "/camera/infra2/image_rect_raw", "/camera/infra2/camera_info",
                  "camera_infra2_optical_frame", -linea_base)]
    cams, tfs = [], []
    stamp = nodo.get_clock().now().to_msg()
    for nombre, tipo, t_img, t_info, frame, tx in defs:
        c = info_camara(m, nombre)
        if c is None:
            log.warn(f"el modelo no tiene la camara '{nombre}'")
            continue
        c.update(tipo=tipo, frame=frame, tx=tx, pub_img=nodo.create_publisher(Image, t_img, 5),
                 pub_info=nodo.create_publisher(CameraInfo, t_info, 5))
        if tipo == "color":
            c["pub_depth"] = nodo.create_publisher(Image, "/camera/depth/image_raw", 5)
            c["pub_depth_info"] = nodo.create_publisher(CameraInfo, "/camera/depth/camera_info", 5)
        cams.append(c)
        for f in [frame] + (["camera_depth_optical_frame"] if tipo == "color" else []):
            t = TransformStamped()
            t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, c["body"], f
            (t.transform.translation.x, t.transform.translation.y,
             t.transform.translation.z) = (float(x) for x in c["pos"])
            (t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z,
             t.transform.rotation.w) = (float(x) for x in c["quat_xyzw"])
            tfs.append(t)
    tf_static = StaticTransformBroadcaster(nodo)
    tf_static.sendTransform(tfs)
    for k, c in enumerate(cams):
        if c["tipo"] == "ir" and emisor:
            c["emisor"] = 1.0 + 0.8 * patron_emisor(c["h"], c["w"], semilla=7 + k)
    if con_ir:
        log.info(f"emisor IR {'ENCENDIDO (patron de puntos)' if emisor else 'apagado'}")
    rend = {}
    for c in cams:
        if (c["w"], c["h"]) not in rend:
            rend[(c["w"], c["h"])] = mujoco.Renderer(m, height=c["h"], width=c["w"])
    log.info("D435 simulada: " + ", ".join(f"{c['frame']} {c['w']}x{c['h']}" for c in cams) + f" a {hz} Hz")

    n_frames, t_log = 0, time.monotonic()
    for _ in _ritmo(1.0 / hz):
        if not rclpy.ok():
            break
        with cerrojo:
            if v["parar"][0]:
                break
            if v["qpos_n"][0] == 0:
                continue
            d.qpos[:] = v["qpos"]
        mujoco.mj_forward(m, d)
        stamp = nodo.get_clock().now().to_msg()
        for c in cams:
            r = rend[(c["w"], c["h"])]
            r.disable_depth_rendering()
            r.update_scene(d, camera=c["id"])
            rgb = r.render()
            info = mensajes.camera_info(c["w"], c["h"], c["fovy"], c["frame"], stamp, c["tx"])
            if c["tipo"] == "color":
                c["pub_img"].publish(mensajes.imagen(rgb, "rgb8", c["frame"], stamp))
                c["pub_info"].publish(info)
                r.enable_depth_rendering()
                r.update_scene(d, camera=c["id"])
                depth = r.render()
                c["pub_depth"].publish(mensajes.imagen(depth, "32FC1", "camera_depth_optical_frame", stamp))
                c["pub_depth_info"].publish(mensajes.camera_info(
                    c["w"], c["h"], c["fovy"], "camera_depth_optical_frame", stamp))
            else:
                gris = ((rgb[..., 0].astype(np.uint16) * 77 + rgb[..., 1].astype(np.uint16) * 150
                         + rgb[..., 2].astype(np.uint16) * 29) >> 8).astype(np.uint8)    # luminancia BT.601
                if "emisor" in c:
                    gris = np.clip(gris * c["emisor"], 0, 255).astype(np.uint8)
                c["pub_img"].publish(mensajes.imagen(gris, "mono8", c["frame"], stamp))
                c["pub_info"].publish(info)
        n_frames += 1
        if time.monotonic() - t_log > 30.0:
            log.info(f"camaras a {n_frames / (time.monotonic() - t_log):.1f} Hz")
            n_frames, t_log = 0, time.monotonic()
    for r in rend.values():
        r.close()
    nodo.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    sh.cerrar()


def proceso_loco(nombre_shm, nq, cerrojo, dominio, interfaz, fsm_inicial):
    """Servicio RPC "loco" del SDK de Unitree (el que usa LocoClient), simulado:
    Move/StopMove (SetVelocity 8105), SetFsmId (8101), SetStandHeight (8104) y los
    Get* que usa el C++ (8001-8005). Las velocidades van a la memoria compartida y
    las aplica la marcha cinematica del proceso de fisica (robot_simulado.py).

    Proceso aparte: el ServerStub del SDK usa un listener en rt/api/loco/request y
    escribe en rt/api/loco/response (topicos distintos: no es el patron que
    interbloqueaba al proceso dds), pero asi no comparte GIL con el estado a 500 Hz."""
    import json

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.h1.loco import h1_loco_api as api
    from unitree_sdk2py.rpc.server import Server

    _preparar_hijo()
    sh = Compartido(nombre_shm, nq)
    v = sh.v
    with cerrojo:
        v["fsm"][0] = fsm_inicial
    ChannelFactoryInitialize(dominio, interfaz)

    def set_velocity(par):
        p = json.loads(par)
        vx, vy, w = (float(x) for x in p.get("velocity", [0.0, 0.0, 0.0]))
        dur = float(p.get("duration", 1.0))
        with cerrojo:
            v["loco_v"][:] = (vx, vy, w)
            v["loco_t_fin"][0] = time.monotonic() + dur
            v["loco_n"][0] += 1
        return 0, ""

    def set_fsm(par):
        fsm = int(json.loads(par).get("data", 0))
        with cerrojo:
            v["fsm"][0] = fsm
            v["loco_v"][:] = 0.0
            v["loco_t_fin"][0] = 0.0
        print(f"[loco simulado] SetFsmId({fsm})", flush=True)
        return 0, ""

    def get(clave):
        def h(_par):
            with cerrojo:
                fsm = int(v["fsm"][0])
            return 0, json.dumps({"data": {"fsm": fsm, "mode": 0, "balance": 0, "swing": 0.08,
                                           "stand": 1.0}[clave]})
        return h

    class LocoSimulado(Server):
        def __init__(self):
            super().__init__(api.LOCO_SERVICE_NAME)

        def Init(self):
            self._SetApiVersion(api.LOCO_API_VERSION)
            self._RegistHandler(api.ROBOT_API_ID_LOCO_SET_VELOCITY, set_velocity, False)
            self._RegistHandler(api.ROBOT_API_ID_LOCO_SET_FSM_ID, set_fsm, False)
            for a in (api.ROBOT_API_ID_LOCO_SET_STAND_HEIGHT, api.ROBOT_API_ID_LOCO_SET_BALANCE_MODE,
                      api.ROBOT_API_ID_LOCO_SET_SWING_HEIGHT):
                self._RegistHandler(a, lambda _p: (0, ""), False)
            for a, clave in ((api.ROBOT_API_ID_LOCO_GET_FSM_ID, "fsm"), (api.ROBOT_API_ID_LOCO_GET_FSM_MODE, "mode"),
                             (api.ROBOT_API_ID_LOCO_GET_BALANCE_MODE, "balance"),
                             (api.ROBOT_API_ID_LOCO_GET_SWING_HEIGHT, "swing"),
                             (api.ROBOT_API_ID_LOCO_GET_STAND_HEIGHT, "stand")):
                self._RegistHandler(a, get(clave), False)

    srv = LocoSimulado()
    srv.Init()
    srv.Start(False)
    print(f"[loco simulado] servicio 'loco' listo (FSM {fsm_inicial})", flush=True)
    while True:
        time.sleep(0.2)
        with cerrojo:
            if v["parar"][0]:
                break
    sh.cerrar()
