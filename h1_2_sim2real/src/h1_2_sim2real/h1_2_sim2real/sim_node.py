"""Nodo del H1-2 simulado: MuJoCo + DDS del SDK de Unitree + manos Modbus + D435.

Lo que ve el resto del mundo (igual que el robot real):
  SDK (dominio 1, lo)   rt/lowstate (pub, 500 Hz), rt/lowcmd y rt/arm_sdk (sub),
                        rt/lowcmd del control interno (pub, mientras manda)
  Modbus TCP            manos Inspire en 127.0.1.211:6000 (izq) y 127.0.1.210:6000 (der)
  ROS 2 (D435)          /camera/color/image_raw + camera_info          RGB
                        /camera/depth/image_raw + camera_info          32FC1 en metros, alineada al RGB
                        /camera/infra1/image_rect_raw + camera_info    IR izquierda, mono8
                        /camera/infra2/image_rect_raw + camera_info    IR derecha, mono8, P[3] = -fx*0.05
  ROS 2 (solo sim)      /sim/joint_states  verdad de terreno (27 + 12 dedos), NO la del robot
                        /sim/fuente        quien manda: interno | interno+arm_sdk | lowcmd
                        /contacts, /sim/ground_truth/<cuerpo>
Los /joint_states "de verdad" los publica el puente a partir de rt/lowstate,
igual con el robot real que con el simulador.

Tres procesos (ver compartido.py): este (fisica a 500 Hz, manos, viewer), el de
DDS y el de camaras.
"""
import multiprocessing as mp
import os
import signal
import threading
import time

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import MarkerArray

from . import mensajes
from .comun import J, recurso
from .compartido import Compartido
from .procesos import proceso_camaras, proceso_dds, proceso_loco
from .robot_simulado import RobotSimulado

IR_LINEA_BASE = 0.05     # m, D435 (la misma que build_scene_hands.py)
ESCENAS = {
    "faja": "h1_2_escena_faja_manos.xml",   # base soldada, mesa + faja + mandarina
    "suelo": "h1_2_escena_suelo.xml",       # robot solo de pie en el suelo, base flotante (construir_escena_suelo.py)
}
CODIGO_FUENTE = {"interno": 0, "interno+arm_sdk": 1, "lowcmd": 2, "debug_sin_ordenes": 3}
CAMPOS_CMD = ["lc_t", "lc_q", "lc_dq", "lc_kp", "lc_kd", "lc_tau", "lc_mode",
              "arm_t", "arm_q", "arm_dq", "arm_kp", "arm_kd", "arm_tau", "arm_w",
              "loco_v", "loco_t_fin", "fsm"]


class SimNode(Node):
    def __init__(self):
        super().__init__("h1_2_sim")
        dp = self.declare_parameter
        dp("escena", "faja")
        dp("modelo", "")              # ruta a un MJCF; si se da, manda sobre `escena`
        dp("banda", False)            # cinta elastica enganchada al arrancar (solo base flotante)
        # marcha cinematica: la base sigue LocoClient.Move como solido rigido (no camina). Solo base flotante.
        dp("marcha", "ninguna")       # ninguna | cinematica | politica (unitree_rl_gym, camina de verdad)
        dp("marcha_retardo", 0.30)    # s   (SUPUESTOS hasta medirlos en el robot, reto 6.3)
        dp("marcha_tau", 0.35)        # s, primer orden
        dp("marcha_deriva_deg", 2.0)  # grados/s andando, signo aleatorio por tirada
        dp("marcha_cadencia", 1.43)   # Hz, balanceo de la camara
        dp("marcha_roll_deg", 1.5)
        dp("marcha_pitch_deg", 0.8)
        dp("marcha_z_mm", 8.0)
        dp("marcha_semilla", -1)      # -1 = aleatoria
        dp("fsm_inicial", 201)        # FSM del servicio loco simulado (201 = ya en marcha con el mando)
        dp("emisor_ir", True)         # patron de puntos del emisor de la D435 en las IR
        dp("modo_robot", "auto")
        dp("keyframe", "")
        dp("grasp_assist", False)
        dp("dominio", 1)
        dp("interfaz", "lo")
        dp("hz_lowstate", 500.0)
        dp("hz_interno", 100.0)
        dp("crc_lowstate", False)
        dp("manos_modbus", True)
        dp("realtime", True)
        dp("use_viewer", True)
        dp("hz_viewer", 30.0)
        dp("camaras", True)
        dp("ir", True)
        dp("hz_camara", 15.0)
        dp("hz_joint_states", 50.0)
        dp("contactos", True)
        dp("hz_contactos", 5.0)
        dp("ground_truth", ["mandarina"])
        g = lambda k: self.get_parameter(k).value  # noqa: E731
        log = self.get_logger()

        if g("escena") not in ESCENAS and not g("modelo"):
            raise RuntimeError(f"escena:={g('escena')}: tiene que ser una de {sorted(ESCENAS)}")
        modelo = g("modelo") or recurso("mjcf", ESCENAS[g("escena")])
        log.info(f"MJCF: {modelo}")
        marcha = None
        if g("marcha") == "cinematica":
            marcha = dict(retardo=g("marcha_retardo"), tau=g("marcha_tau"), deriva_deg=g("marcha_deriva_deg"),
                          cadencia=g("marcha_cadencia"), roll_deg=g("marcha_roll_deg"),
                          pitch_deg=g("marcha_pitch_deg"), z_mm=g("marcha_z_mm"),
                          semilla=None if g("marcha_semilla") < 0 else g("marcha_semilla"))
        elif g("marcha") not in ("ninguna", "politica"):
            raise RuntimeError(f"marcha:={g('marcha')}: ninguna | cinematica | politica")
        self.sim = RobotSimulado(modelo, g("modo_robot"), g("keyframe"), grasp_assist=g("grasp_assist"),
                                 banda=g("banda"), marcha=marcha, politica=g("marcha") == "politica",
                                 log=lambda s: log.info(s))
        nq = self.sim.model.nq
        log.info(f"modo_robot={g('modo_robot')} · fisica {1 / self.sim.dt:.0f} Hz · "
                 f"manos simuladas: {sorted(self.sim.manos)} · keyframe '{self.sim.keyframe}'")
        if self.sim.politica is not None:
            log.info(f"MARCHA CON LA POLITICA de unitree_rl_gym (H1-2, 12 juntas, 50 Hz): camina de verdad con "
                     f"LocoClient.Move (max {self.sim.politica.pol.max_cmd.tolist()}); fisica completa, puede caerse. "
                     f"Servicio loco en FSM {g('fsm_inicial')}")
        elif self.sim.marcha is not None:
            m_ = marcha
            log.info(f"MARCHA CINEMATICA (no camina: la base se desplaza con LocoClient.Move como solido rigido): "
                     f"retardo {m_['retardo']} s, tau {m_['tau']} s, deriva {m_['deriva_deg']} grados/s, "
                     f"balanceo {m_['cadencia']} Hz (roll {m_['roll_deg']}, pitch {m_['pitch_deg']} grados, "
                     f"z {m_['z_mm']} mm). Servicio loco en FSM {g('fsm_inicial')}")
        elif self.sim.flotante:
            log.info("BASE FLOTANTE: gravedad, contactos y piernas activas. TF world -> pelvis = verdad de "
                     f"terreno. Cinta elastica {'ENGANCHADA' if self.sim.banda else 'suelta'} "
                     "(viewer: 7 sube, 8 baja, 9 engancha/suelta)")

        # ---------------- memoria compartida y procesos hijos
        self.nombre_shm = f"h12_s2r_{os.getpid()}"
        self.sh = Compartido(self.nombre_shm, nq, crear=True)
        v = self.sh.v
        v["int_q"][:], v["int_kp"][:], v["int_kd"][:] = self.sim.q_interno, self.sim.kp, self.sim.kd
        # el Lock tiene que ser del MISMO contexto que los procesos: uno de 'fork'
        # pasado a un proceso 'spawn' da segfault en Python 3.10, sin aviso
        ctx = mp.get_context("spawn")
        self.ctx = ctx
        self.cerrojo = ctx.Lock()
        self.sh.v["fsm"][0] = g("fsm_inicial")
        self._escribir_estado(0)
        self.args_dds = (self.nombre_shm, nq, self.cerrojo, g("dominio"), g("interfaz"), g("hz_lowstate"),
                         g("hz_interno"), g("crc_lowstate"))
        self.hijos = [ctx.Process(target=proceso_dds, name="h12_dds", daemon=True, args=self.args_dds)]
        if g("camaras"):
            self.hijos.append(ctx.Process(target=proceso_camaras, name="h12_camaras", daemon=True, args=(
                self.nombre_shm, nq, self.cerrojo, modelo, g("ir"), g("hz_camara"), IR_LINEA_BASE,
                g("emisor_ir"))))
        self.hijos.append(ctx.Process(target=proceso_loco, name="h12_loco", daemon=True, args=(
            self.nombre_shm, nq, self.cerrojo, g("dominio"), g("interfaz"), g("fsm_inicial"))))
        for p in self.hijos:
            p.start()
        log.info(f"SDK: dominio {g('dominio')} sobre '{g('interfaz')}' · rt/lowstate {g('hz_lowstate'):.0f} Hz "
                 f"· acepta rt/lowcmd y rt/arm_sdk (proceso aparte)")

        if g("manos_modbus") and self.sim.manos:
            self.srv_manos = self.sim.servir_manos()
            log.info("manos Inspire por Modbus TCP: izq " + J.MANO_IP_SIM["izq"] + ":6000, der "
                     + J.MANO_IP_SIM["der"] + ":6000")

        # ---------------- ROS (lento)
        self.pub_js = self.create_publisher(JointState, "/sim/joint_states", 10)
        self.pub_fuente = self.create_publisher(String, "/sim/fuente", 10)
        self.contactos = g("contactos")
        if self.contactos:
            self.pub_con = self.create_publisher(MarkerArray, "/contacts", 10)
        cuerpos_gt = [b for b in g("ground_truth") if b]
        if self.sim.flotante and "pelvis" not in cuerpos_gt:
            # pose de la base en el mundo en un topico ligero: leerla de /tf obliga a
            # procesar en Python la TF de todo el robot (con RViz abierto, miles/s)
            cuerpos_gt.append("pelvis")
        self.gt = [(b, self.create_publisher(PoseStamped, f"/sim/ground_truth/{b}", 10))
                   for b in cuerpos_gt if self.sim.pose_cuerpo(b) is not None]
        self.tf = TransformBroadcaster(self) if self.sim.flotante else None
        self.create_timer(1.0 / g("hz_joint_states"), self._publicar_lento)
        self.k_con = max(1, int(round(g("hz_joint_states") / g("hz_contactos"))))
        self.n_lento = 0
        self.ultima_fuente = None
        self.vigia_ls = (0.0, time.monotonic() + 5.0)     # (ls_n visto, desde cuando no cambia); 5 s de arranque
        self.reinicios_dds = 0

        self.viewer = None
        if g("use_viewer"):
            try:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.sim.model, self.sim.data,
                                                           key_callback=self.sim.tecla)
                if self.sim.flotante:          # base flotante: la camara del viewer sigue al robot
                    with self.viewer.lock():
                        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                        self.viewer.cam.trackbodyid = self.sim.pelvis
                        self.viewer.cam.distance, self.viewer.cam.elevation = 4.0, -30.0
                        self.viewer.cam.azimuth = 0.0      # desde detras del robot (mira hacia +x)
            except Exception as e:  # sin display
                log.warn(f"sin viewer de MuJoCo: {e}")

        self.stop = threading.Event()
        self.hz_view = g("hz_viewer")
        self.realtime = g("realtime")
        self.hilo = threading.Thread(target=self._fisica, daemon=True, name="fisica")
        self.hilo.start()
        log.info("H1-2 simulado listo. Agente: robot/agente_sdk.py --sim · scripts del SDK con la interfaz 'lo'")

    # ------------------------------------------------------------ fisica
    def _escribir_estado(self, tick, qpos=False):
        sim = self.sim
        q, dq, tau = sim.estado_motores()
        quat, rpy, gyro, acc = sim.imu()
        v = self.sh.v
        with self.cerrojo:
            v["tick"][0] = tick
            v["q"][:], v["dq"][:], v["tau"][:] = q, dq, tau
            v["quat"][:], v["rpy"][:], v["gyro"][:], v["acc"][:] = quat, rpy, gyro, acc
            v["fuente"][0] = CODIGO_FUENTE.get(sim.fuente, 0)
            if qpos:
                v["qpos"][:] = sim.data.qpos
                v["qpos_n"][0] += 1

    def _fisica(self):
        sim, v = self.sim, self.sh.v
        k_view = max(1, int(round(1.0 / (sim.dt * self.hz_view))))
        k_qpos = max(1, int(round(1.0 / (sim.dt * 60.0))))     # qpos para las camaras a 60 Hz
        tick, t_sig, aviso, atraso = 0, time.monotonic(), 0.0, 0
        while not self.stop.is_set():
            ahora = time.monotonic()
            with self.cerrojo:
                c = {k: v[k].copy() for k in CAMPOS_CMD}
            sim.paso(ahora, c)
            tick += 1
            self._escribir_estado(tick, qpos=tick % k_qpos == 0)
            if self.viewer is not None and tick % k_view == 0:
                if self.viewer.is_running():
                    with sim.cerrojo:
                        self.viewer.sync()
                else:
                    self.viewer = None
            if self.realtime:
                t_sig += sim.dt
                espera = t_sig - time.monotonic()
                if espera > 0:
                    time.sleep(espera)
                elif espera < -0.25:
                    atraso += 1
                    if ahora - aviso > 5.0:
                        self.get_logger().warn(f"la simulacion va {-espera:.2f} s por detras del tiempo real "
                                               f"({atraso} re-sincronizaciones)")
                        aviso = ahora
                    t_sig = time.monotonic()

    # ------------------------------------------------------------ ROS lento
    def _publicar_lento(self):
        sim = self.sim
        stamp = self.get_clock().now().to_msg()
        q, dq, tau = sim.estado_motores()
        js = JointState()
        js.header.stamp = stamp
        js.name = list(J.NOMBRES)
        js.position = [float(x) for x in q]
        js.velocity = [float(x) for x in dq]
        js.effort = [float(x) for x in tau]
        for lado, mano in sim.manos.items():
            for k, (j, _) in enumerate(J.DOF_MANO):
                js.name.append(J.PREFIJO[lado] + j)
                js.position.append(float(sim.data.qpos[mano.qadr[k]]))
                js.velocity.append(0.0)
                js.effort.append(float(sim.data.actuator_force[mano.act[k]]))
        self.pub_js.publish(js)
        if self.tf is not None:
            pos, qwxyz = sim.pose_base()
            t = TransformStamped()
            t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, "world", "pelvis"
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (float(x) for x in pos)
            (t.transform.rotation.w, t.transform.rotation.x, t.transform.rotation.y,
             t.transform.rotation.z) = (float(x) for x in qwxyz)
            self.tf.sendTransform(t)
        if sim.fuente != self.ultima_fuente:
            self.get_logger().info(f"manda: {sim.fuente}")
            self.ultima_fuente = sim.fuente
        self.n_lento += 1
        if self.n_lento % 10 == 0:
            self.pub_fuente.publish(String(data=f"{sim.fuente} peso_arm_sdk={sim.peso_arm:.2f}"))
            for p in self.hijos[1:]:
                if not p.is_alive():
                    self.get_logger().error(f"el proceso {p.name} ha muerto (codigo {p.exitcode})",
                                            throttle_duration_sec=10.0)
            self._vigilar_dds()
        if self.contactos and self.n_lento % self.k_con == 0:
            self.pub_con.publish(mensajes.contactos(sim.contactos(), sim.frame_mundo, stamp))
        for b, pub in self.gt:
            pos, qwxyz = sim.pose_cuerpo(b)
            p = PoseStamped()
            p.header.stamp, p.header.frame_id = stamp, sim.frame_mundo
            p.pose.position.x, p.pose.position.y, p.pose.position.z = (float(x) for x in pos)
            (p.pose.orientation.w, p.pose.orientation.x, p.pose.orientation.y,
             p.pose.orientation.z) = (float(x) for x in qwxyz)
            pub.publish(p)

    def _vigilar_dds(self):
        """Si rt/lowstate deja de salir (muerto, o vivo pero parado: paso una vez
        justo despues de matar una sesion anterior y no se pudo reproducir), se pide
        su traza (SIGUSR1 -> faulthandler, sale en este log) y se reinicia."""
        n = float(self.sh.v["ls_n"][0])
        visto, desde = self.vigia_ls
        ahora = time.monotonic()
        if n != visto:
            self.vigia_ls = (n, ahora)
            return
        if ahora - desde < 2.0:
            return
        p = self.hijos[0]
        log = self.get_logger()
        log.error(f"rt/lowstate parado {ahora - desde:.1f} s (proceso dds {'vivo' if p.is_alive() else 'muerto'}, "
                  f"{int(n)} publicados): traza y reinicio")
        if p.is_alive():
            os.kill(p.pid, signal.SIGUSR1)
            time.sleep(0.3)
            p.terminate()
            p.join(timeout=2.0)
            if p.is_alive():
                p.kill()
        self.reinicios_dds += 1
        self.hijos[0] = self.ctx.Process(target=proceso_dds, name="h12_dds", daemon=True, args=self.args_dds)
        self.hijos[0].start()
        self.vigia_ls = (n, time.monotonic() + 5.0)
        log.warn(f"proceso dds reiniciado ({self.reinicios_dds}a vez)")

    def cerrar(self):
        self.stop.set()
        self.hilo.join(timeout=2.0)
        with self.cerrojo:
            self.sh.v["parar"][0] = 1.0
        for p in self.hijos:
            p.join(timeout=3.0)
            if p.is_alive():
                p.terminate()
        if self.viewer is not None:
            self.viewer.close()
        for s in getattr(self, "srv_manos", []):
            s.shutdown()
        self.sh.cerrar()


def main(args=None):
    rclpy.init(args=args)
    nodo = SimNode()
    try:
        rclpy.spin(nodo)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        nodo.cerrar()
        nodo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
