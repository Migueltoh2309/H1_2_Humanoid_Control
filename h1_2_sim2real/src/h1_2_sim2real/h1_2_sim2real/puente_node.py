"""Puente ROS 2 <-> agente SDK. EL MISMO para el simulador y para el robot real:
solo cambia la IP del agente (127.0.0.1 si el agente corre en este PC contra el
simulador, 192.168.0.143 si corre en el robot).

  /joint_cmd   (sensor_msgs/JointState, sub)  misma interfaz que h1_2_mujoco_sim_bridge:
               por NOMBRE y parcial; position = q, velocity = dq (opcional),
               effort = par feedforward (opcional). Los dedos se mandan con su
               nombre del URDF (L_index_proximal_joint, R_thumb_proximal_yaw_joint...)
               en radianes; se convierten al angulo 0-1000 de la Inspire.
  /joint_states (pub)  27 juntas + 12 dedos, del rt/lowstate y del Modbus de las manos
  /h1_2/imu     (sensor_msgs/Imu, pub)
  /h1_2/agente  (std_msgs/String, pub, 2 Hz)   modo, peso de arm_sdk, banderas, latencia
  /h1_2/soltar  (std_msgs/Empty, sub)          el agente devuelve el control y termina

Las juntas que nunca han aparecido en /joint_cmd no se mandan: el agente las
sostiene en la postura que midio al arrancar. El puente manda ordenes a hz_orden
aunque no lleguen /joint_cmd nuevos (latido): si el puente o la WiFi caen, el
watchdog del agente sostiene y, en arm_sdk, suelta.
"""
import json
import socket
import threading
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Empty, String

from .comun import J, P

NOMBRES_BANDERAS = [(P.B_LOWSTATE, "lowstate"), (P.B_ORDENES, "ordenes"), (P.B_MANOS, "manos"),
                    (P.B_SOLTANDO, "soltando"), (P.B_ACTIVO, "activo"), (P.B_FALLO, "fallo")]


class PuenteNode(Node):
    def __init__(self):
        super().__init__("h1_2_puente")
        dp = self.declare_parameter
        dp("agente_ip", "127.0.0.1")
        dp("puerto_orden", P.PUERTO_ORDEN)
        dp("puerto_estado", P.PUERTO_ESTADO)
        dp("hz_orden", 50.0)
        dp("joint_cmd_topic", "/joint_cmd")
        dp("joint_state_topic", "/joint_states")
        dp("caduca_ff", 0.2)             # s: dq/par de /joint_cmd valen hasta entonces
        dp("err_seguimiento", 0.20)      # rad: aviso si una junta no sigue la consigna
        g = lambda k: self.get_parameter(k).value  # noqa: E731
        self.agente = (g("agente_ip"), g("puerto_orden"))
        self.err_seg = g("err_seguimiento")

        self.cerrojo = threading.Lock()
        self.q = [0.0] * J.N
        self.dq = [0.0] * J.N
        self.tau = [0.0] * J.N
        self.mascara = 0
        self.ang = [1000] * 12
        self.mascara_mano = 0
        self.seq = 0
        self.soltar_hasta = 0.0
        self.t_cmd = 0.0
        self.caduca_ff = g("caduca_ff")
        self.indice_mano = {n: k for k, n in enumerate(J.NOMBRES_MANO)}

        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.bind(("0.0.0.0", g("puerto_estado")))
        self.rx.settimeout(0.2)

        self.pub_js = self.create_publisher(JointState, g("joint_state_topic"), 10)
        self.pub_imu = self.create_publisher(Imu, "/h1_2/imu", 10)
        self.pub_ag = self.create_publisher(String, "/h1_2/agente", 10)
        self.create_subscription(JointState, g("joint_cmd_topic"), self._al_cmd, 10)
        self.create_subscription(Empty, "/h1_2/soltar", self._al_soltar, 10)
        self.create_timer(1.0 / g("hz_orden"), self._mandar)
        self.create_timer(0.5, self._resumen)

        self.ultimo = None
        self.t_ultimo = 0.0
        self.n_rx = 0
        self.desde_err = {}
        self.desconocidos = set()
        self.vivo = True
        threading.Thread(target=self._recibir, daemon=True, name="udp_estado").start()
        self.get_logger().info(f"puente: {g('joint_cmd_topic')} -> UDP {self.agente[0]}:{self.agente[1]} · "
                               f"estado en UDP {g('puerto_estado')} -> {g('joint_state_topic')}")

    # ------------------------------------------------------------ ordenes
    def _al_cmd(self, msg):
        n = len(msg.name)
        pos, vel, eff = list(msg.position), list(msg.velocity), list(msg.effort)
        if len(pos) != n:
            self.get_logger().warn("/joint_cmd: position no tiene un valor por nombre; ignorado", throttle_duration_sec=5.0)
            return
        with self.cerrojo:
            self.t_cmd = time.time()
            for k, nombre in enumerate(msg.name):
                i = J.INDICE.get(nombre)
                if i is not None:
                    self.q[i] = pos[k]
                    self.dq[i] = vel[k] if len(vel) == n else 0.0
                    self.tau[i] = eff[k] if len(eff) == n else 0.0
                    self.mascara |= 1 << i
                    continue
                m = self.indice_mano.get(nombre)
                if m is not None:
                    self.ang[m] = J.rad_a_angulo(m % 6, pos[k])
                    self.mascara_mano |= 1 << m
                elif nombre not in self.desconocidos:
                    self.desconocidos.add(nombre)
                    self.get_logger().warn(f"/joint_cmd: junta desconocida '{nombre}' (ignorada)")

    def _al_soltar(self, _msg):
        self.get_logger().warn("SOLTAR: el agente devuelve el control y termina")
        self.soltar_hasta = time.time() + 1.0

    def _mandar(self):
        soltar = time.time() < self.soltar_hasta
        with self.cerrojo:
            # dq y par feedforward son de UNA consigna: si /joint_cmd deja de llegar,
            # el latido sigue mandando solo la posicion (sostener, sin empujar)
            viejo = time.time() - self.t_cmd > self.caduca_ff
            datos = P.empaquetar_orden(self.seq, P.SOLTAR if soltar else P.NORMAL, self.mascara,
                                       self.mascara_mano, self.q, [0.0] * J.N if viejo else self.dq,
                                       [0.0] * J.N if viejo else self.tau, self.ang)
        self.seq += 1
        try:
            self.tx.sendto(datos, self.agente)
        except OSError as e:
            self.get_logger().warn(f"no se puede mandar al agente: {e}", throttle_duration_sec=5.0)

    # ------------------------------------------------------------ estado
    def _recibir(self):
        while self.vivo:
            try:
                datos, _ = self.rx.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                return
            e = P.desempaquetar_estado(datos)
            if e is None:
                continue
            self.ultimo, self.t_ultimo = e, time.time()
            self.n_rx += 1
            self._publicar(e)

    def _publicar(self, e):
        stamp = self.get_clock().now().to_msg()
        js = JointState()
        js.header.stamp = stamp
        js.name = list(J.NOMBRES) + list(J.NOMBRES_MANO)
        js.position = list(e["q"]) + [J.angulo_a_rad(k % 6, a) for k, a in enumerate(e["ang_mano"])]
        js.velocity = list(e["dq"]) + [0.0] * 12
        js.effort = list(e["tau"]) + [float(f) for f in e["fuerza_mano"]]   # dedos: gramos-fuerza
        self.pub_js.publish(js)
        imu = Imu()
        imu.header.stamp, imu.header.frame_id = stamp, "pelvis"
        imu.orientation.w, imu.orientation.x, imu.orientation.y, imu.orientation.z = e["quat"]
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = e["gyro"]
        imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z = e["acc"]
        self.pub_imu.publish(imu)
        # aviso si una junta comandada no sigue (p.ej. robot que no esta en el modo correcto)
        ahora = time.time()
        for i in range(J.N):
            if not (self.mascara >> i) & 1:
                continue
            if abs(e["q_obj"][i] - e["q"][i]) > self.err_seg:
                t0 = self.desde_err.setdefault(i, ahora)
                if ahora - t0 > 2.0:
                    self.get_logger().warn(
                        f"{J.NOMBRES[i]} no sigue la consigna: objetivo {e['q_obj'][i]:+.2f}, "
                        f"medido {e['q'][i]:+.2f} rad (¿modo del robot? ¿choque?)", throttle_duration_sec=5.0)
            else:
                self.desde_err.pop(i, None)

    def _resumen(self):
        e = self.ultimo
        if e is None or time.time() - self.t_ultimo > 1.0:
            self.pub_ag.publish(String(data=json.dumps({"conectado": False})))
            self.get_logger().warn(f"sin estado del agente en {self.agente[0]} (¿agente arrancado?)",
                                   throttle_duration_sec=10.0)
            return
        info = {"conectado": True, "modo": P.NOMBRE_MODO.get(e["modo"], "?"), "peso_arm_sdk": round(e["peso"], 3),
                "banderas": [n for b, n in NOMBRES_BANDERAS if e["banderas"] & b],
                "latencia_ms": round((time.time_ns() - e["t_ns"]) / 1e6, 1), "mode_machine": e["mode_machine"],
                "juntas_comandadas": [J.NOMBRES[i] for i in range(J.N) if (self.mascara >> i) & 1]}
        self.pub_ag.publish(String(data=json.dumps(info)))

    def cerrar(self):
        self.vivo = False
        self.rx.close()


def main(args=None):
    rclpy.init(args=args)
    nodo = PuenteNode()
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
