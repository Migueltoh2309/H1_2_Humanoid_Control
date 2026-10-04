"""Bloque SUPERVISOR (reto, sec. 5, 6.4 y 10): maquina de estados explicita. Es el
UNICO bloque que decide Move / StopMove.

  ESPERA         robot quieto hasta `frames_listo` medidas buenas seguidas
  SEGUIMIENTO    aplica el control
  LINEA_PERDIDA  sin medida buena: sigue la linea recordada (o sostiene el rumbo de
                 la IMU) despacio; vuelve a SEGUIMIENTO si la recupera; para si pasa
                 de perdida_max_m o perdida_max_s (6.4.3)
  GIRO_ESQUINA   nivel 4: StopMove, pausa, giro sobre el eje de +-90 grados cerrado con
                 el yaw de la IMU (constantes de cuadrado.py), y vuelta a SEGUIMIENTO
  LLEGADA        la barra de fin ya se vio: se sigue hasta que la punta del pie esta
                 sobre ella segun la estimacion (la barra entra en la zona ciega
                 antes: 6.4.4) y StopMove
  FIN            parado al terminar (llegada o linea perdida)
  PARADA         parada de emergencia del vigilante: camara o IMU sin datos,
                 roll/pitch > 20 grados, motor en fallo, excepcion o Ctrl+C
"""
import math

from .control import saturar
from .estimacion import envolver
from .tipos import EstadoLinea, Orden

ESPERA, SEGUIMIENTO, PERDIDA, GIRO, LLEGADA, FIN, PARADA = (
    "ESPERA", "SEGUIMIENTO", "LINEA_PERDIDA", "GIRO_ESQUINA", "LLEGADA", "FIN", "PARADA")


class Supervisor:
    def __init__(self, cfg, log=print):
        self.s = cfg["supervisor"]
        self.conf_min = cfg["estimacion"]["confianza_min"]
        self.log = log
        self.estado = ESPERA
        self.motivo = ""
        self.buenas = 0
        self.t_estado = None
        self.giro = None
        self.eventos = []
        self.arrancar = None            # callback: el principal reinicia la estimacion al arrancar

    def _cambiar(self, t, nuevo, motivo=""):
        if nuevo != self.estado:
            self.eventos.append({"t": round(t, 3), "de": self.estado, "a": nuevo, "motivo": motivo})
            self.log(f"[supervisor] {self.estado} -> {nuevo}" + (f" ({motivo})" if motivo else ""))
        self.estado, self.t_estado, self.motivo = nuevo, t, motivo

    # ------------------------------------------------------------ vigilante
    def vigilar(self, t, salud):
        v = self.s["vigilante"]
        if salud.get("excepcion"):
            return f"excepcion: {salud['excepcion']}"
        if t - salud.get("t_camara", t) > v["camara_s"]:
            return f"camara sin fotogramas {t - salud['t_camara']:.2f} s"
        if t - salud.get("t_imu", t) > v["imu_s"]:
            return f"rt/lowstate sin datos {t - salud['t_imu']:.2f} s"
        if not salud.get("motores_ok", True):
            return "motor en fallo"
        lim = math.radians(v["inclinacion_deg"])
        if abs(salud.get("roll", 0.0)) > lim or abs(salud.get("pitch", 0.0)) > lim:
            return f"inclinacion roll {math.degrees(salud['roll']):+.1f} pitch {math.degrees(salud['pitch']):+.1f}"
        return None

    # ------------------------------------------------------------ ciclo
    def paso(self, t, est: EstadoLinea, medida_buena: bool, orden: Orden, salud) -> Orden:
        if self.t_estado is None:
            self.t_estado = t
        if self.estado in (FIN, PARADA):
            return Orden(parar=True, motivo=self.estado)
        razon = self.vigilar(t, salud)
        if razon:
            self._cambiar(t, PARADA, razon)
            return Orden(parar=True, motivo=razon)
        s = self.s

        if self.estado == ESPERA:
            self.buenas = self.buenas + 1 if medida_buena else 0
            if self.buenas >= s["frames_listo"]:
                if self.arrancar:
                    self.arrancar()          # referencia de rumbo tomada AHORA, no al lanzar el programa
                self._cambiar(t, SEGUIMIENTO, "linea vista")
            return Orden(parar=True, motivo="espera")

        if self.estado == GIRO:
            return self._girar(t, est, salud)

        # barra de fin: se ignora al principio (el borde del cuadro de inicio parece una barra)
        if est.barra is not None and est.recorrido > s["ignorar_barra_m"] and self.estado != LLEGADA \
                and est.barra < s.get("llegada_m", 1.2):
            self._cambiar(t, LLEGADA, f"barra a {est.barra:.2f} m")
        if self.estado == LLEGADA:
            falta = (est.barra if est.barra is not None else 0.0) - s["pie_delante"]
            if falta <= 0.0:
                self._cambiar(t, FIN, "pie sobre la barra (estimado)")
                return Orden(parar=True, motivo="fin")
            o = orden if est.valida else Orden(vx=0.1)
            o.vx = min(o.vx, max(0.08, 0.6 * falta))      # frenar al acercarse
            o.motivo = f"llegada, faltan {falta:.2f} m"
            return o

        # esquina (nivel 4): cuando el centro del robot llega a ella
        if est.esquina is not None and est.esquina[0] <= s.get("adelanto_esquina", 0.0):
            self._cambiar(t, GIRO, f"esquina hacia la {'izquierda' if est.esquina[1] > 0 else 'derecha'}")
            self.giro = {"lado": est.esquina[1], "fase": "pausa", "t": t, "yaw0": est.yaw, "t_ok": None}
            return Orden(parar=True, motivo="esquina: pausa")

        # con una esquina confirmada cerca, que la linea se acabe es lo esperado
        hacia_esquina = est.esquina is not None and est.esquina[0] < s.get("esquina_cerca_m", 1.0)
        perdida = ((not medida_buena and est.sin_linea_s > 0.3) or not est.valida) and not hacia_esquina
        if perdida:
            if est.sin_linea_m > s["perdida_max_m"] or est.sin_linea_s > s["perdida_max_s"]:
                self._cambiar(t, FIN, f"linea perdida {est.sin_linea_m:.2f} m / {est.sin_linea_s:.1f} s")
                return Orden(parar=True, motivo="linea perdida")
            if self.estado != PERDIDA:
                self._cambiar(t, PERDIDA, "sin medida buena")
            o = orden if est.valida else Orden(vx=s["vx_perdida"])
            o.vx = min(o.vx, s["vx_perdida"])
            o.motivo = "buscando"
            return o
        if self.estado == PERDIDA:
            self._cambiar(t, SEGUIMIENTO, "linea recuperada")
        return orden

    def _girar(self, t, est, salud):
        g, gc = self.giro, self.s["giro_esquina"]
        if g["fase"] == "pausa":
            if t - g["t"] < gc["pausa_s"]:
                return Orden(parar=True, motivo="esquina: pausa (inercia)")
            g["fase"], g["t"] = "giro", t
            g["objetivo"] = envolver(g["yaw0"] + g["lado"] * math.pi / 2)
        err = envolver(g["objetivo"] - est.yaw)
        if abs(math.degrees(err)) < gc["tolerancia_deg"]:
            g["t_ok"] = g["t_ok"] or t
            if t - g["t_ok"] >= gc["estable_s"]:
                self._cambiar(t, SEGUIMIENTO, "giro completado")
                if self.on_giro:
                    self.on_giro()
                return Orden(parar=True, motivo="giro hecho")
        else:
            g["t_ok"] = None
        if t - g["t"] > gc["max_s"]:
            self._cambiar(t, PARADA, "el giro no termina")
            return Orden(parar=True, motivo="giro sin terminar")
        mag = max(gc["vyaw_min"], min(self.s.get("vyaw_giro_max", 0.4), gc["kp"] * abs(err)))
        return Orden(0.0, 0.0, math.copysign(mag, err), motivo=f"giro, faltan {math.degrees(err):+.1f} grados")

    on_giro = None
