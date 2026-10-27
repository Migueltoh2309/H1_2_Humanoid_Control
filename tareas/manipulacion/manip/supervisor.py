"""Bloque SUPERVISOR (sec. 5 y 10): máquina de estados explícita, ÚNICO que cambia el peso de
rt/arm_sdk y único que da por terminado un objeto.

    ESPERA -> (palabra) -> ENTRADA (rampa de peso 12 s) -> OBSERVAR -> PLANIFICAR -> APROXIMAR
    -> ASENTAR (lazo con encoders + corrección del cuerpo por la IMU) -> DESCENDER -> CERRAR
    -> VERIFICAR (dedos parados lejos de 0 con fuerza) -> ELEVAR -> TRANSPORTAR -> BAJAR
    -> SOLTAR (a la preforma) -> RETIRAR -> VOLVER -> OBSERVAR (¿quedó en la bandeja?) ...
    -> SALIDA (a la postura del principio y rampa de peso >= 5 s hasta 0.00) -> ESPERA

Desde cualquier estado, `Parada` (vigilante: paso/giro/inclinación, error de seguimiento, FSM,
cámara, mano sin respuesta, fuerza > corte; o Ctrl+C / excepción) -> SALIDA_SEGURA: congelar el
brazo (la consigna se queda donde estaba), abrir la mano (a la preforma), subirla a la altura de
seguridad y solo entonces bajar el peso a 0 (sec. 10).
"""
import math
import time
from dataclasses import replace

import numpy as np

from . import cinematica as C
from .brazo import Parada
from .compensacion import por_cuerpo, rot_rpy
from .mano import ErrorMano
from .robot_io import INDICES
from .seguridad import Vigilante

REPOSO_MANO = [1000, 1000, 1000, 1000, 1000, 1000]


class _Contacto(Exception):
    """El índice notó la mesa (nivel 1): corta el sondeo."""


class Supervisor:
    def __init__(self, estado, fsm, brazo, manos, fuente, percepcion, plan, comp, est, catalogo, cfg_sup,
                 registro, log=print, simulacro=False):
        self.estado, self.fsm, self.brazo, self.manos = estado, fsm, brazo, manos
        self.fuente, self.percep, self.plan, self.comp = fuente, percepcion, plan, comp
        self.est, self.cat, self.cfg, self.reg = est, catalogo, cfg_sup, registro
        self.log, self.simulacro = log, simulacro
        self.vig = Vigilante(estado, fsm, brazo, fuente, estado.rpy()[2])
        self.n_ciclo = 0
        self.agarre_actual = {}           # lado -> Agarre que lleva puesto (para la salida segura)
        self.q_ini = {lado: estado.brazo(lado) for lado in C.LADOS}
        self.q_obs = {k: np.asarray(v, float) for k, v in plan.q_obs.items()}
        self.al_estado = None             # gancho de pruebas (el instructor del simulador)
        self.sobre_corte = {lado: 0 for lado in C.LADOS}
        self.tapada = 0
        self.giro_simulado = None          # (instante, grados): prueba del hito 3
        self.terminados = []

    # ------------------------------------------------------------ vigilancia (cada ciclo)
    def comprobar(self):
        if self.giro_simulado is not None and time.monotonic() >= self.giro_simulado[0]:
            # hito 3: "simular un giro del cuerpo" (inyección en la guiñada de referencia)
            self.vig.yaw0 -= math.radians(self.giro_simulado[1])
            self.reg.evento("giro_simulado", grados=self.giro_simulado[1])
            self.log(f"  [prueba] giro del cuerpo simulado: {self.giro_simulado[1]:.0f} grados")
            self.giro_simulado = None
        self.vig()
        self.n_ciclo += 1
        if self.n_ciclo % 25 == 0:
            self._camara_tapada()
        if self.n_ciclo % 5 == 0:
            lado = C.LADOS[(self.n_ciclo // 5) % 2]
            e = self.manos[lado].estado()
            if not e.responde:
                raise Parada(f"la mano {lado} no responde")
            self.reg.mano(lado, e.angulos, e.fuerzas)
            # corte por fuerza con DOS lecturas seguidas por encima (~0.2 s): una lectura suelta
            # disparatada (3000 gf en un dedo con los vecinos a 0, sec. 7) no es una fuerza
            if np.nanmax(e.fuerzas) > self.cfg["corte_gf"]:
                self.sobre_corte[lado] += 1
                self.reg.evento("fuerza_alta", lado=lado, fuerzas=e.fuerzas, lecturas=self.sobre_corte[lado])
                if self.sobre_corte[lado] >= 2:
                    raise Parada(f"mano {lado}: {np.nanmax(e.fuerzas):.0f} gf > corte {self.cfg['corte_gf']}")
            else:
                self.sobre_corte[lado] = 0

    def _camara_tapada(self):
        """La cámara sigue dando fotogramas pero tapada (una mano encima): casi negra y sin
        profundidad. Dos comprobaciones seguidas (~1 s) -> parada."""
        f = self.fuente.fotograma(timeout=0.05, despues_de=0.0)
        if f is None:
            return
        prof = np.asarray(f.prof)
        valida = float(np.mean(np.isfinite(prof) & (prof > 0.2)))
        brillo = float(np.mean(f.ir)) if f.ir is not None else 255.0
        if valida < self.cfg["camara_prof_min"] or brillo < self.cfg["camara_brillo_min"]:
            self.tapada += 1
            if self.tapada >= 2:
                raise Parada(f"cámara tapada (profundidad válida {100 * valida:.0f} %, brillo {brillo:.0f})")
        else:
            self.tapada = 0

    def esperar(self, segundos):
        fin = time.monotonic() + segundos
        while time.monotonic() < fin:
            self.comprobar()
            time.sleep(0.02)

    def estado_sup(self, nombre, objeto=""):
        self.reg.estado_sup(nombre, objeto)
        if self.al_estado is not None:
            try:
                self.al_estado(nombre, objeto, self.terminados)
            except Exception as e:                   # una prueba no puede tumbar la tirada
                self.log(f"  (al_estado: {e})")

    # ------------------------------------------------------------ mano
    def _mano(self, lado, accion, ag=None):
        # en el simulacro la mano SÍ se mueve (abre y cierra en el aire: el brazo no se publica)
        m = self.manos[lado]
        if accion == "preforma":
            m.preforma(ag.preforma)
        elif accion == "cerrar":
            m.poner_force_set(ag.force_set)        # SIEMPRE antes de cerrar (sec. 10)
            if ag.previo:
                m.mover(ag.previo)
                self.esperar(self.cfg["t_previo"])
            m.mover(ag.cierre)
        elif accion == "soltar":
            # a la preforma (abrir del todo mete el pulgar en la mesa) o a la postura de suelta propia
            m.preforma(ag.suelta or ag.preforma)
        elif accion == "reposo":
            # lejos de la mesa: todo abierto y el pulgar recogido a lo largo de la mano (como al
            # empezar). Con la preforma en la observación el pulgar cuelga 9 cm, tapa la bandeja y
            # la percepción lo toma por un objeto
            m.mover(REPOSO_MANO)

    # ------------------------------------------------------------ percepción
    def observar(self, nombre="observar"):
        """Brazos en la pose de observación, quietos; N fotogramas fusionados."""
        self.estado_sup("OBSERVAR")
        lejos = max(float(np.max(np.abs(self.brazo.consigna(ld) - self.q_obs[ld]))) for ld in C.LADOS)
        if lejos > 0.02:
            self.brazo.mover(self.q_obs, comprobar=self.comprobar)
        self.esperar(self.cfg["t_quieto"])
        fs, t = [], time.monotonic()
        R0 = rot_rpy(self.estado.rpy())
        for _ in range(self.cfg["fotogramas"]):
            f = self.fuente.fotograma(timeout=2.0, despues_de=t)
            if f is None:
                raise Parada("la cámara no da imágenes")
            t = f.t
            q = self.estado.q()
            fs.append(replace(f, q_brazos=np.concatenate([q[13:20], q[20:27]])))
            self.comprobar()
        lista = self.percep(fs)
        self.ultimo_fotograma = fs[-1]
        self.reg.imagen(nombre, fs[-1].color if fs[-1].color is not None else fs[-1].ir)
        self.reg.evento("percepcion", ms=lista.ms, plano=lista.plano_mesa,
                        objetos=[{"clase": o.clase, "centro": o.centro, "guinada": o.guinada,
                                  "confianza": o.confianza, "visible": o.fraccion_visible} for o in lista.objetos])
        self.log("  visto: " + ", ".join(f"{o.clase} ({o.confianza:.2f}) en "
                                         f"({o.centro[0]:.3f}, {o.centro[1]:+.3f})" for o in lista.objetos))
        return lista, R0

    def en_bandeja(self, lista, clase, ocupadas_antes=()):
        """Colocado = ya no está en la zona de recogida y hay algo NUEVO dentro de su bandeja. No
        se mira la etiqueta de lo que hay en la bandeja: sus paredes recortan la huella y los
        rasgos no valen (el cubo dentro de A salió como pelota)."""
        letra = self.cat[clase]["bandeja"]
        b = self.est.bandejas[letra]
        dentro = any(b.contiene(o.centro[0], o.centro[1], margen=-0.01) for o in lista.objetos)
        sigue = any(o.clase == clase and o.confianza >= self.cfg["conf_min"] and self.en_zona(o)
                    for o in lista.objetos)
        return dentro and not sigue and letra not in ocupadas_antes

    def en_zona(self, o, margen=-0.03):
        """Candidato a coger: sobre el tablero y fuera de las bandejas (un objeto que rodó fuera de
        la zona de recogida se reintenta; el planificador dice si se alcanza)."""
        x, y = o.centro[0], o.centro[1]
        if not self.est.tablero.contiene(x, y, margen=0.02):
            return False
        return not any(b.contiene(x, y, margen=-0.01) for b in self.est.bandejas.values())

    def ocupadas(self, lista):
        """Bandejas con algo DENTRO antes de soltar: el centro a 1.5 cm o más de las paredes (un
        objeto de 4-7 cm apoyado en la bandeja lo tiene). Con -1 cm, la pared cercana de una
        bandeja vacía, vista como 'desconocido' en su borde, la daba por ocupada y el prisma
        recién dejado no contaba como colocado (nivel 3 s1, 2026-10-15)."""
        return {k for k, b in self.est.bandejas.items()
                if any(b.contiene(o.centro[0], o.centro[1], margen=0.015) for o in lista.objetos)}

    # ------------------------------------------------------------ un objeto
    def ejecutar(self, p, R0):
        """Ejecuta un Plan. Devuelve (cogido, motivo)."""
        lado, ag, clase = p.mano, p.agarre, p.objeto.clase
        t = {tr.nombre: tr for tr in p.tramos}
        ir = lambda nombre, ruta, dur=None: self.brazo.seguir_ruta(lado, ruta, duracion=dur,  # noqa: E731
                                                               comprobar=self.comprobar)
        self.agarre_actual[lado] = ag
        self._mano(lado, "preforma", ag)
        self.estado_sup("APROXIMAR", clase)
        self.vig.mirar_seguimiento = not self.simulacro      # en el simulacro el brazo no se mueve
        ir("aproximar", t["aproximar"].q_ruta)
        self.estado_sup("ASENTAR", clase)
        err = self.brazo.asentar(lado, comprobar=self.comprobar)
        # el cuerpo se inclina al extender el brazo: el objeto, medido en la observación, está
        # en otro sitio del marco del torso (6.3.4); se rehace el descenso hacia él
        baja, sube = t["descender"], t["elevar"]
        R1 = rot_rpy(self.estado.rpy())
        off = self.plan.off(lado, clase)
        T = C.fk(lado, baja.q_ruta[-1])
        p_ag = T[:3, :3] @ off + T[:3, 3]
        p_nuevo = por_cuerpo(p_ag, R0, R1, self.cfg["pivote_cuerpo"])
        nuevo = self.plan.rehacer_descenso(p, p_nuevo, self.brazo.consigna(lado))
        if nuevo is not None:
            baja, sube = nuevo
        self.reg.evento("asentar", objeto=clase, error_lazo_deg=math.degrees(err),
                        correccion_cuerpo_mm=1000 * np.linalg.norm(p_nuevo - p_ag),
                        inclinacion_deg=math.degrees(math.acos(min(1.0, (np.trace(R1.T @ R0) - 1) / 2))))
        self.estado_sup("DESCENDER", clase)
        ir("descender", baja.q_ruta)
        self.estado_sup("CERRAR", clase)
        self._mano(lado, "cerrar", ag)
        self.esperar(self.cfg["t_cerrar"])
        self.estado_sup("VERIFICAR", clase)
        if self.simulacro:
            ok, det = True, {"simulacro": True}             # en el aire no hay nada que coger
        else:
            ok, det = self.manos[lado].agarre()
        self.reg.evento("agarre", objeto=clase, ok=ok, **det)
        f = self.fuente.fotograma(timeout=1.0)
        if f is not None:
            self.reg.imagen(f"agarre_{clase}", f.color if f.color is not None else f.ir)
        if not ok:
            # sin agarre: soltar (preforma), subir por la misma recta y volver a observar
            self.log(f"  {clase}: la mano no lo tiene ({det.get('tocan')}); se retira")
            self._mano(lado, "soltar", ag)
            self.esperar(self.cfg["t_soltar"])
            self.estado_sup("ELEVAR", clase)
            ir("elevar", sube.q_ruta)
            self._mano(lado, "reposo")
            self.estado_sup("VOLVER", clase)
            self.brazo.mover({lado: self.q_obs[lado]}, comprobar=self.comprobar)
            self.agarre_actual.pop(lado, None)
            return False, "la mano no detecta el objeto"
        self.comp.poner_carga(lado, self.cat[clase]["masa"], off)
        self.estado_sup("ELEVAR", clase)
        ir("elevar", sube.q_ruta)
        # segunda comprobación, ya en el aire: si se escurrió al subir (la pelota), los dedos se
        # han cerrado del todo y no se transporta una mano vacía
        if not self.simulacro:
            self.esperar(0.3)
            ok2, det2 = self.manos[lado].agarre(dedos_min=self.plan.dedos_elevado.get(clase, 1))
            self.reg.evento("agarre_elevado", objeto=clase, ok=ok2, **det2)
            if not ok2:
                self.log(f"  {clase}: agarre débil o perdido al elevar ({det2.get('tocan')}); "
                         "se deja y se vuelve a observar")
                bajado = self.manos[lado].agarre()[0]
                if bajado:
                    # aún lo tiene, pero mal: se baja por la misma recta y se deja en su sitio
                    # (soltarlo en el aire lo tira y, la pelota, rueda fuera de la zona)
                    self.estado_sup("DESCENDER", clase)
                    ir("descender", sube.q_ruta[::-1])
                self.comp.poner_carga(lado, 0.0)
                self._mano(lado, "soltar", ag)
                self.esperar(self.cfg["t_soltar"])
                if bajado:
                    self.estado_sup("ELEVAR", clase)
                    ir("elevar", sube.q_ruta)
                self._mano(lado, "reposo")
                self.estado_sup("VOLVER", clase)
                self.brazo.mover({lado: self.q_obs[lado]}, comprobar=self.comprobar)
                self.agarre_actual.pop(lado, None)
                return False, "se escurrió al elevar"
        self.estado_sup("TRANSPORTAR", clase)
        ir("transportar", t["transportar"].q_ruta, self.plan.duracion_transporte.get(clase))
        # tercera comprobación: si se cayó por el camino (la pelota, en la mano izquierda), no se
        # baja ni se suelta en el aire; se vuelve a observar dónde ha quedado
        if not self.simulacro:
            ok3, det3 = self.manos[lado].agarre()
            self.reg.evento("agarre_transportado", objeto=clase, ok=ok3, **det3)
            if not ok3:
                self.log(f"  {clase}: se cayó en el transporte ({det3.get('tocan')}); se vuelve a observar")
                self.comp.poner_carga(lado, 0.0)
                self._mano(lado, "reposo")
                self.estado_sup("VOLVER", clase)
                self.brazo.mover({lado: self.q_obs[lado]}, comprobar=self.comprobar)
                self.agarre_actual.pop(lado, None)
                return False, "se cayó en el transporte"
        # otra vez el cuerpo: con el brazo extendido hacia la bandeja o la cruz también se inclina
        self.estado_sup("ASENTAR", clase)
        self.brazo.asentar(lado, comprobar=self.comprobar)
        baja_s, retira = t["bajar"], t["retirar"]
        p_su = self.plan.punto_suelta(p)
        p_su_n = por_cuerpo(p_su, R0, rot_rpy(self.estado.rpy()), self.cfg["pivote_cuerpo"])
        nuevo = self.plan.rehacer_suelta(p, p_su_n, self.brazo.consigna(lado))
        if nuevo is not None:
            baja_s, retira = nuevo
        self.reg.evento("asentar_suelta", objeto=clase, correccion_cuerpo_mm=1000 * np.linalg.norm(p_su_n - p_su))
        self.estado_sup("BAJAR", clase)
        ir("bajar", baja_s.q_ruta)
        self.estado_sup("SOLTAR", clase)
        self._mano(lado, "soltar", ag)
        self.esperar(self.cfg["t_soltar"])
        self.comp.poner_carga(lado, 0.0)
        self.estado_sup("RETIRAR", clase)
        ir("retirar", retira.q_ruta)
        self._mano(lado, "reposo")
        self.estado_sup("VOLVER", clase)
        ir("volver", t["volver"].q_ruta)
        self.agarre_actual.pop(lado, None)
        return True, ""

    # ------------------------------------------------------------ niveles
    def _planear(self, o, lista, mano=None, destino=None):
        q = {lado: self.brazo.consigna(lado) for lado in C.LADOS}
        otros = [x for x in lista.objetos if x is not o]
        t_pl = time.monotonic()
        p = self.plan.planificar(o, q, bandeja=destino, mano=mano, otros=otros, plano=lista.plano_mesa)
        self.reg.evento("plan", objeto=o.clase, ok=p.ok, motivo=p.motivo, mano=p.mano,
                        destino=destino if destino is not None else p.bandeja, s=time.monotonic() - t_pl)
        return p

    def elegir(self, lista, pendientes, intentos, cambio_mano):
        """El siguiente plan: primero los objetos que su mano coge directamente (los cercanos
        antes: dejan sitio); si ninguno y se permite, el cambio de mano por la mesa (nivel 3):
        la mano del lado del objeto lo deja en un sitio libre al alcance de las dos."""
        cands = [o for o in lista.objetos if o.clase in pendientes and o.confianza >= self.cfg["conf_min"]
                 and intentos[o.clase] < self.cfg["intentos_max"] and self.en_zona(o)]
        cands.sort(key=lambda o: o.centro[0])
        for o in cands:
            mano = self.est.lado_bandeja(self.cat[o.clase]["bandeja"])
            # directo solo desde el lado de su bandeja (con una franja central): cruzando la línea
            # media el agarre queda al límite del alcance y en el simulador se escapaba
            signo = 1.0 if mano == "izq" else -1.0
            if signo * o.centro[1] < -self.cfg["cruce_max"]:
                continue
            p = self._planear(o, lista, mano=mano)
            if p.ok:
                return p, "directo"
            self.log(f"  {o.clase}: sin plan directo con la mano {mano} ({p.motivo[:90]})")
        if not cambio_mano:
            for o in cands:
                intentos[o.clase] += 1
            return None, ""
        for o in cands:
            mano = self.est.lado_bandeja(self.cat[o.clase]["bandeja"])
            otra = "der" if mano == "izq" else "izq"
            q = {lado: self.brazo.consigna(lado) for lado in C.LADOS}
            punto = self.plan.punto_comun(o, lista.objetos, q, receptora=mano)
            if punto is None:
                self.log(f"  {o.clase}: no hay sitio común libre para el cambio de mano")
                intentos[o.clase] += 1
                continue
            p = self._planear(o, lista, mano=otra, destino=tuple(punto))
            if p.ok:
                return p, "entrega"
            self.log(f"  {o.clase}: sin plan hasta el sitio común ({p.motivo[:90]})")
            intentos[o.clase] += 1
        return None, ""

    def clasificar(self, t_max, cambio_mano=False):
        """Niveles 2 y 3: cada objeto a su bandeja. Se observa de nuevo antes de cada agarre (el
        plan no se calcula una sola vez: el instructor puede haber movido algo)."""
        pendientes = [k for k in self.cat]
        intentos = {k: 0 for k in self.cat}
        entregas = {k: 0 for k in self.cat}
        t0 = time.monotonic()
        lista, R0 = self.observar("inicio")
        vacias = 0
        while pendientes and time.monotonic() - t0 < t_max:
            self.estado_sup("PLANIFICAR")
            p, tipo = self.elegir(lista, pendientes, intentos, cambio_mano)
            if p is None:
                vacias += 1
                if vacias >= 2:
                    self.log("no queda ningún objeto pendiente que se pueda coger")
                    break
                lista, R0 = self.observar()
                continue
            vacias = 0
            clase = p.objeto.clase
            intentos[clase] += 1
            t_obj = time.monotonic()
            ocupadas = self.ocupadas(lista)
            destino = p.bandeja if tipo == "directo" else f"mesa ({p.bandeja[0]:.2f}, {p.bandeja[1]:+.2f})"
            self.log(f"{clase}: mano {p.mano} -> {destino} ({tipo}, intento {intentos[clase]})")
            self.terminados = [k for k in self.cat if k not in pendientes]
            cogido, motivo = self.ejecutar(p, R0)
            lista, R0 = self.observar(f"tras_{clase}")
            if tipo == "entrega":
                entregas[clase] += 1
                if cogido and entregas[clase] <= 2:
                    intentos[clase] -= 1                 # dejarlo en la mesa no gasta un intento...
                # ...salvo a partir de la tercera entrega: si al soltarlo vuelve al lado de la mano
                # que lo entrega (la pelota que rueda), se repetía sin fin hasta agotar el tiempo
                # ...pero un cambio de mano fallido sí (si no, se reintenta sin fin: nivel 3)
                self.reg.objeto(clase=clase, mano=p.mano, entrega=True, cogido=cogido, s=time.monotonic() - t_obj,
                                motivo=motivo)
                self.log(f"  {clase}: {'dejado para el cambio de mano' if cogido else 'cambio de mano fallido'}")
                continue
            colocado = cogido and self.en_bandeja(lista, clase, ocupadas)
            self.reg.objeto(clase=clase, mano=p.mano, bandeja=p.bandeja, intento=intentos[clase], cogido=cogido,
                            colocado=colocado, s=time.monotonic() - t_obj, motivo=motivo,
                            giro_max_deg=math.degrees(self.vig.giro_obs))
            self.log(f"  {clase}: {'COLOCADO' if colocado else 'no colocado'} "
                     f"({time.monotonic() - t_obj:.0f} s){' - ' + motivo if motivo else ''}")
            if colocado:
                pendientes.remove(clase)
                self.percep.colocados.add(clase)
        return pendientes

    def nivel4(self, clase, t_max, giro=math.pi / 2, pasos_max=4):
        """Colocar con orientación (sec. 4): `clase` sobre la cruz del centro del tablero, girado
        `giro` respecto de como está (centro <= 1.5 cm, orientación <= 15 grados). La muñeca no da
        para 90 grados desde cualquier sitio: se gira por pasos (todo lo que permita la mano en
        cada uno), observando de nuevo tras cada paso."""
        cruz = np.array([self.est.tablero.cx, self.est.tablero.cy])
        t0 = time.monotonic()
        lista, R0 = self.observar("inicio")
        # la cruz, con la cámara (antes de tapar nada): objetivo y objeto medidos con la misma cámara
        from .dianas import detectar_cruz
        c3 = detectar_cruz(self.ultimo_fotograma, self.percep.T, lista.plano_mesa, cruz)
        if c3 is not None:
            self.log(f"cruz vista en ({c3[0]:.3f}, {c3[1]:+.3f}); la de la config a "
                     f"{1000 * np.linalg.norm(c3[:2] - cruz):.1f} mm")
            self.reg.evento("cruz", vista=c3, config=cruz)
            cruz = c3[:2].copy()
        else:
            self.log("no veo la cruz: uso la de la config")
        o = self._buscar(lista, clase)
        if o is None:
            self.log(f"no veo el {clase}")
            return None
        sim = o.simetria if o.simetria > 0 else 2 * math.pi
        objetivo = o.guinada + giro
        self.reg.evento("nivel4", objeto=clase, guinada_inicial_deg=math.degrees(o.guinada),
                        objetivo_deg=math.degrees(objetivo), cruz=cruz)
        self.log(f"{clase}: guiñada {math.degrees(o.guinada):.0f} -> objetivo {math.degrees(objetivo):.0f} "
                 f"(módulo {math.degrees(sim):.0f}), cruz en ({cruz[0]:.3f}, {cruz[1]:+.3f})")
        for paso in range(pasos_max):
            if time.monotonic() - t0 > t_max:
                break
            falta = math.remainder(objetivo - o.guinada, sim)
            err_pos = float(np.linalg.norm(o.centro[:2] - cruz))
            if paso > 0 and abs(falta) <= math.radians(self.cfg["n4_tol_giro_deg"]) and \
                    err_pos <= self.cfg["n4_tol_pos"]:
                break
            self.estado_sup("PLANIFICAR", clase)
            p = None
            signo = 1.0 if falta >= 0 else -1.0
            giros = [falta] + ([-falta] if abs(abs(falta) - sim / 2) < 1e-3 else []) + \
                [signo * math.radians(a) for a in (60, 45, 30) if math.radians(a) < abs(falta)]
            if abs(falta) <= math.radians(self.cfg["n4_tol_giro_deg"]):
                giros = [falta]                 # solo falta centrarlo
            for g in giros:
                for mano in sorted(C.LADOS, key=lambda ld: abs(o.centro[1] - (0.1 if ld == "izq" else -0.1))):
                    q = {ld: self.brazo.consigna(ld) for ld in C.LADOS}
                    otros = [x for x in lista.objetos if x is not o]
                    p = self.plan.planificar(o, q, bandeja=tuple(cruz), mano=mano, otros=otros, plano=lista.plano_mesa,
                                             giro=g)
                    self.reg.evento("plan", objeto=clase, ok=p.ok, motivo=p.motivo, mano=mano,
                                    giro_deg=math.degrees(g))
                    if p.ok:
                        break
                if p is not None and p.ok:
                    break
            if p is None or not p.ok:
                self.log(f"  {clase}: sin plan para girarlo ({p.motivo[:90] if p else ''})")
                break
            self.log(f"  paso {paso + 1}: mano {p.mano}, giro {math.degrees(g):+.0f} grados")
            cogido, motivo = self.ejecutar(p, R0)
            lista, R0 = self.observar(f"paso{paso + 1}_{clase}")
            o2 = self._buscar(lista, clase)
            if not cogido or o2 is None:
                self.log(f"  {clase}: {'no lo cogió' if not cogido else 'no lo veo'} {motivo}")
                if o2 is None:
                    break
            e_pos = float(np.linalg.norm(o2.centro[:2] - cruz))
            e_gui = abs(math.remainder(objetivo - o2.guinada, sim))
            self.log(f"  tras el paso {paso + 1}: {1000 * e_pos:.1f} mm de la cruz, {math.degrees(e_gui):.1f} grados "
                     f"del giro pedido")
            self.reg.evento("nivel4_paso", paso=paso + 1, error_mm=1000 * e_pos, error_deg=math.degrees(e_gui),
                            giro_pedido_deg=math.degrees(g))
            o = o2
        err_pos = float(np.linalg.norm(o.centro[:2] - cruz))
        err_gui = abs(math.remainder(objetivo - o.guinada, sim))
        ok = err_pos <= 0.015 and err_gui <= math.radians(15)
        self.reg.objeto(clase=clase, nivel4=True, error_cruz_mm=1000 * err_pos,
                        error_guinada_deg=math.degrees(err_gui), ok=ok, s=time.monotonic() - t0)
        self.log(f"{clase}: sobre la cruz a {1000 * err_pos:.1f} mm y {math.degrees(err_gui):.1f} grados del giro "
                 f"pedido -> {'OK' if ok else 'NO'} (percepción)")
        return ok

    def nivel1(self, t_max, preforma=(0, 0, 0, 900, 1000, 1000), f_contacto=60.0):
        """Tocar dianas (sec. 4): la yema del índice en el centro de cada una (<= 2 cm), contacto
        detectado por la propia mano (fuerza del índice; <= 300 gf), retirarse y la siguiente."""
        from .dianas import detectar
        t0 = time.monotonic()
        lista, R0 = self.observar("inicio")
        dianas = detectar(self.ultimo_fotograma, self.percep.T, lista.plano_mesa, self.est.zona)
        self.reg.evento("dianas", centros=[d[0] for d in dianas], area_px=[d[1] for d in dianas],
                        contraste=[d[2] for d in dianas])
        self.log(f"dianas: {len(dianas)} -> " + ", ".join(f"({d[0][0]:.3f}, {d[0][1]:+.3f})" for d in dianas))
        tocadas = 0
        for k, (p, _, _) in enumerate(sorted(dianas, key=lambda d: d[0][0])):
            if time.monotonic() - t0 > t_max:
                break
            lado = "izq" if p[1] >= 0 else "der"
            otro = "der" if lado == "izq" else "izq"
            self.estado_sup("PLANIFICAR", f"diana{k}")
            plan, motivo = self.plan.tocar(lado, p, self.brazo.consigna(lado), self.brazo.consigna(otro),
                                           list(preforma))
            if plan is None:
                lado, otro = otro, lado
                plan, motivo = self.plan.tocar(lado, p, self.brazo.consigna(lado), self.brazo.consigna(otro),
                                               list(preforma))
            if plan is None:
                self.log(f"  diana {k}: sin plan ({motivo[:90]})")
                self.reg.objeto(diana=k, centro=p, tocada=False, motivo=motivo)
                continue
            if not self.simulacro:
                self.manos[lado].poner_force_set([300] * 6)
                self.manos[lado].mover(list(preforma))
            ir = lambda r: self.brazo.seguir_ruta(lado, r, comprobar=self.comprobar)  # noqa: E731
            self.estado_sup("APROXIMAR", f"diana{k}")
            self.vig.mirar_seguimiento = True
            ir(plan["aproximar"].q_ruta)
            self.brazo.asentar(lado, comprobar=self.comprobar)
            self.estado_sup("DESCENDER", f"diana{k}")
            ir(plan["descender"].q_ruta)
            # sondeo: recta lenta hacia abajo; se para en cuanto el índice nota la mesa
            self.estado_sup("SONDEAR", f"diana{k}")
            f0 = 0.0 if self.simulacro else float(self.manos[lado].fuerzas()[3])
            contacto = {}

            def tocar_comprobar():
                self.comprobar()
                if self.simulacro:
                    return
                f = float(self.manos[lado].fuerzas()[3])
                if f - f0 > f_contacto:
                    contacto.update(f=f, q=self.estado.brazo(lado))
                    raise _Contacto()
            try:
                self.brazo.seguir_ruta(lado, plan["sondear"].q_ruta, comprobar=tocar_comprobar)
            except _Contacto:
                with self.brazo.cerrojo:             # congelar donde está (la medida, no la consigna)
                    for j, v in zip(INDICES[lado], contacto["q"]):
                        self.brazo.obj[j] = float(v)
            yema = None
            if contacto:
                T = C.fk(lado, contacto["q"])
                yema = T[:3, :3] @ C.PUNTOS_MANO["indice_toque"][lado] + T[:3, 3]
                tocadas += 1
            err = float(np.linalg.norm(yema[:2] - p[:2])) if yema is not None else float("nan")
            self.reg.objeto(diana=k, centro=p, tocada=bool(contacto), fuerza_gf=contacto.get("f"), yema=yema,
                            error_mm=1000 * err, mano=lado)
            if contacto:
                self.log(f"  diana {k}: contacto a {contacto['f']:.0f} gf, yema a {1000 * err:.1f} mm del centro")
            else:
                self.log(f"  diana {k}: SIN contacto")
            self.estado_sup("RETIRAR", f"diana{k}")
            ir(plan["retirar"].q_ruta)
        self.brazo.mover(self.q_obs, comprobar=self.comprobar)
        for lado in C.LADOS:
            self._mano(lado, "reposo")
        return tocadas, len(dianas)

    def _buscar(self, lista, clase):
        os_ = [o for o in lista.objetos if o.clase == clase]
        return max(os_, key=lambda o: o.confianza) if os_ else None

    # ------------------------------------------------------------ entrada, salida
    def entrada(self):
        self.estado_sup("ENTRADA")
        self.brazo.rampa_entrada(comprobar=self.comprobar)

    def solo_observacion(self):
        """Hito 3: un único movimiento real a la pose de observación y vuelta, sin objeto."""
        self.vig.mirar_seguimiento = True
        self.estado_sup("OBSERVAR")
        self.brazo.mover(self.q_obs, comprobar=self.comprobar)
        self.esperar(2.0)

    def salida(self):
        """Normal: a la postura del principio y peso a 0."""
        self.estado_sup("SALIDA")
        self.vig.mirar_seguimiento = not self.simulacro
        self.brazo.mover(self.q_ini, comprobar=self.comprobar)
        self.vig.mirar_seguimiento = False
        self.brazo.rampa_salida()

    def salida_segura(self, motivo):
        """Congelar, abrir la mano, subirla a la altura de seguridad, peso a 0 (sec. 10)."""
        self.estado_sup("SALIDA_SEGURA")
        self.reg.evento("salida_segura", motivo=str(motivo))
        self.log(f"SALIDA SEGURA: {motivo}")
        self.vig.mirar_seguimiento = False
        for lado in C.LADOS:                     # abrir (a la preforma si llevaba un agarre)
            try:
                ag = self.agarre_actual.get(lado)
                if ag is not None:
                    self.manos[lado].preforma(ag.preforma)
                else:
                    self.manos[lado].abrir()
            except (ErrorMano, OSError, Exception) as e:   # noqa: B014
                self.log(f"  mano {lado}: {e}")
        time.sleep(1.0)
        subir = {}
        for lado in C.LADOS:                     # subir las manos que estén bajas
            q = self.brazo.consigna(lado)
            falta = self.plan.z_mesa + self.cfg["altura_seguridad"] - self.plan.mano_baja(lado, q)
            if falta <= 0:
                continue
            T = C.fk(lado, q)
            q2, ep, eo, ok = C.ik(lado, T[:3, 3] + np.array([0, 0, falta + 0.01]), T[:3, :3], q0=q,
                                  lim=self.plan.lim[lado], nombre="muneca")
            if ok:
                subir[lado] = q2
        if subir:
            try:
                self.estado_sup("SUBIR")
                self.brazo.mover(subir)          # sin vigilante: es la salida
            except Exception as e:
                self.log(f"  no se pudo subir: {e}")
        self.brazo.rampa_salida()
