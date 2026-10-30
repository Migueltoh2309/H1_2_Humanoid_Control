"""Qué explica la ganancia de 'suave': física (kd, interp, asentado) o modelo de tiempos."""
import sys, os, numpy as np
VAR = os.environ["VAR"]; sys.argv = ['x']
sys.path.insert(0, os.path.expanduser('~/Github/H1_2_Humanoid_Control/h1_2_algoritms/demos'))
import bc_faja as B
from h1_2_algoritms.aprendizaje import habilidades as H
P = B.PERFILES
tE, tS = P["espera"]["timing"], P["suave"]["timing"]
V = {"espera": dict(P["espera"]), "suave": dict(P["suave"]),
     "espera_tS": {**P["espera"], "timing": tS},                 # física vieja, tiempos nuevos
     "suave_tE": {**P["suave"], "timing": tE},                   # física nueva, tiempos viejos
     "solo_kd": {**P["espera"], "kd_escala": 4.0, "timing": tS},
     "solo_asentado": {**P["espera"], "home_vel_tol": 0.5, "timing": tS}}
P["_v"] = V[VAR]; B.PERFIL = "_v"; B.ESCENARIO = "exigente"
H.TIMING.update(P["_v"]["timing"])
res = B._pool([("experto", s, False, "", "sphere", True) for s in range(5000, 5024)], 6)
dep = [r["depositadas"] for r in res]; ag = sum(r["agarrar"] for r in res); pe = sum(r["perdida"] for r in res)
print(f"{VAR:14s} depositadas {sum(dep)}/240  intentos {ag}  abortos {pe}  t_fin {np.mean([r['t_fin'] for r in res]):.1f}  {dep}")
