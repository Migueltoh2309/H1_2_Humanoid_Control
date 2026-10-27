"""Variantes del experto (perfil espera) en semillas de desarrollo."""
import sys, os, json, numpy as np
sys.argv=['x']; PK=os.path.expanduser('~/Github/H1_2_Humanoid_Control/h1_2_algoritms'); sys.path.insert(0,PK+'/demos')
import bc_faja as B
from h1_2_algoritms.aprendizaje import alto_nivel as AN
from h1_2_algoritms.aprendizaje.habilidades import TIMING
B.ESCENARIO='exigente'; B.PERFIL='espera'; TIMING.update(B.PERFILES['espera']['timing'])
seeds=[int(x) for x in os.environ['SEEDS'].split(',')]
V=json.loads(os.environ['VARS'])
init0=AN.ExpertPolicy.__init__; W0=dict(AN.WINDOWS)
for name,kw in V.items():
    pol=kw.pop('pol','experto')
    AN.WINDOWS.clear(); AN.WINDOWS.update(W0); AN.WINDOWS.update({k:tuple(v) for k,v in kw.pop('windows',{}).items()})
    def init(self, _kw=dict(kw), **k): init0(self, **{**_kw, **k})
    AN.ExpertPolicy.__init__=init
    res=B._pool([(pol,s,False,'','sphere') for s in seeds],6)
    S=lambda k: sum(int(r.get(k,0)) for r in res)
    print(f"{name:10s} {pol:8s} dep {S('depositadas')}/{S('n_frutas')} intentos {S('agarrar')} fallos {S('fallo')} abortos {S('perdida')} perdidas_faja {S('perdidas_faja')}", [r['depositadas'] for r in res], flush=True)
