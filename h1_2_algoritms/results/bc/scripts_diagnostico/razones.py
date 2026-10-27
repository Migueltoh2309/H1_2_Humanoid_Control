"""Perfil + razones de fallo/aborto (de los logs de eventos)."""
import sys, os, json, collections, re, numpy as np
sys.argv=['x']; PK=os.path.expanduser('~/Github/H1_2_Humanoid_Control/h1_2_algoritms'); sys.path.insert(0,PK+'/demos')
import bc_faja as B
from h1_2_algoritms.aprendizaje.habilidades import TIMING
P=json.loads(os.environ['PERFILES']); seeds=[int(x) for x in os.environ['SEEDS'].split(',')]
B.ESCENARIO=os.environ.get('ESC','exigente'); T0=dict(TIMING)
for name,kw in P.items():
    B.PERFILES[name]=kw; B.PERFIL=name
    TIMING.clear(); TIMING.update(T0); TIMING.update(kw.get('timing',{}))
    res=B._pool([(os.environ.get('POL','fifo'),s,False,'','sphere',True) for s in seeds],8)
    why=collections.Counter()
    for r in res:
        for log in r.get('_eventos',{}).values():
            for e in log:
                if e[1] in ('fallo','perdida'): why[(e[1], re.sub(r'[\d.]+','#',str(e[3]))[:50])]+=1
    S=lambda k: sum(int(r.get(k,0)) for r in res)
    print(f"{name:8s} dep {S('depositadas')}/{S('n_frutas')} intentos {S('agarrar')} verif {S('verificado')} caidas {S('caidas')} perdidas_faja {S('perdidas_faja')}", dict(why), flush=True)
