"""Compara perfiles de tiempos de las habilidades con FIFO (no usa el modelo de tiempos)."""
import sys, os, json, numpy as np
sys.argv=['x']; PK=os.path.expanduser('~/Github/H1_2_Humanoid_Control/h1_2_algoritms'); sys.path.insert(0,PK+'/demos')
import bc_faja as B
P=json.loads(os.environ['PERFILES'])
seeds=[int(x) for x in os.environ['SEEDS'].split(',')]
B.ESCENARIO=os.environ.get('ESC','exigente')
from h1_2_algoritms.aprendizaje.habilidades import TIMING
T0=dict(TIMING)
for name,kw in P.items():
    B.PERFILES[name]=kw; B.PERFIL=name
    TIMING.clear(); TIMING.update(T0); TIMING.update(kw.get('timing',{}))
    res=B._pool([(os.environ.get('POL','fifo'),s,False,'','sphere',True) for s in seeds],8)
    dur={k:[] for k in ('alcance','cierra->verif','depositar','volver')}
    pairs=[('agarrar','cierra','alcance'),('cierra','verificado','cierra->verif'),('verificado','suelta','depositar'),('suelta','libre','volver')]
    for r in res:
        for log in r.get('_eventos',{}).values():
            for a,b in zip(log,log[1:]):
                for e0,e1,k in pairs:
                    if a[1]==e0 and b[1]==e1 and a[2]==b[2]: dur[k].append(b[0]-a[0])
    S=lambda k: sum(int(r.get(k,0)) for r in res)
    ok=[r for r in res if 'error' not in r]
    print(f"{name:10s} depositadas {S('depositadas')}/{S('n_frutas')}  intentos {S('agarrar')} cierra {S('cierra')} verif {S('verificado')} fallos {S('fallo')} abortos {S('perdida')} caidas {S('caidas')} errores {len(res)-len(ok)}  |  " +
          "  ".join(f"{k} {np.median(v):.2f}" for k,v in dur.items() if v) + f"  ciclo {sum(np.median(v) for v in dur.values() if v):.2f}", flush=True)
