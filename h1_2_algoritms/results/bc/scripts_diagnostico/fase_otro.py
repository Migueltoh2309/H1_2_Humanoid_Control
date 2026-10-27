"""Fallos de agarre según lo que hace el OTRO brazo al cerrar."""
import sys, os, json, collections, numpy as np
sys.argv=['x']; PK=os.path.expanduser('~/Github/H1_2_Humanoid_Control/h1_2_algoritms'); sys.path.insert(0,PK+'/demos')
import bc_faja as B
from h1_2_algoritms.aprendizaje.habilidades import TIMING
seeds=[int(x) for x in os.environ['SEEDS'].split(',')]
B.ESCENARIO='exigente'; B.PERFIL=os.environ.get('PERF','tactil'); TIMING.update(B.PERFILES[B.PERFIL].get('timing',{}))
res=B._pool([('experto',s,False,'','sphere',True) for s in seeds],8)
NEXT={'agarrar':'alcance','cierra':'agarre','verificado':'depositar','suelta':'volver','libre':'libre','fallo':'volver','perdida':'volver'}
def fase(log,t):
    f='libre'
    for e in log:
        if e[0]>t: break
        f=NEXT.get(e[1],f)
    return f
C=collections.Counter()
for r in res:
    L=r['_eventos']
    for s,o in (('left','right'),('right','left')):
        log=L[s]
        for a,b in zip(log,log[1:]):
            if a[1]=='cierra':
                C[(fase(L[o],a[0]+0.3), b[1])]+=1
for k in sorted(C): print(k,C[k])
