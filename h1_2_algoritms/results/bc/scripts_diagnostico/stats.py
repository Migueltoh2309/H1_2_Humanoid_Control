import sys, os, collections, numpy as np
sys.argv=['x']; PK=os.path.expanduser('~/Github/H1_2_Humanoid_Control/h1_2_algoritms'); sys.path.insert(0,PK+'/demos')
import bc_faja as B
pol=os.environ.get('POL','fifo'); seeds=[int(x) for x in os.environ['SEEDS'].split(',')]
res=B._pool([(pol,s,False,'','sphere',True) for s in seeds],int(os.environ.get('W','8')))
C=collections.Counter(); E=collections.defaultdict(list)
for r in res:
    print(r['seed'], r['depositadas'], 'intentos', r['agarrar'], 'fallo', r['fallo'], 'perdida', r['perdida'], 't', r['t_fin'])
    for s,log in r['_eventos'].items():
        for i,e in enumerate(log):
            if e[1]=='cierra':
                nxt = next((x[1] for x in log[i+1:] if x[1] in ('verificado','fallo')), '?')
                E[(s,nxt)].append(e[4]*1e3 if len(e)>4 and e[4] is not None else np.nan)
            if e[1] in ('fallo','perdida'): C[(s,e[1],e[3][:30])]+=1
print('depositadas', sum(r['depositadas'] for r in res), '/', sum(r['n_frutas'] for r in res))
for k,v in sorted(C.items()): print(k,v)
for k,v in sorted(E.items()): print('err_close', k, len(v), 'mediana %.1f mm  p90 %.1f' % (np.nanmedian(v), np.nanpercentile(v,90)))
