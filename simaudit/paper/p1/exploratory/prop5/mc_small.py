import numpy as np
from scipy.stats import norm
from mc import mle_strat, est_rand, theory
rng=np.random.default_rng(1)
edges=np.r_[-np.inf,np.linspace(-1.5,2.5,7),np.inf]
for pi1 in [0.5,0.3,0.1]:
  for d in [1.0,2.0]:
    c1=np.diff(norm.cdf(edges)); c2=np.diff(norm.cdf(edges-d))
    p=pi1*c1+(1-pi1)*c2; eta=pi1*c1/p
    VS,VR,pp=theory(p,eta)
    row=[]
    for m in [1,3,5,10,20]:
        B=2*m; es=[];er=[];ec=[]
        for _ in range(600):
            n1=rng.multinomial(m,c1); n2=rng.multinomial(m,c2)
            es.append(mle_strat(n1,n2,p,m))
            x=rng.choice(len(p),size=B,p=p); y=(rng.random(B)<eta[x]).astype(float)
            er.append(est_rand(x,y,p,len(p))); ec.append(y.mean())
        f=lambda e: np.sqrt(np.mean((np.array(e)-pi1)**2))
        row.append(f"m={m}: S+pool {f(es):.3f} / R+pool {f(er):.3f} / Rcount {f(ec):.3f}")
    print(f"pi1={pi1} d={d} (binned; B*V_S={VS:.3f}, B*V_R={VR:.3f}, pi1pi2={pp:.3f})"); print("   "+"\n   ".join(row))
