import numpy as np
from scipy.optimize import minimize
rng=np.random.default_rng(0)
def theory(p,eta):
    psi=p@eta; pi2=1-psi
    w=eta*(1-eta); r=(1-eta)/psi+eta/pi2
    G=p@(w/r); K=1/psi**2+1/pi2**2
    return 2*G/(1-K*G), p@w, psi*pi2   # B*V_S, B*V_R, pi1pi2
def mle_strat(n1,n2,p,m):
    def nll(z):
        eta=1/(1+np.exp(-z)); psi=p@eta
        return -(n1@np.log(eta)+n2@np.log(1-eta)-m*np.log(psi)-m*np.log(1-psi))
    z0=np.zeros(len(p))
    res=minimize(nll,z0,method='L-BFGS-B',bounds=[(-12,12)]*len(p))
    eta=1/(1+np.exp(-res.x)); return p@eta
def est_rand(x,y,p,J):
    eta=np.array([y[x==j].mean() if (x==j).any() else y.mean() for j in range(J)])
    return p@eta
if __name__=="__main__":
  for p,eta in [(np.array([.25,.25,.25,.25]),np.array([.8,.6,.3,.1])),
               (np.array([.4,.3,.2,.1]),np.array([.05,.1,.5,.9])),
               (np.array([.45,.45,.1]),np.array([.02,.3,.95]))]:
     VS,VR,pp=theory(p,eta); psi=p@eta
     for m in [50,400]:
         B=2*m; es=[];er=[];ec=[]
         p1=p*eta/psi; p2=p*(1-eta)/(1-psi)
         for _ in range(1000):
             n1=rng.multinomial(m,p1); n2=rng.multinomial(m,p2)
             es.append(mle_strat(n1,n2,p,m))
             x=rng.choice(len(p),size=B,p=p); y=(rng.random(B)<eta[x]).astype(float)
             er.append(est_rand(x,y,p,len(p))); ec.append(y.mean())
         es,er,ec=map(np.array,(es,er,ec))
         print(f"pi1={psi:.3f} m={m}: B*MSE strat+pool={B*np.mean((es-psi)**2):.4f} (bound {VS:.4f}); "
               f"rand+pool={B*np.mean((er-psi)**2):.4f} (bound {VR:.4f}); rand count={B*np.mean((ec-psi)**2):.4f} ({pp:.4f})")
