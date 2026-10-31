import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
rng=np.random.default_rng(20261029)
def gauss_grid(pi1,d,n=20001):
    x=np.linspace(-14,d+14,n); dx=x[1]-x[0]
    p1=norm.pdf(x); p2=norm.pdf(x,d); p=pi1*p1+(1-pi1)*p2
    return x,dx,p1,p2,p
def VS_formula(pi1,d):
    pi2=1-pi1; x,dx,p1,p2,p=gauss_grid(pi1,d)
    eta=pi1*p1/p; w=eta*(1-eta); r=(1-eta)/pi1+eta/pi2
    Om=np.sum(p*w)*dx; G=np.sum(p*w/r)*dx; K=1/pi1**2+1/pi2**2
    return 2*G/(1-K*G),Om,G,K
# --- check A: formula vs brute-force Fisher info inverse in a finite-cell nonparametric model
def VS_fisher_cells(pi1,d,J):
    pi2=1-pi1
    edges=np.r_[-np.inf,np.linspace(-1.5,d+1.5,J-1),np.inf]
    q1=np.diff(norm.cdf(edges)); q2=np.diff(norm.cdf(edges,d)); p=pi1*q1+pi2*q2
    # theta=(pi1, q1_1..q1_{J-1}); q1_J=1-sum; q2=(p-pi1 q1)/(1-pi1)
    k=J
    D1=np.zeros((J,k)); D1[:J-1,1:]=np.eye(J-1); D1[J-1,1:]=-1
    # dq2/dpi1 = (-q1(1-pi1)+(p-pi1 q1))/(1-pi1)^2 = (p - q1)/pi2^2
    D2=np.zeros((J,k)); D2[:,0]=(p-q1)/pi2**2; D2[:,1:]=-pi1/pi2*D1[:,1:]
    I=D1.T@np.diag(1/q1)@D1 + D2.T@np.diag(1/q2)@D2   # per pair
    v=np.linalg.inv(I)[0,0]
    eta=pi1*q1/p; w=eta*(1-eta); r=(1-eta)/pi1+eta/pi2
    G=np.sum(p*w/r); K=1/pi1**2+1/pi2**2; Om=np.sum(p*w)
    return 2*v, 2*G/(1-K*G), Om, q1,q2,p
print("A: brute-force inverse Fisher (cells) vs closed form V_S")
for pi1,d,J in [(0.5,1,4),(0.3,2,6),(0.1,4,8),(0.1,1,30),(0.24,1.5,3)]:
    a,b,Om,*_=VS_fisher_cells(pi1,d,J); print(f" pi1={pi1} d={d} J={J}: invFisher*2={a:.6f} formula={b:.6f} Omega={Om:.5f}")
print("B: Gaussian closed form; consequences")
for pi1 in [0.5,0.3,0.1]:
  for d in [0.5,1,2,4]:
    VS,Om,G,K=VS_formula(pi1,d); pmin=min(pi1,1-pi1); pmax=1-pmin; pp=pi1*(1-pi1)
    cond=Om>pmax*pp
    print(f" pi1={pi1} d={d}: VS={VS:.4f} Om={Om:.4f} pi1pi2={pp:.3f} VS/Om={VS/Om:.3f} VS>=2pminOm:{VS>=2*pmin*Om-1e-12} "
          f"cond(b)={cond} VS>pp={VS>pp}" + (f" OmOver(1-4Om)={Om/(1-4*Om):.4f}" if pi1==0.5 else ""))
# crossover d where VS=pi1pi2 and VS=Omega
from scipy.optimize import brentq
for pi1 in [0.5,0.3,0.1]:
    pp=pi1*(1-pi1)
    d1=brentq(lambda d: VS_formula(pi1,d)[0]-pp,0.5,4)
    s=[VS_formula(pi1,d)[0]-VS_formula(pi1,d)[1] for d in np.linspace(0.5,6,23)]
    print(f" pi1={pi1}: VS=pi1pi2 at d={d1:.3f}; VS-Om sign over d in[0.5,6]: {''.join('+' if v>0 else '-' for v in s)}")
# --- check C: Monte Carlo continuous Gaussian, oracle-efficient moment estimator f=eta/(pi2 r), plus f=x
print("C: MC, continuous Gaussian classes, m per class, B=2m")
for pi1,d,m,R in [(0.5,1,4000,2000),(0.1,4,4000,2000),(0.3,2,4000,2000),(0.1,1,4000,2000)]:
    pi2=1-pi1; x,dx,p1,p2,p=gauss_grid(pi1,d); eta=pi1*p1/p; r=(1-eta)/pi1+eta/pi2
    fgrid=eta/(pi2*r); mu=np.sum(p*fgrid)*dx; mux=pi1*0+pi2*d
    f=lambda z: np.interp(z,x,fgrid)
    est=[];estx=[];estR=[];estRc=[]
    for _ in range(R):
        z1=rng.standard_normal(m); z2=rng.standard_normal(m)+d
        a1,a2=f(z1).mean(),f(z2).mean(); est.append((mu-a2)/(a1-a2))
        estx.append((mux-z2.mean())/(z1.mean()-z2.mean()))
        B=2*m; y=rng.random(B)<pi1; zz=np.where(y,rng.standard_normal(B),rng.standard_normal(B)+d)
        e=np.interp(zz,x,eta); estR.append(y.mean()-e.mean()+pi1); estRc.append(y.mean())
    B=2*m; VS,Om,G,K=VS_formula(pi1,d)
    mse=lambda e: B*np.mean((np.array(e)-pi1)**2)
    se=lambda e: B*np.std((np.array(e)-pi1)**2)/np.sqrt(R)
    print(f" pi1={pi1} d={d} m={m}: S+pool eff-moment B*MSE={mse(est):.4f}+-{se(est):.4f} (V_S={VS:.4f}); S+pool mean-moment {mse(estx):.4f}; "
          f"R+pool oracle {mse(estR):.4f} (Om={Om:.4f}); R count {mse(estRc):.4f} (pp={pi1*pi2:.4f})")
# --- check D: Le Cam pair of part (iv) - pool law preserved and TV for random design
pi1,t=0.2,0.25; pi2=1-pi1; D=pi1*t; eps=t/(1+t)
x,dx,p1,p2,p=gauss_grid(pi1,1.0)
pp=(pi1+D)*((1-eps)*p1+eps*p2)+(pi2-D)*p2
print("D: max|p'-p| =",np.abs(pp-p).max(), " TV joint one random row =",0.5*(np.sum(np.abs(pi1*p1-(pi1+D)*((1-eps)*p1+eps*p2)))+np.sum(np.abs(pi2*p2-(pi2-D)*p2)))*dx," Delta=",D)
