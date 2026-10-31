import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
# Gaussian classes p1=N(0,1), p2=N(d,1); known pool law p.
def quantities(pi1,d):
    pi2=1-pi1
    p=lambda x: pi1*norm.pdf(x)+pi2*norm.pdf(x,d)
    eta=lambda x: pi1*norm.pdf(x)/p(x)
    w=lambda x: eta(x)*(1-eta(x))
    r=lambda x: (1-eta(x))/pi1+eta(x)/pi2
    L,U=-12,d+12
    Om=quad(lambda x:p(x)*w(x),L,U,limit=200)[0]
    G=quad(lambda x:p(x)*w(x)/r(x),L,U,limit=200)[0]
    K=1/pi1**2+1/pi2**2
    VS=2*G/(1-K*G)   # B * asymptotic variance bound, stratified + pool
    VR=Om            # B * efficient variance, random + pool
    return Om,G,K*G,VS,VR,pi1*pi2
if __name__=="__main__":
 print("pi1  d   Omega/(pi1pi2)  B*V_S   B*V_R   pi1pi2  V_S/V_R  V_S/(pi1pi2)")
 for pi1 in [0.5,0.3,0.1]:
     for d in [0.25,0.5,1,1.5,2,3,4]:
         Om,G,KG,VS,VR,pp=quantities(pi1,d)
         print(f"{pi1:.1f} {d:4.2f} {Om/pp:8.3f} {VS:9.4f} {VR:8.4f} {pp:6.3f} {VS/VR:7.2f} {VS/pp:7.2f}")
