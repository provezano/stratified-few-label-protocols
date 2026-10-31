"""Round-15 check (EXPLORATORY; pure computation, no data): V_S of Prop. 4(iii) against the label-shift
efficiency bounds of Lee, Ma & Zhao (JASA 2025, Prop. 1 and Cor. 1) and Chow (Yale PhD thesis 2022,
arXiv:2211.04274, Thms. 3.7 and 3.9).

Binary Y, finite feature space with Z values, posterior eta(x), pool law p(x); target prior theta = q1.
(A) LMZ bound var{sqrt(pi) phi_eff} at labelled share pi, source prior s (here 1/2);
(B) inverse multinomial Fisher information, i.i.d. source with a free source prior s;
(C) the same with fixed class counts (stratified source; s not a parameter).
Checks: A = B = C at finite pi; A -> V_S as pi -> 0; Chow (1 - tau) V_eff(tau) = A at pi = 1 - tau and
-> V_S as tau -> 1; Chow closed form at tau -> 1: V_S = 2(pi1^2 + pi2^2)[I(g*)^-1 - g*(1 - g*)],
g* = pi1^2/(pi1^2 + pi2^2), I(g) = sum (p1 - p2)^2/(g p1 + (1 - g) p2).
Reproduces the scratchpad checks chk/lmz2.py and chk/chow2.py (same seeds and cases).
Writes out/rv_r15.tex.
Run from simaudit/: nice -n 15 .venv/bin/python paper/p1/exploratory/review_reanalysis/r15_lmz_chow.py
"""

from __future__ import annotations

import numpy as np

from rv_common import Macros

def lmz(px,eta,s,pi):
    p1=px@eta; p2=1-p1
    g1=px*eta/p1; g0=px*(1-eta)/p2            # class laws (shared)
    rho1,rho0=p1/s,p2/(1-s)
    c=pi/(1-pi)
    Erho=rho1*eta+rho0*(1-eta)               # E_q{rho|x}
    f1=eta/(Erho+c); f0=(1-eta)/(Erho+c)      # b = a1 f1 + a0 f0 = E_q{a|x}/(E_q{rho|x}+c)
    A=np.array([[g1@f1,g1@f0],[g0@f1,g0@f0]]) # E[b|Y=y]=y
    a1,a0=np.linalg.solve(A,[1,0]); b=a1*f1+a0*f0
    Ep=s*rho1**2*(g1@(1-b)**2)+(1-s)*rho0**2*(g0@b**2)
    Eq=px@(b-p1)**2
    return Ep+c*Eq
def fisher(px,eta,s,pi,strat):
    p1=px@eta; g1=px*eta/p1; g0=px*(1-eta)/(1-p1); Z=len(px)
    lam=(1-pi)/pi
    # params: g1[0:Z-1], g0[0:Z-1], q1, (s if iid)
    k=2*(Z-1)+1+(0 if strat else 1)
    def dg(i):  # derivative of a Z-vector w.r.t. its i-th free coord
        d=np.zeros(Z); d[i]=1; d[-1]=-1; return d
    J=np.zeros((k,k))
    # target cells
    P=p1*g1+(1-p1)*g0; D=np.zeros((Z,k))
    for i in range(Z-1): D[:,i]=p1*dg(i); D[:,Z-1+i]=(1-p1)*dg(i)
    D[:,2*(Z-1)]=g1-g0
    J+=lam*D.T@(D/P[:,None])
    if strat:  # half a row from each class, likelihood g_y(x)
        for y,g in ((1,g1),(0,g0)):
            D=np.zeros((Z,k))
            for i in range(Z-1): D[:,(0 if y else Z-1)+i]=dg(i)
            J+=0.5*D.T@(D/g[:,None])
    else:
        P=np.concatenate([s*g1,(1-s)*g0]); D=np.zeros((2*Z,k))
        for i in range(Z-1): D[:Z,i]=s*dg(i); D[Z:,Z-1+i]=(1-s)*dg(i)
        D[:Z,-1]=g1; D[Z:,-1]=-g0
        J+=D.T@(D/P[:,None])
    return np.linalg.inv(J)[2*(Z-1),2*(Z-1)]


def vs_of(px, eta):
    p1 = px @ eta
    p2 = 1 - p1
    w, r = eta * (1 - eta), (1 - eta) / p1 + eta / p2
    G, K = px @ (w / r), 1 / p1**2 + 1 / p2**2
    return 2 * G / (1 - K * G)


def chow_veff(px, eta, tau):
    p1 = px @ eta
    p2 = 1 - p1
    f1, f0 = px * eta / p1, px * (1 - eta) / p2
    ptr, ps = np.array([.5, .5]), np.array([p2, p1])
    S = np.sum(ps**2 / ptr)
    g = (p1 / tau + p1**2 / ptr[1] / (1 - tau)) / (1 / tau + S / (1 - tau))
    I = np.sum((f1 - f0) ** 2 / (g * f1 + (1 - g) * f0))
    return (1 / tau + S / (1 - tau)) * (1 / I - g * (1 - g)) + p1 * p2 / tau


def chow_closed(px, eta):
    p1 = px @ eta
    p2 = 1 - p1
    f1, f0 = px * eta / p1, px * (1 - eta) / p2
    g = p1**2 / (p1**2 + p2**2)
    I = np.sum((f1 - f0) ** 2 / (g * f1 + (1 - g) * f0))
    return 2 * (p1**2 + p2**2) * (1 / I - g * (1 - g))


def main() -> None:
    rng = np.random.default_rng(7)  # as chk/lmz2.py
    abc, lim, zs = 0.0, 0.0, []
    for _ in range(8):
        Z = int(rng.integers(3, 8))
        px = rng.dirichlet(np.ones(Z))
        eta = rng.uniform(0.05, 0.95, Z)
        zs.append(Z)
        for pi in (0.5, 0.1, 1e-3, 1e-7):
            A, B, C = lmz(px, eta, 0.5, pi), fisher(px, eta, 0.5, pi, False), fisher(px, eta, 0.5, pi, True)
            abc = max(abc, abs(A / B - 1), abs(C / B - 1))
        lim = max(lim, abs(lmz(px, eta, 0.5, 1e-9) / vs_of(px, eta) - 1))
    rng = np.random.default_rng(11)  # as chk/chow2.py
    ch, chl, cc = 0.0, 0.0, 0.0
    for _ in range(6):
        Z = int(rng.integers(3, 8))
        px = rng.dirichlet(np.ones(Z))
        eta = rng.uniform(0.05, 0.95, Z)
        zs.append(Z)
        for tau in (0.3, 0.9, 0.999):
            ch = max(ch, abs((1 - tau) * chow_veff(px, eta, tau) / lmz(px, eta, 0.5, 1 - tau) - 1))
        chl = max(chl, abs((1e-9) * chow_veff(px, eta, 1 - 1e-9) / vs_of(px, eta) - 1))
        cc = max(cc, abs(chow_closed(px, eta) / vs_of(px, eta) - 1))
    print(f"LMZ=iid=strat max rel {abc:.1e}; LMZ->V_S {lim:.1e}; Chow=LMZ {ch:.1e}; Chow->V_S {chl:.1e}; closed {cc:.1e}")

    def digits(x: float) -> str:
        return str(int(np.floor(-np.log10(max(x, 1e-16)))))

    m = Macros("r15")
    src = "r15_lmz_chow.py (pure computation"
    m.put("RfifteenCasesLMZ", "8", src + "): random saturated binary cases for LMZ")
    m.put("RfifteenCasesChow", "6", src + "): random cases for Chow")
    m.put("RfifteenZMin", str(min(zs)), src + "): smallest number of feature values")
    m.put("RfifteenZMax", str(max(zs)), src + "): largest number of feature values")
    m.put("RfifteenDigitsABC", digits(abc), src + "): digits of agreement LMZ bound = i.i.d. Fisher = stratified Fisher at labelled shares 0.5 to 1e-7")
    m.put("RfifteenDigitsLimit", digits(max(lim, chl)), src + "): digits of agreement of the LMZ and Chow bounds with V_S in the infinite-pool limit (share 1e-9)")
    m.put("RfifteenDigitsChow", digits(ch), src + "): digits of agreement Chow (1-tau)V_eff = LMZ at tau in {0.3, 0.9, 0.999}")
    m.put("RfifteenDigitsClosed", digits(cc), src + "): digits of agreement of Chow's closed form with V_S")
    m.write()


if __name__ == "__main__":
    main()
