import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from scipy.integrate import quad
from bounds import quantities
def bayes(pi1,d):
    p1=lambda x: pi1*norm.pdf(x); p2=lambda x: (1-pi1)*norm.pdf(x,d)
    return quad(lambda x: min(p1(x),p2(x)),-12,d+12,limit=200)[0]
for pi1 in [0.5,0.3,0.1]:
    f=lambda d: quantities(pi1,d)[3]-quantities(pi1,d)[5]
    g=lambda d: quantities(pi1,d)[3]-quantities(pi1,d)[4]
    d1=brentq(f,0.3,4); 
    try: d2=brentq(g,1,8)
    except Exception: d2=float('nan')
    print(f"pi1={pi1}: S+pool worse than random label frequency for d<{d1:.2f} (Bayes error {bayes(pi1,d1):.3f}); worse than random+pool for d<{d2:.2f} (Bayes err {bayes(pi1,d2) if d2==d2 else float('nan'):.4f})")
