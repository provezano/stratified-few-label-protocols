# Exploratory sensitivity: drop the 3 post-release tables published before Kumo-Tabular's public release commit (2026-09-28).
import sys, json
sys.path.insert(0, 'code')
import pandas as pd, numpy as np
from scipy.stats import spearmanr, wilcoxon
import family as F
t = pd.read_csv('results/fresh_pertable.csv')
t['gap_mf_mean'] = t[[f'gap_mf_fm{k}' for k in (1,2,3,5)]].mean(axis=1) if 'gap_mf_mean' not in t else t['gap_mf_mean']
t['pred_m2'] = F.predict_m2(t)
early = ['z22960380_citizenstatus', 'f34004118_hfref', 'z22997149_menohealth']
out = {}
for name, s in (('all', t), ('drop3', t[~t.key.isin(early)])):
    d = dict(n=len(s), rho_m2=spearmanr(s.pred_m2, s.gap_mf_mean)[0], rho_pre=spearmanr(s.pred_pre, s.gap_mf_mean)[0],
             resid_m2=float((s.gap_mf_mean - s.pred_m2).mean()), obs=float(s.gap_mf_mean.mean()))
    for k in (1,2,3,5):
        x = s[f'gap_mf_fm{k}']; d[f'gap_fm{k}'] = float(x.mean()); d[f'p_fm{k}'] = float(wilcoxon(x, alternative='greater').pvalue)
    out[name] = d
print(json.dumps(out, indent=1))
json.dump(out, open(sys.argv[1], 'w'), indent=1)
