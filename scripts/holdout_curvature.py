#!/usr/bin/env python3
"""Holdout de codigo, analise POST HOC (nao registrada): o desvio de cada familia em
relacao a curva ajustada sem ela acompanha a posicao da familia no eixo de KL?
Tambem refaz os desvios restritos a KL < 0.25. Grava analysis/holdout_curvature.json.

    python3 scripts/holdout_curvature.py
"""
import json, math, os, numpy as np, torch
from scipy import stats
from glod.paths import OUT
from glod.pipelines.fidelity.holdout import family
R = os.environ.get("GLOD_HOLDOUT", "/local/user_beatrizalmeida/ews_results/holdout")
torch.set_grad_enabled(False)
pts, win = [], []
for d in sorted(os.listdir(R)):
    if "__fp32" not in d: continue
    base = f"{R}/{d}/{d}"; ref = torch.load(f"{base}/bf16.pt", weights_only=False)
    rows = []
    for f in sorted(os.listdir(base)):
        if f == "bf16.pt": continue
        c = torch.load(f"{base}/{f}", weights_only=False)
        rows.append((family(f[:-3]), c["kl"].double().clamp_min(0).mean().item(), (c["top1"] != ref["top1"]).double().mean().item()))
    for fam in {r[0] for r in rows}:
        tr = [r for r in rows if r[0] != fam]; te = [r for r in rows if r[0] == fam]
        a, b = np.polyfit(np.log([r[1] for r in tr]), np.log([r[2] for r in tr]), 1)
        dev = math.exp(np.mean([math.log(r[2]) - (b + a * math.log(r[1])) for r in te])) - 1
        pts.append((d, fam, np.mean([math.log(r[1]) for r in te]), dev))
    # mesma coisa restrita a KL < 0.25 (regime quadratico publicado)
    rw = [r for r in rows if 1e-3 < r[1] < 0.25]
    for fam in {r[0] for r in rw}:
        tr = [r for r in rw if r[0] != fam]; te = [r for r in rw if r[0] == fam]
        if len(tr) < 4: continue
        a, b = np.polyfit(np.log([r[1] for r in tr]), np.log([r[2] for r in tr]), 1)
        win.append((d, fam, abs(math.exp(np.mean([math.log(r[2]) - (b + a * math.log(r[1])) for r in te])) - 1) * 100))
x = [p[2] for p in pts]; y = [p[3] for p in pts]
rho = stats.spearmanr(x, y)
# por modelo
per = [stats.spearmanr([p[2] for p in pts if p[0] == d], [p[3] for p in pts if p[0] == d]).statistic for d in sorted({p[0] for p in pts})]
out = {"spearman_all": rho.statistic, "p": rho.pvalue, "n": len(pts), "per_model": per,
       "window_median_absdev_pct": float(np.median([w[2] for w in win])), "window_max_absdev_pct": float(max(w[2] for w in win)),
       "window_n": len(win), "full_median_absdev_pct": float(np.median([abs(p[3]) * 100 for p in pts]))}
print(json.dumps(out, indent=1))
json.dump({"summary": out, "family_points": pts, "window": win}, open(OUT / "analysis" / "holdout_curvature.json", "w"), indent=1)
