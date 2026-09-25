#!/usr/bin/env python3
"""Analise do estudo de fidelidade: lei sqrt(KL), teoria de kappa, D3, D4.

    python -m glod analyze law        # ajuste por modelo + razao obs/iso + KL truncado vs completo
    python -m glod analyze theory     # kappa previsto por rho_gap(0) e curvatura; versao anisotropica
    python -m glod analyze d3         # modelos distintos / checkpoints / base-vs-instruct
    python -m glod analyze d4         # saturacao do KL no modelo isotropico; aditividade por camada
Tudo le /local/user_beatrizalmeida/ews_results/fid e grava analysis/*.json.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch

from glod.paths import OUT as ROOT
from glod.paths import corpus_of
AN = ROOT / "analysis"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ carga
def refs() -> list[str]:
    return sorted(d.name for d in ROOT.iterdir() if (d / d.name / "bf16.pt").exists())


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", mmap=True, weights_only=False)


def family(cfg: str) -> str:
    for pre, fam in (("ofc@", "oficial"), ("gptq", "gptq"), ("awq", "awq"), ("sgpt", "sparsegpt"),
                     ("wanda", "wanda"), ("kv", "kv"), ("mag", "magnitude"), ("skip", "skip"),
                     ("lq", "camada"), ("raw", "outro modelo")):
        if cfg.startswith(pre):
            return fam
    if re.fullmatch(r"u\d+", cfg):
        return "rtn"
    if re.fullmatch(r"(s\d+)?g\d+", cfg):
        return "gauss"
    return "?"


def points(ref_slug: str, model_slug: str | None = None) -> list[dict]:
    ref = load(ROOT / ref_slug / ref_slug / "bf16.pt")
    sub = ref.get("subset")
    out = []
    dirs = [ROOT / ref_slug / (model_slug or ref_slug)]
    for d in dirs:
        for f in sorted(d.glob("*.pt")):
            if f.stem == "bf16":
                continue
            c = load(f)
            flip = (c["top1"] != ref["top1"])
            row = {"ref": ref_slug, "model": d.name, "config": f.stem.replace("--", "/"), "family": family(f.stem),
                   "flip": flip.double().mean().item(), "kl": c["kl"].double().mean().item(),
                   "tv": c["tv"].double().mean().item()}
            if "kl_full" in c and sub is not None:
                row["kl_full_sub"] = c["kl_full"].double().nanmean().item()
                row["kl_top_sub"] = c["kl"][sub].double().mean().item()
                row["tv_full_sub"] = c["tv_full"].double().nanmean().item()
                row["flip_sub"] = flip[sub].double().mean().item()
            out.append(row)
    return out


# ------------------------------------------------------ modelo isotropico
class Iso:
    """Ruido N(0, s^2) em TODOS os logits do subconjunto (vocabulario completo)."""

    def __init__(self, ref: dict, n_tok: int = 2048, samples: int = 4, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        full = ref["full"]
        idx = torch.randperm(len(full), generator=g)[:n_tok]
        self.P = full[idx].float().to(DEV).log_softmax(-1)
        self.top = self.P.argmax(-1)
        self.samples = samples
        self.seed = seed

    @torch.no_grad()
    def at(self, s: float) -> tuple[float, float]:
        gen = torch.Generator(device=DEV).manual_seed(self.seed)
        kls, flips = [], []
        for chunk in torch.split(torch.arange(len(self.P), device=DEV), 256):
            lp = self.P[chunk]
            for _ in range(self.samples):
                q = (lp + s * torch.randn(lp.shape, generator=gen, device=DEV)).log_softmax(-1)
                kls.append((lp.exp() * (lp - q)).sum(-1))
                flips.append((q.argmax(-1) != self.top[chunk]).float())
        return torch.cat(kls).double().mean().item(), torch.cat(flips).double().mean().item()

    def flips_at_kl(self, kl: float) -> tuple[float, float]:
        """Interpola a curva (KL(s), flips(s)) pre-computada em escala log."""
        if not hasattr(self, "_grid"):
            ss = np.geomspace(0.01, 30, 40)
            vals = [self.at(float(x)) for x in ss]
            self._grid = (ss, np.log([v[0] for v in vals]), np.log([max(v[1], 1e-9) for v in vals]))
        ss, lk, lf = self._grid
        x = math.log(max(kl, 1e-9))
        return float(np.exp(np.interp(x, lk, lf))), float(np.exp(np.interp(x, lk, np.log(ss))))


def fit_loglog(rows, xkey="kl", ykey="flip", fam_exclude=("outro modelo",)):
    r = [x for x in rows if x["family"] not in fam_exclude and 1e-4 < x[xkey] and 0 < x[ykey] < 0.5]
    if len(r) < 3:
        return None
    x = np.log([p[xkey] for p in r]); y = np.log([p[ykey] for p in r])
    A = np.vstack([x, np.ones_like(x)]).T
    c = np.linalg.lstsq(A, y, rcond=None)[0]
    res = y - A @ c
    return {"slope": float(c[0]), "intercept": float(c[1]), "r2": float(1 - res.var() / y.var()), "n": len(r)}


# ------------------------------------------------------------------- law
def cmd_law(args) -> None:
    AN.mkdir(exist_ok=True)
    allrows, fits = [], {}
    for rs in refs():
        ref = load(ROOT / rs / rs / "bf16.pt")
        rows = points(rs)
        if not rows:
            continue
        iso = Iso(ref) if "full" in ref else None
        for p in rows:
            if iso is not None and p["kl"] < 3:
                pred, s = iso.flips_at_kl(p["kl_full_sub"] if "kl_full_sub" in p else p["kl"])
                p["iso_pred_sub"] = pred
                p["obs_over_iso"] = (p.get("flip_sub", p["flip"])) / pred if pred > 0 else float("nan")
                p["sigma"] = s
        for p in rows:
            p["corpus"] = corpus_of(rs)
        fits[rs] = {"top": fit_loglog(rows), "full_sub": fit_loglog(rows, "kl_full_sub", "flip_sub")}
        print(f"\n=== {rs}: {fits[rs]}")
        print(f"{'config':34s} {'familia':10s} {'flip%':>7s} {'KL':>8s} {'KLfull':>8s} {'k=f/sqrtKL':>10s} {'obs/iso':>7s}")
        for p in sorted(rows, key=lambda r: r["kl"]):
            print(f"{p['config'][:34]:34s} {p['family'][:10]:10s} {100*p['flip']:7.2f} {p['kl']:8.4f} "
                  f"{p.get('kl_full_sub', float('nan')):8.4f} {p['flip']/max(p['kl'],1e-12)**.5:10.3f} "
                  f"{p.get('obs_over_iso', float('nan')):7.2f}")
        allrows += rows
    # o ajuste conjunto do paper e o do corpus mix; cada corpus novo tem o seu. Juntar
    # tudo mudaria a inclinacao reportada sem ninguem ter pedido.
    pooled = fit_loglog([p for p in allrows if p["corpus"] == "mix"])
    pooled_by_corpus = {c: fit_loglog([p for p in allrows if p["corpus"] == c])
                        for c in sorted({p["corpus"] for p in allrows})}
    print("\nPOOLED (corpus do paper, mix):", pooled)
    for c, f in pooled_by_corpus.items():
        print(f"  POOLED {c}: {f}")
    fams = {}
    for p in allrows:
        if "obs_over_iso" in p and p["kl"] < 0.5:
            fams.setdefault(p["family"], []).append(p["obs_over_iso"])
    print("obs/iso por familia (KL<0.5): ", {k: (round(float(np.median(v)), 3), round(float(np.min(v)), 3),
                                                round(float(np.max(v)), 3), len(v)) for k, v in fams.items()})
    trunc = [(p["kl_top_sub"], p["kl_full_sub"]) for p in allrows if "kl_full_sub" in p]
    ratio = np.array([b / a for a, b in trunc if a > 1e-4])
    print("KL completo / KL top-64 (subconjunto): mediana %.3f, p5 %.3f, p95 %.3f" % tuple(np.percentile(ratio, [50, 5, 95])))
    (AN / "law.json").write_text(json.dumps({"rows": allrows, "fits": fits, "pooled": pooled,
                                             "pooled_by_corpus": pooled_by_corpus}, indent=1))


# ---------------------------------------------------------------- theory
def analytic_iso(ref: dict, sigmas, mask=None, tie_eps: float = 1e-6):
    """Previsao ANALITICA (sem Monte Carlo) do ruido isotropico N(0, s^2) nos logits.

    flips(s) ~ m_tie * 1/2 + E_t[ sum_j Phi(-g_tj / (s*sqrt2)) ]  (competidores independentes, 1a ordem em uniao)
    KL(s)    ~ 1/2 s^2 E_t[1 - sum_i p_i^2]
    Empates exatos (g < tie_eps) - artefato da grade dos logits bf16 - entram como termo constante.
    """
    from scipy.stats import norm
    v = ref["topv"].double()
    if mask is not None:
        v = v[mask]
    gaps = (v[:, :1] - v[:, 1:]).numpy()             # [T, 63]
    tie = gaps[:, 0] < tie_eps
    curv = float((1 - (np.exp(v.numpy()) ** 2).sum(-1)).mean())
    out = []
    for s in sigmas:
        cont = norm.sf(gaps[~tie] / (s * math.sqrt(2))).sum(-1)
        flips = (tie.sum() * 0.5 + np.minimum(cont, 1.0).sum()) / len(gaps)
        kl = 0.5 * s * s * curv
        out.append({"sigma": float(s), "kl": kl, "flips": float(flips)})
    # densidade continua em 0 (so faz sentido com logits fp32): fracao com 0<g<h / h
    g0 = gaps[:, 0][~tie]
    rho = [float(((g0 < h).sum() / len(gaps)) / h) for h in (0.01, 0.02, 0.05)]
    return {"curve": out, "tie_mass": float(tie.mean()), "curv": curv, "rho0_top2_cont": rho,
            "kappa0_theory": rho[1] * math.sqrt(2 / (math.pi * curv))}


def cmd_theory(args) -> None:
    law = json.loads((AN / "law.json").read_text())
    out = {}
    for rs in refs():
        ref = load(ROOT / rs / rs / "bf16.pt")
        corpus = json.loads((ROOT / "corpora" / f"{rs.replace('__fp32', '')}.json").read_text())
        src = torch.tensor([r["source"] == "gsm8k" for r, g in zip(corpus["records"], corpus["gen_ids"]) for _ in g])
        rows = [p for p in law["rows"] if p["ref"] == rs and p["model"] == rs and p["family"] != "outro modelo"]
        if not rows:
            continue
        res = {"domains": {}}
        for dom, mask in (("todos", None), ("gsm8k", src), ("mmlu_pt", ~src)):
            sig = np.geomspace(0.02, 8, 60)
            th = analytic_iso(ref, sig, mask)
            res["domains"][dom] = th
        th = res["domains"]["todos"]
        ks, kl_c, fl_c = th["curve"], np.log([c["kl"] for c in th["curve"]]), np.log([c["flips"] for c in th["curve"]])
        comp = []
        for p in sorted(rows, key=lambda r: r["kl"]):
            pred = float(np.exp(np.interp(math.log(p["kl"]), kl_c, fl_c)))
            # anisotropia: deslocamento do gap top1-top2 medido
            c = load(ROOT / rs / rs / (p["config"].replace("/", "--") + ".pt"))
            v = ref["topv"].double(); g12 = v[:, 0] - v[:, 1]
            dgap = (c["at_ref"].double()[:, 0] - c["at_ref"].double()[:, 1]) - g12
            sigma_iso = math.sqrt(2 * p["kl"] / th["curv"])
            comp.append({"config": p["config"], "family": p["family"], "kl": p["kl"], "flip": p["flip"],
                         "analytic_iso": pred, "obs_over_analytic": p["flip"] / pred,
                         "gap_shift_ratio": dgap.std().item() / (sigma_iso * math.sqrt(2))})
        res["configs"] = comp
        small = [x["flip"] / math.sqrt(x["kl"]) for x in comp if 1e-3 < x["kl"] < 0.05]
        res["kappa_obs_smallKL"] = float(np.median(small)) if small else None
        out[rs] = res
        rat = [x["obs_over_analytic"] for x in comp if x["kl"] < 0.5]
        print(f"\n=== {rs}: empates {100*th['tie_mass']:.2f}% | kappa0 teoria (rho continuo) {th['kappa0_theory']:.3f} | "
              f"kappa obs (KL 1e-3..0.05) {res['kappa_obs_smallKL']} | obs/analitico mediana {np.median(rat):.2f} "
              f"[{np.min(rat):.2f}, {np.max(rat):.2f}] n={len(rat)}")
        for dom in ("gsm8k", "mmlu_pt"):
            d = res["domains"][dom]
            print(f"    {dom}: empates {100*d['tie_mass']:.2f}% curvatura {d['curv']:.3f} kappa0 {d['kappa0_theory']:.3f}")
        for x in comp:
            if x["kl"] < 0.6:
                print(f"    {x['config'][:26]:26s} {x['family'][:9]:9s} KL {x['kl']:.4f} flip {100*x['flip']:6.2f}% analitico {100*x['analytic_iso']:6.2f}% "
                      f"obs/an {x['obs_over_analytic']:.2f} | std(dgap)/iso {x['gap_shift_ratio']:.2f}")
    (AN / "theory.json").write_text(json.dumps(out, indent=1))


# -------------------------------------------------------------------- d3
def temperature_residual(ref: dict, c: dict) -> dict:
    """KL restante apos a melhor temperatura global aplicada ao modelo distinto (top-64 renormalizado)."""
    P = ref["topv"].double()
    P = P - torch.logsumexp(P, -1, keepdim=True)
    Q = c["at_ref"].double()
    best = (float("inf"), 1.0)
    for tau in np.geomspace(0.25, 4.0, 49):
        Qt = (Q / tau) - torch.logsumexp(Q / tau, -1, keepdim=True)
        kl = (P.exp() * (P - Qt)).sum(-1).mean().item()
        best = min(best, (kl, float(tau)))
    Q1 = Q - torch.logsumexp(Q, -1, keepdim=True)
    kl1 = (P.exp() * (P - Q1)).sum(-1).mean().item()
    return {"kl_top_renorm": kl1, "kl_after_temperature": best[0], "tau": best[1]}


def cmd_d3(args) -> None:
    """Modelos distintos x compressao: obs/iso antes e depois de remover a parte do KL explicada por temperatura."""
    out = []
    for rs in refs():
        ref = load(ROOT / rs / rs / "bf16.pt")
        if "full" not in ref:
            continue
        iso = Iso(ref)
        for d in sorted((ROOT / rs).iterdir()):
            if not d.is_dir():
                continue
            distinct = d.name != rs
            for f in sorted(d.glob("*.pt")):
                if f.stem == "bf16" or f.stem.startswith("lq"):
                    continue
                c = load(f)
                kl_full = c["kl_full"].double().nanmean().item()
                if kl_full < 1e-3 or kl_full > 2:
                    continue
                flip_sub = (c["top1"] != ref["top1"]).double()[ref["subset"]].mean().item()
                tr = temperature_residual(ref, c)
                frac = tr["kl_after_temperature"] / max(tr["kl_top_renorm"], 1e-12)
                pred, _ = iso.flips_at_kl(kl_full)
                pred_t, _ = iso.flips_at_kl(kl_full * frac)
                row = {"ref": rs, "model": d.name, "config": f.stem, "distinct": distinct, "family": family(f.stem),
                       "kl_full_sub": kl_full, "flip_sub": flip_sub, "obs_over_iso": flip_sub / pred,
                       "obs_over_iso_after_temp": flip_sub / pred_t, **tr}
                out.append(row)
                if distinct or f.stem in ("u4", "u3", "gptq3", "wanda50", "mag40", "kv2", "sgpt50", "awq3"):
                    print(f"{rs[:22]:22s} <- {d.name[:42]:42s} {f.stem:7s} KL {kl_full:.3f} obs/iso {row['obs_over_iso']:.2f} "
                          f"-> apos temperatura {row['obs_over_iso_after_temp']:.2f} (tau* {tr['tau']:.2f}, KL x{frac:.2f})")
    dist = [r for r in out if r["distinct"] and r["config"] == "raw"]
    comp = [r for r in out if not r["distinct"] and 0.05 < r["kl_full_sub"] < 1.0]
    for name, grp in (("modelos distintos (raw)", dist), ("compressao (mesmo modelo, KL 0.05-1)", comp)):
        if grp:
            a = np.array([r["obs_over_iso"] for r in grp]); b = np.array([r["obs_over_iso_after_temp"] for r in grp])
            t = np.array([r["tau"] for r in grp]); k = np.array([r["kl_after_temperature"] / r["kl_top_renorm"] for r in grp])
            print(f"RESUMO {name}: n={len(grp)} obs/iso mediana {np.median(a):.2f} -> apos temp {np.median(b):.2f} | "
                  f"tau* mediana {np.median(t):.2f} | fracao de KL restante {np.median(k):.2f}")
    (AN / "d3.json").write_text(json.dumps(out, indent=1))


# -------------------------------------------------------------------- d4
def cmd_d4(args) -> None:
    out = {}
    for rs in refs():
        ref = load(ROOT / rs / rs / "bf16.pt")
        iso = Iso(ref, n_tok=1024)
        curv = analytic_iso(ref, [1.0])["curv"]
        grid = []
        for s in np.geomspace(0.05, 6, 14):
            kl, fl = iso.at(float(s))
            grid.append({"sigma": float(s), "kl_true": kl, "kl_linear": 0.5 * s * s * curv, "flip": fl})
        k_small = grid[1]["flip"] / math.sqrt(grid[1]["kl_true"])
        for g in grid:
            g["pred_true"] = k_small * math.sqrt(g["kl_true"])
            g["pred_linear"] = k_small * math.sqrt(g["kl_linear"])
        out[rs] = {"iso_grid": grid}
        print(f"\n=== {rs} (isotropico): sigma | KL real | KL linearizado | flips | k*sqrt(KL real) | k*sqrt(KL lin)")
        for g in grid:
            print(f"  {g['sigma']:.3f} | {g['kl_true']:.4f} | {g['kl_linear']:.4f} | {100*g['flip']:.2f}% | "
                  f"{100*g['pred_true']:.2f}% | {100*g['pred_linear']:.2f}%")
        # aditividade por camada (se houver lq3_L com at_ref)
        d = ROOT / rs / rs
        singles = sorted(d.glob("lq3_*.pt"), key=lambda f: int(f.stem.split("_")[1]))
        if singles and (d / "u3.pt").exists():
            v = ref["topv"].double(); g12 = v[:, 0] - v[:, 1]
            def stats(f):
                c = load(f)
                q = c["at_ref"].double()
                dg = (q[:, 0] - q[:, 1]) - g12
                return {"kl": c["kl"].double().mean().item(), "var_gap": dg.var().item(),
                        "flip": (c["top1"] != ref["top1"]).double().mean().item()}
            per = [stats(f) for f in singles]
            u3 = stats(d / "u3.pt")
            s_kl = sum(p["kl"] for p in per); s_var = sum(p["var_gap"] for p in per)
            out[rs]["additivity"] = {"sum_kl": s_kl, "kl_u3": u3["kl"], "sum_var_gap": s_var,
                                     "var_gap_u3": u3["var_gap"], "flip_u3": u3["flip"], "n_layers": len(per)}
            print(f"  aditividade ({len(per)} camadas): sum KL {s_kl:.4f} vs KL(u3) {u3['kl']:.4f} (x{s_kl/u3['kl']:.2f}) | "
                  f"sum Var(dgap) {s_var:.3f} vs Var(dgap u3) {u3['var_gap']:.3f} (x{s_var/u3['var_gap']:.2f})")
    AN.mkdir(exist_ok=True)
    (AN / "d4.json").write_text(json.dumps(out, indent=1))


# ------------------------------------------------------------------ prop
def cmd_prop(args) -> None:
    """Proposicao de 1a ordem: flips = m_tie*pi_tie + rho_c(0) * E|dgap| / 2, sem ajuste."""
    law = json.loads((AN / "law.json").read_text())
    out = {}
    for rs in refs():
        ref = load(ROOT / rs / rs / "bf16.pt")
        v = ref["topv"].double()
        g12 = v[:, 0] - v[:, 1]
        tie = g12 < 1e-6
        hs = torch.tensor([0.02, 0.05, 0.1, 0.2])
        dens = torch.stack([((g12 > 1e-6) & (g12 < h)).double().mean() / h for h in hs])
        # extrapolacao linear para h -> 0 (densidade continua de gaps em 0)
        A = torch.stack([torch.ones_like(hs), hs], 1).double()
        rho0 = float(torch.linalg.lstsq(A, dens[:, None]).solution[0])
        rows = [p for p in law["rows"] if p["ref"] == rs and p["model"] == rs and p["family"] != "outro modelo"]
        res = []
        for p in sorted(rows, key=lambda r: r["kl"]):
            c = load(ROOT / rs / rs / (p["config"].replace("/", "--") + ".pt"))
            q = c["at_ref"].double()
            dgap = (q[:, 0] - q[:, 1]) - g12
            near = (~tie) & (g12 < 1.0)
            e_abs = dgap[near].abs().mean().item()
            flip = (c["top1"] != ref["top1"])
            pi_tie = flip[tie].double().mean().item() if tie.any() else 0.0
            pred = tie.double().mean().item() * pi_tie + rho0 * e_abs / 2
            aniso = e_abs / math.sqrt(2 * p["kl"])  # isotropico: E|N(0,2s^2)| / sqrt(s^2 C) = (2/sqrt(pi))/sqrt(C)
            res.append({"config": p["config"], "family": p["family"], "kl": p["kl"], "flip": p["flip"],
                        "pred_first_order": pred, "ratio": p["flip"] / pred, "E_abs_dgap": e_abs,
                        "anisotropy_A": aniso, "pi_tie": pi_tie})
        out[rs] = {"rho0_cont": rho0, "tie_mass": tie.double().mean().item(), "configs": res}
        r = [x["ratio"] for x in res if x["kl"] < 0.5]
        med_str = f"{np.median(r):.2f} [{np.min(r):.2f},{np.max(r):.2f}]" if r else "-"
        print(f"\n=== {rs}: rho_c(0) {rho0:.4f}/nat | empates {100*out[rs]['tie_mass']:.2f}% | obs/pred mediana "
              f"{med_str} n={len(r)}")
        for x in res:
            if x["kl"] < 0.6:
                print(f"   {x['config'][:24]:24s} {x['family'][:9]:9s} KL {x['kl']:.4f} flip {100*x['flip']:6.2f}% "
                      f"pred {100*x['pred_first_order']:6.2f}% obs/pred {x['ratio']:.2f} | A {x['anisotropy_A']:.3f} | pi_tie {x['pi_tie']:.2f}")
    (AN / "prop.json").write_text(json.dumps(out, indent=1))


# ------------------------------------------------------------ closedloop
def cmd_closedloop(args) -> None:
    """Preve mudanca de resposta/acuracia GSM8K e aceitacao especulativa a partir do teacher forcing."""
    law = json.loads((AN / "law.json").read_text())
    rows_out = []
    for d in sorted((ROOT / "closedloop").iterdir()):
        rs = d.name
        base_f = d / "gsm8k_bf16.json"
        corpus = json.loads((ROOT / "corpora" / f"{rs}.json").read_text()) if (ROOT / "corpora" / f"{rs}.json").exists() else None
        ref = load(ROOT / rs / rs / "bf16.pt") if (ROOT / rs / rs / "bf16.pt").exists() else None
        if ref is None or corpus is None:
            continue
        gsm_mask = torch.tensor([r["source"] == "gsm8k" for r, g in zip(corpus["records"], corpus["gen_ids"]) for _ in g])
        L = np.mean([len(g) for r, g in zip(corpus["records"], corpus["gen_ids"]) if r["source"] == "gsm8k"])
        base = json.loads(base_f.read_text()) if base_f.exists() else None
        for f in sorted(d.glob("gsm8k_*.json")):
            cfg = f.stem[len("gsm8k_"):]
            if cfg == "bf16" or base is None:
                continue
            j = json.loads(f.read_text())
            tf = ROOT / rs / rs / f"{cfg}.pt"
            if not tf.exists():
                continue
            c = load(tf)
            flip_g = (c["top1"] != ref["top1"])[gsm_mask].double().mean().item()
            kl_g = c["kl"][gsm_mask].double().mean().item()
            changed = np.mean([a["answer"] != b["answer"] for a, b in zip(j["rows"], base["rows"])])
            rows_out.append({"model": rs, "config": cfg, "kl_gsm": kl_g, "flip_gsm": flip_g, "L": float(L),
                             "acc": j["acc"], "acc_bf16": base["acc"], "d_acc": j["acc"] - base["acc"],
                             "answer_changed": float(changed)})
    if not rows_out:
        print("sem dados de malha fechada ainda")
        return
    # modelo de 1 parametro: P(resposta muda) = 1 - exp(-lambda * L * flip); lambda ajustado deixando um modelo de fora
    from scipy.optimize import minimize_scalar
    models = sorted({r["model"] for r in rows_out})
    def fit(rows):
        f = lambda lam: sum((1 - math.exp(-lam * r["L"] * r["flip_gsm"]) - r["answer_changed"]) ** 2 for r in rows)
        return minimize_scalar(f, bounds=(1e-4, 5), method="bounded").x
    preds = []
    for m in models:
        lam = fit([r for r in rows_out if r["model"] != m]) if len(models) > 1 else fit(rows_out)
        for r in rows_out:
            if r["model"] == m:
                r["pred_changed_lomo"] = 1 - math.exp(-lam * r["L"] * r["flip_gsm"])
                r["lambda_lomo"] = lam
                preds.append((r["pred_changed_lomo"], r["answer_changed"]))
    p_, o_ = np.array(preds).T
    r2 = 1 - ((o_ - p_) ** 2).sum() / ((o_ - o_.mean()) ** 2).sum()
    print(f"P(resposta muda) ~ 1-exp(-lambda*L*flip): R2 deixando-um-modelo-de-fora = {r2:.3f} (n={len(preds)})")
    # d_acc vs sqrt(KL)
    x = np.sqrt([r["kl_gsm"] for r in rows_out]); y = np.array([r["d_acc"] for r in rows_out])
    c = np.polyfit(x, y, 1); res = y - np.polyval(c, x)
    print(f"d_acc = {c[0]:.3f}*sqrt(KL) + {c[1]:.3f}: R2 {1 - res.var()/y.var():.3f}; "
          f"vs linear em KL: R2 {1 - (y - np.polyval(np.polyfit(x**2, y, 1), x**2)).var()/y.var():.3f}")
    for r in sorted(rows_out, key=lambda r: (r["model"], r["kl_gsm"])):
        print(f"   {r['model'][:24]:24s} {r['config']:10s} KL {r['kl_gsm']:.4f} flip {100*r['flip_gsm']:5.2f}% | muda {100*r['answer_changed']:5.1f}% "
              f"(prev {100*r.get('pred_changed_lomo', float('nan')):5.1f}%) | acc {100*r['acc']:5.1f} (bf16 {100*r['acc_bf16']:5.1f})")
    # especulativa
    spec = []
    for d in sorted((ROOT / "closedloop").iterdir()):
        rs = d.name
        ref_p = ROOT / rs / rs / "bf16.pt"
        if not ref_p.exists():
            continue
        ref = load(ref_p)
        corpus = json.loads((ROOT / "corpora" / f"{rs}.json").read_text())
        gsm_mask = torch.tensor([r["source"] == "gsm8k" for r, g in zip(corpus["records"], corpus["gen_ids"]) for _ in g])
        for f in sorted(d.glob("spec_*.json")):
            j = json.loads(f.read_text())
            name = f.stem[len("spec_"):]
            tf = ROOT / rs / (name.split("_")[0] if name.startswith(("Qwen", "gemma")) and "_" in name else rs) / \
                 f"{name.split('_')[-1]}.pt"
            if not tf.exists():
                continue
            c = load(tf)
            flip = (c["top1"] != ref["top1"])[gsm_mask].double().mean().item()
            kl = c["kl"][gsm_mask].double().mean().item()
            tv = c["tv"][gsm_mask].double().mean().item()
            spec.append({"model": rs, "draft": name, "greedy_accept": j["greedy"]["accept_rate"],
                         "sample_accept_prob": j["sample"]["mean_accept_prob"], "tf_agree": 1 - flip,
                         "tf_1_minus_tv": 1 - tv, "kl": kl})
    for sp in spec:
        print(f"   ESPEC {sp['model'][:20]:20s} draft {sp['draft']:22s} KL {sp['kl']:.4f} | greedy: aceita {sp['greedy_accept']:.3f} "
              f"(1-flip TF {sp['tf_agree']:.3f}) | amostra: E[min(1,p/q)] {sp['sample_accept_prob']:.3f} (1-TV TF {sp['tf_1_minus_tv']:.3f})")
    (AN / "closedloop.json").write_text(json.dumps({"gsm8k": rows_out, "spec": spec, "lomo_r2": r2}, indent=1))


# -------------------------------------------------------------------- d2
def cmd_d2(args) -> None:
    from math import comb
    def mcnemar(a, b):
        n = a + b
        if n == 0:
            return 1.0
        k = min(a, b)
        return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    out = {}
    for f in sorted((ROOT / "d2").glob("*.json")):
        st = json.loads(f.read_text())
        if "picks" not in st:
            continue
        test = st["test"]
        if json.dumps({}, sort_keys=True) not in test:
            print(f"(pulando {f.stem}: bf16 ainda nao avaliado no teste)")
            continue
        base = test[json.dumps({}, sort_keys=True)]
        print(f"\n=== {f.stem}: bf16 raw {base['raw']:.4f} inv {base['inv']:.4f} pc {base['pc']:.4f} prior {base['prior']}")
        res = {}
        for name, cfg in st["picks"].items():
            s = test.get(json.dumps(cfg, sort_keys=True))
            if s is None or name == "bf16":
                continue
            row = {"prior": s["prior"]}
            for m in ("raw", "inv", "pc"):
                a = np.array(s[f"{m}_vec"]); b = np.array(base[f"{m}_vec"])
                row[m] = s[m]; row[f"d_{m}"] = s[m] - base[m]
                row[f"p_{m}"] = mcnemar(int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum()))
            tv = 0.5 * sum(abs(x - y) for x, y in zip(s["prior"], base["prior"]))
            row["prior_tv"] = tv
            res[name] = row
            print(f"   {name:16s} raw {100*row['d_raw']:+6.2f}pp (p={row['p_raw']:.3f}) | inv {100*row['d_inv']:+6.2f}pp "
                  f"(p={row['p_inv']:.3f}) | pc {100*row['d_pc']:+6.2f}pp | TV prior {tv:.3f}")
        # correlacao na busca de precisao mista (calib): ganho raw x ganho inv x TV
        cal = st["calib"]; cb = cal[json.dumps({}, sort_keys=True)]
        mixed = [v for k, v in cal.items() if '"bits"' in k]
        if mixed:
            d_raw = np.array([v["raw"] - cb["raw"] for v in mixed]); d_inv = np.array([v["inv"] - cb["inv"] for v in mixed])
            tvs = np.array([0.5 * sum(abs(x - y) for x, y in zip(v["prior"], cb["prior"])) for v in mixed])
            from scipy.stats import spearmanr
            print(f"   busca mista (calib, n={len(mixed)}): spearman(d_raw, d_inv) {spearmanr(d_raw, d_inv)[0]:.2f} | "
                  f"spearman(d_raw - d_inv, TV prior) {spearmanr(d_raw - d_inv, tvs)[0]:.2f}")
        loo = [(k, v) for k, v in cal.items() if '"skip"' in k and len(json.loads(k)["skip"]) == 1]
        if loo:
            from scipy.stats import spearmanr
            r_raw = [v["raw"] for _, v in loo]; r_inv = [v["inv"] for _, v in loo]; r_nll = [-v["nll"] for _, v in loo]
            print(f"   LOO de camadas (calib, n={len(loo)}): spearman(raw, inv) {spearmanr(r_raw, r_inv)[0]:.2f} | "
                  f"spearman(-nll, inv) {spearmanr(r_nll, r_inv)[0]:.2f} | camadas cuja remocao 'melhora' raw: "
                  f"{sum(v['raw'] > cb['raw'] for _, v in loo)} vs inv: {sum(v['inv'] > cb['inv'] for _, v in loo)}")
        out[f.stem] = res
    (AN / "d2.json").write_text(json.dumps(out, indent=1))


# ------------------------------------------------------------ prop (ablacao de limiar)
def cmd_prop_thresh(args) -> None:
    """Ablacao do limiar de 1 nat: E|dgap| e phi sao condicionados a g12 < h.

    A previsao de 1a ordem usa E|dgap| medido nos tokens frageis (g12 < 1 nat por
    padrao). Se a razao obs/previsto nao se mover ao variar h, o limiar e uma
    conveniencia de estimacao e nao um parametro ajustado; e isso que se testa aqui.
    """
    law = json.loads((AN / "law.json").read_text())
    hs_thresh = [0.5, 1.0, 2.0]
    out = {}
    for rs in refs():
        if "__fp32" not in rs:
            continue
        ref = load(ROOT / rs / rs / "bf16.pt")
        v = ref["topv"].double()
        g12 = v[:, 0] - v[:, 1]
        tie = g12 < 1e-6
        hs = torch.tensor([0.02, 0.05, 0.1, 0.2])
        dens = torch.stack([((g12 > 1e-6) & (g12 < h)).double().mean() / h for h in hs])
        A = torch.stack([torch.ones_like(hs), hs], 1).double()
        rho0 = float(torch.linalg.lstsq(A, dens[:, None]).solution[0])
        rows = [p for p in law["rows"] if p["ref"] == rs and p["model"] == rs and p["family"] != "outro modelo"]
        per_h = {}
        for h in hs_thresh:
            near = (~tie) & (g12 < h)
            res = []
            for p in sorted(rows, key=lambda r: r["kl"]):
                f = ROOT / rs / rs / (p["config"].replace("/", "--") + ".pt")
                if not f.exists():
                    continue
                c = load(f)
                q = c["at_ref"].double()
                dgap = (q[:, 0] - q[:, 1]) - g12
                e_abs = dgap[near].abs().mean().item()
                flip = (c["top1"] != ref["top1"])
                pi_tie = flip[tie].double().mean().item() if tie.any() else 0.0
                pred = tie.double().mean().item() * pi_tie + rho0 * e_abs / 2
                res.append({"config": p["config"], "kl": p["kl"], "flip": p["flip"],
                            "ratio": p["flip"] / pred if pred > 0 else float("nan"),
                            "anisotropy_A": e_abs / math.sqrt(2 * p["kl"])})
            per_h[str(h)] = {"phi": float(((g12 > 1e-6) & (g12 < h)).double().mean()), "configs": res}
        out[rs] = {"rho0_cont": rho0, "by_threshold": per_h}
        msg = "  ".join(f"h={h}: obs/pred {np.median([x['ratio'] for x in per_h[str(h)]['configs'] if x['kl'] < 0.25]):.3f}"
                        for h in hs_thresh if per_h[str(h)]["configs"])
        print(f"=== {rs}: {msg}")
    (AN / "prop_threshold.json").write_text(json.dumps(out, indent=1))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["law", "theory", "d3", "d4", "prop", "prop-thresh", "closedloop", "d2"])
    args = p.parse_args(argv)
    torch.set_grad_enabled(False)
    {"law": cmd_law, "theory": cmd_theory, "d3": cmd_d3, "d4": cmd_d4, "prop": cmd_prop,
     "prop-thresh": cmd_prop_thresh, "closedloop": cmd_closedloop, "d2": cmd_d2}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
