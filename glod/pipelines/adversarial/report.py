#!/usr/bin/env python3
"""P1: consolida a evidencia adversarial.

Junta, por modelo:
  * kappa dos compressores HONESTOS (mediana das configs do estudo de fidelidade);
  * ataque de UMA camada, delta de rank completo no down_proj final (`glod adv-single`);
  * ataque MULTICAMADA, delta de baixo rank em todas as projecoes de todas as
    camadas, com gradiente pelo modelo inteiro (`glod adv-multi`),
    nos dois sentidos: maligno (maximiza flips a KL fixo) e benigno (minimiza,
    controle positivo que mostra que o gradiente tem poder sobre kappa);
  * referencia analitica de temperatura (KL > 0 com zero flips no limite);
  * teto do atacante oraculo, que pode escolher o KL token a token;
  * geometria que explica a assimetria (rank efetivo das direcoes de flip).

    python -m glod adv-report
"""
from __future__ import annotations

import json
import math

import numpy as np

from glod.paths import OUT as ROOT
from glod.paths import corpus_of, model_of
AN = ROOT / "analysis"


def test_mask(model_slug: str, every: int = 5):
    """Mascara dos tokens de TESTE, com a mesma regra do ataque (indice de sequencia % 5 == 0).

    Necessario para comparar macas com macas: o kappa do ataque medido no corpus todo
    inclui os tokens em que ele treinou. Os compressores honestos nao treinam em nada,
    mas o kappa deles tem de ser recalculado na MESMA mascara.
    """
    import torch
    f = ROOT / "corpora" / f"{model_slug.replace('__fp32', '')}.json"
    if not f.exists():
        return None
    gen = json.loads(f.read_text())["gen_ids"]
    return torch.cat([torch.full((len(g),), i % every == 0, dtype=torch.bool)
                      for i, g in enumerate(gen)])


def honest_test(model_slug: str) -> tuple[float, int]:
    """kappa dos honestos restrito aos tokens de teste do ataque."""
    import torch
    m = test_mask(model_slug)
    ref_p = ROOT / model_slug / model_slug / "bf16.pt"
    if m is None or not ref_p.exists():
        return float("nan"), 0
    ref = torch.load(ref_p, map_location="cpu", mmap=True, weights_only=False)
    ks = []
    for f in sorted((ROOT / model_slug / model_slug).glob("*.pt")):
        if f.stem == "bf16":
            continue
        c = torch.load(f, map_location="cpu", mmap=True, weights_only=False)
        if "kl" not in c:
            continue
        kl = c["kl"][m].double().mean().item()
        if 1e-3 < kl < 0.25:
            ks.append((c["top1"] != ref["top1"])[m].double().mean().item() / math.sqrt(kl))
    return (float(np.median(ks)) if ks else float("nan")), len(ks)


def honest_near_kl(model_slug: str, kl: float, tol: float = 0.35) -> tuple[float, int]:
    """kappa honesto MEDIDO PERTO DO MESMO KL, nos tokens de teste.

    Comparar o melhor orcamento do ataque contra a mediana honesta de todo o intervalo
    favorece o ataque: kappa honesto e levemente maior em KL baixo (piso de empates do
    bf16) e o ataque rende mais em KL alto.
    """
    import torch
    m = test_mask(model_slug)
    ref_p = ROOT / model_slug / model_slug / "bf16.pt"
    if m is None or not ref_p.exists():
        return float("nan"), 0
    ref = torch.load(ref_p, map_location="cpu", mmap=True, weights_only=False)
    ks = []
    for f in sorted((ROOT / model_slug / model_slug).glob("*.pt")):
        if f.stem == "bf16":
            continue
        c = torch.load(f, map_location="cpu", mmap=True, weights_only=False)
        if "kl" not in c:
            continue
        k = c["kl"][m].double().mean().item()
        if abs(k / kl - 1) <= tol:
            ks.append((c["top1"] != ref["top1"])[m].double().mean().item() / math.sqrt(k))
    return (float(np.median(ks)) if ks else float("nan")), len(ks)


def honest(model_slug: str) -> tuple[float, int]:
    law = json.loads((AN / "law.json").read_text())
    ks = [r["flip"] / math.sqrt(r["kl"]) for r in law["rows"]
          if r["ref"] == model_slug and r["model"] == model_slug
          and r.get("family") != "outro modelo" and 1e-3 < r["kl"] < 0.25]
    if not ks:
        import torch
        ref_p = ROOT / model_slug / model_slug / "bf16.pt"
        if ref_p.exists():
            ref = torch.load(ref_p, map_location="cpu", mmap=True, weights_only=False)
            for f in sorted((ROOT / model_slug / model_slug).glob("*.pt")):
                if f.stem == "bf16":
                    continue
                c = torch.load(f, map_location="cpu", mmap=True, weights_only=False)
                if "kl" not in c:
                    continue
                kl = c["kl"].double().mean().item()
                if 1e-3 < kl < 0.25:
                    ks.append((c["top1"] != ref["top1"]).double().mean().item() / math.sqrt(kl))
    return (float(np.median(ks)) if ks else float("nan")), len(ks)


def anisotropy(model_slug: str) -> dict:
    """A = E|dgap| / sqrt(2 KL) para os ataques salvos e para os honestos do mesmo modelo."""
    import torch
    ref_p = ROOT / model_slug / model_slug / "bf16.pt"
    if not ref_p.exists():
        return {}
    ref = torch.load(ref_p, map_location="cpu", mmap=True, weights_only=False)
    v = ref["topv"].double()
    g12 = v[:, 0] - v[:, 1]
    near = (g12 > 1e-6) & (g12 < 1.0)
    tie = g12 < 1e-6          # empates exatos da grade do bf16 (~0,8% dos passos)

    def one(c) -> dict:
        q = c["at_ref"].double()
        dgap = (q[:, 0] - q[:, 1]) - g12
        e_abs = dgap[near].abs().mean().item()
        kl = c["kl"].double().mean().item()
        fl = (c["top1"] != ref["top1"])
        n_tie_flip = fl[tie].double().sum().item()
        # o fator /2 da previsao de 1a ordem (flips = rho(0) E|dgap| / 2) supoe SINAL
        # simetrico: metade dos deslocamentos fecha o gap, metade abre. Um ataque que
        # alinhe o sinal ganharia ate 2x sem aumentar |dgap| nenhum.
        frac_neg = (dgap[near] < 0).double().mean().item()
        return {"E_abs_dgap": e_abs, "kl": kl, "A": e_abs / math.sqrt(2 * max(kl, 1e-12)),
                "flip": fl.double().mean().item(), "frac_dgap_negative": frac_neg,
                # quanto dos flips vem dos empates: um atacante poderia explorar o
                # artefato do bf16 (virar empates custa KL ~ 0)
                "tie_share_of_flips": n_tie_flip / max(fl.double().sum().item(), 1.0),
                "tie_mass": tie.double().mean().item()}

    out = {"attack": {}, "honest": {}}
    for f in sorted((ROOT / "adversarial" / model_slug).glob("fid_*.pt")):
        if f.stem.count("_") != 2:
            continue   # fid_<tag>_<modo>_<kl>.pt e variante; so o ataque principal entra
        out["attack"][f.stem[len("fid_"):]] = one(torch.load(f, map_location="cpu", mmap=True,
                                                             weights_only=False))
    for f in sorted((ROOT / model_slug / model_slug).glob("*.pt")):
        if f.stem == "bf16":
            continue
        c = torch.load(f, map_location="cpu", mmap=True, weights_only=False)
        if "at_ref" in c and 1e-3 < c["kl"].double().mean().item() < 0.25:
            out["honest"][f.stem] = one(c)
    return out


def fp32_check() -> dict:
    """P5: o mesmo ataque com lm_head em bf16 e em fp32, kappa de teste / kappa honesto de teste.

    Se o piso de empates exatos do bf16 inflasse kappa_honesto no KL baixo, a razao do
    ataque subiria em fp32 justamente no orcamento pequeno.
    """
    out = {}
    print("\n  P5: bf16 x fp32 (kappa de teste / kappa honesto de teste, mesmo ataque)")
    print(f"    {'modelo':24s} {'hon. bf16':>9s} {'hon. fp32':>9s}  {'ponto':18s} {'bf16':>6s} {'fp32':>6s} {'delta':>6s}")
    deltas = {"malign": [], "benign": []}
    for d in sorted((ROOT / "adversarial").glob("*__fp32")):
        ms = d.name.replace("__fp32", "")
        fb, ff = ROOT / "adversarial" / ms / "results_multi.json", d / "results_fp32.json"
        if not (fb.exists() and ff.exists()):
            continue
        rb, rf = json.loads(fb.read_text()), json.loads(ff.read_text())
        hb, hf = honest_test(ms)[0], honest_test(d.name)[0]
        out[ms] = {"honest_test_bf16": hb, "honest_test_fp32": hf, "points": {}}
        for key in sorted(set(rb) & set(rf)):
            if "_multi@" not in key or rb[key].get("rank") != rf[key].get("rank"):
                continue
            xb, xf = rb[key]["test"]["kappa"] / hb, rf[key]["test"]["kappa"] / hf
            out[ms]["points"][key] = {"bf16": xb, "fp32": xf}
            deltas[key.split("_")[0]].append(xf - xb)
            print(f"    {ms[:24]:24s} {hb:9.3f} {hf:9.3f}  {key:18s} {xb:6.2f} {xf:6.2f} {xf - xb:+6.2f}")
    for mode, v in deltas.items():
        if len(v) > 1:
            se = np.std(v, ddof=1) / math.sqrt(len(v))
            print(f"    delta fp32 - bf16, {mode:6s}: {np.mean(v):+.3f}  IC95 [{np.mean(v) - 1.96*se:+.3f}, "
                  f"{np.mean(v) + 1.96*se:+.3f}]  (n={len(v)})")
    out["deltas"] = deltas
    return out


def seed_check() -> dict:
    """Variancia entre sementes do mesmo ataque (rank 16, com escala de saida), e P7 (sem escala).

    kappa de teste / kappa honesto de teste. A semente 0 e o arquivo results_multi.json.
    """
    out = {"seeds": {}, "no_norm": {}}
    print("\n  sementes: kappa de teste / honesto de teste, por ponto (semente 0, 1, 2)")
    for d in sorted((ROOT / "adversarial").iterdir()):
        files = sorted(d.glob("results_seed*.json"))
        if d.name.endswith("__fp32") or not files or not (d / "results_multi.json").exists():
            continue
        kht = honest_test(d.name)[0]
        runs = [json.loads((d / "results_multi.json").read_text())] + [json.loads(f.read_text()) for f in files]
        for key in sorted(runs[0]):
            if "_multi@" not in key:
                continue
            v = [r[key]["test"]["kappa"] / kht for r in runs if key in r]
            if len(v) < 2:
                continue
            out["seeds"][f"{d.name}/{key}"] = v
            print(f"    {d.name[:24]:24s} {key:20s} " + " ".join(f"{x:5.2f}" for x in v) +
                  f" | media {np.mean(v):.2f} desvio {np.std(v, ddof=1):.3f} amplitude {max(v) - min(v):.2f}")
    print("\n  P7: sem acesso a escala de saida (peso da norma final), KL 0,05, kappa de teste / honesto de teste")
    for d in sorted((ROOT / "adversarial").iterdir()):
        fs, fm = d / "results_semnorma.json", d / "results_multi.json"
        if not (fs.exists() and fm.exists()):
            continue
        kht = honest_test(d.name)[0]
        rs, rm = json.loads(fs.read_text()), json.loads(fm.read_text())
        for key in sorted(set(rs) & set(rm)):
            xm, xs = rm[key]["test"]["kappa"] / kht, rs[key]["test"]["kappa"] / kht
            out["no_norm"][f"{d.name}/{key}"] = {"with_norm": xm, "no_norm": xs}
            print(f"    {d.name[:24]:24s} {key:20s} com norma {xm:5.2f} | sem norma {xs:5.2f} | delta {xs - xm:+.2f}")
    return out


def _ci(v: list[float]) -> str:
    if not v:
        return "-"
    if len(v) < 2:
        return f"{v[0]:.2f}x (n=1)"
    se = np.std(v, ddof=1) / math.sqrt(len(v))
    return f"{np.mean(v):.2f}x [{np.mean(v) - 1.96*se:.2f}, {np.mean(v) + 1.96*se:.2f}] (n={len(v)})"


def corpus_summary(by_corpus: dict, fp32: dict, seeds: dict) -> dict:
    """A pergunta da varredura: kappa e a razao do ataque mudam com o corpus?

    Tudo em tokens de teste. Razao = kappa do ataque / kappa honesto no mesmo KL (+-35%),
    melhor variante do atacante por (modelo, orcamento), sementes extras fora.
    """
    out = {}
    order = ["mix"] + sorted(c for c in by_corpus if c != "mix")
    print("\n  POR CORPUS (mix = corpus do paper): razao kappa_ataque / kappa_honesto, pareada por KL, teste")
    print(f"    {'corpus':9s} {'maligno':30s} {'benigno':30s} {'fp32-bf16 malig.':>17s} "
          f"{'desvio sementes':>16s} {'sem norma':>10s}")
    for c in order:
        if c not in by_corpus:
            continue
        g = by_corpus[c]
        d32 = [pt["fp32"] - pt["bf16"] for m, v in fp32.items() if m != "deltas" and corpus_of(m) == c
               for k, pt in v["points"].items() if k.startswith("malign")]
        sd = [float(np.std(v, ddof=1)) for k, v in seeds["seeds"].items()
              if corpus_of(k.split("/")[0]) == c and "/malign" in k and len(v) > 1]
        nn = [v["no_norm"] - v["with_norm"] for k, v in seeds["no_norm"].items()
              if corpus_of(k.split("/")[0]) == c and "/malign" in k]
        row = {"malign": g["malign"], "benign": g["benign"], "fp32_delta_malign": d32,
               "seed_sd_malign": sd, "no_norm_delta_malign": nn}
        out[c] = row
        f = lambda v: f"{np.mean(v):+.3f} (n={len(v)})" if v else "-"
        print(f"    {c:9s} {_ci(g['malign']):30s} {_ci(g['benign']):30s} {f(d32):>17s} "
              f"{(f'{np.median(sd):.3f} (n={len(sd)})' if sd else '-'):>16s} {f(nn):>10s}")

    # tabela modelo x corpus: kappa honesto (teste) e razao media do maligno
    models = sorted({pt["model"] for g in by_corpus.values() for pt in g["points"]})
    print("\n  kappa honesto (teste) | razao do maligno (media dos orcamentos), por modelo x corpus")
    print("    " + f"{'modelo':26s}" + "".join(f"{c:>18s}" for c in order if c in by_corpus))
    table = {}
    for m in models:
        cells = []
        for c in order:
            if c not in by_corpus:
                continue
            ref = m if c == "mix" else f"{m}__{c}"
            kh = honest_test(ref)[0]
            rr = [pt["ratio"] for pt in by_corpus[c]["points"] if pt["model"] == m and pt["mode"] == "malign"]
            table.setdefault(m, {})[c] = {"kappa_honest_test": kh, "malign_ratio": rr}
            cells.append(("-" if math.isnan(kh) else f"{kh:.3f}") + " | " +
                         (f"{np.mean(rr):.2f}x" if rr else "  -  "))
        print("    " + f"{m[:26]:26s}" + "".join(f"{x:>18s}" for x in cells))
    out["by_model"] = table
    return out


def main() -> int:
    fd = json.loads((AN / "flipdirs.json").read_text()) if (AN / "flipdirs.json").exists() else {}
    geo = {k.split("/")[-1]: v for k, v in fd.items()}
    out = {"models": {}}
    print(f"{'modelo':24s} {'kappa hon.':>10s} {'1 camada':>20s} {'multi maligno':>22s} "
          f"{'multi benigno':>22s} {'temperatura':>12s}")
    print(f"{'':24s} {'(n)':>10s} {'kappa (x hon.)':>20s} {'kappa (x hon.)':>22s} {'kappa (x hon.)':>22s}")
    rows = []
    for d in sorted((ROOT / "adversarial").iterdir()):
        ms = d.name
        if ms.endswith("__fp32"):
            continue   # checagem de robustez (P5), comparada a parte em fp32_check(); contaria o ataque 2x
        kh, nh = honest(ms)
        if math.isnan(kh):
            continue
        single, multi_m, multi_b, temp = [], [], [], []
        for f in d.glob("results*.json"):
            if "smoke" in f.name or "_v2" in f.name or "_seed" in f.name:
                # v2 = checagem de protocolo com os mesmos ajustes (duplicaria pontos);
                # sementes extras medem variancia (seed_check) e escolher a melhor inflaria o ataque
                continue
            for key, r in json.loads(f.read_text()).items():
                if key.startswith("temperatura"):
                    temp.append(r["kappa"])
                elif r.get("mode") == "malign_multi":
                    multi_m.append((r["all"]["kappa"], r["all"]["kl"], r["all"]["flip"],
                                    r["ceiling_fraction"], r.get("kappa_val_initial"),
                                    r["test"]["kappa"], r["budget"]))
                elif r.get("mode") == "benign_multi":
                    multi_b.append((r["all"]["kappa"], r["all"]["kl"], r["all"]["flip"],
                                    r["ceiling_fraction"], r.get("kappa_val_initial"),
                                    r["test"]["kappa"], r["budget"]))
                elif r.get("mode") == "malign":
                    single.append((r["kappa"], r["kl"], r["flip"]))
        fmt = lambda v: f"{max(v):.3f} ({max(v)/kh:.2f}x)" if v else "-"
        fmt1 = lambda v: f"{max(x[0] for x in v):.3f} ({max(x[0] for x in v)/kh:.2f}x)" if v else "-"
        fmtb = lambda v: f"{min(x[0] for x in v):.3f} ({min(x[0] for x in v)/kh:.2f}x)" if v else "-"
        print(f"{ms[:24]:24s} {kh:6.3f}({nh:3d}) {fmt1(single):>20s} {fmt1(multi_m):>22s} "
              f"{fmtb(multi_b):>22s} {fmt(temp):>12s}")
        rows.append({"model": ms, "kappa_honest": kh, "n_honest": nh,
                     "single_layer": single, "multi_malign": multi_m, "multi_benign": multi_b,
                     "temperature": temp,
                     "eff_rank": geo.get(ms, {}).get("eff_rank"), "H": geo.get(ms, {}).get("H"),
                     "gain_rank1": geo.get(ms, {}).get("gain_rank1")})
        out["models"][ms] = rows[-1]

    print("\n  comparacao pareada por KL, no conjunto de TESTE (tokens nunca vistos pelo ataque)")
    print("  (por (modelo, orcamento) fica o MELHOR esforco do atacante entre as variantes:")
    print("   rank 16, rank completo, com e sem acesso a escala de saida)")
    print(f"    {'modelo':22s} {'KL':>7s} {'modo':8s} {'kappa':>7s} {'honesto no mesmo KL':>20s} {'razao':>7s}")
    # melhor variante por (modelo, orcamento): o atacante escolhe seu melhor ataque
    best_var: dict = {}
    for r in rows:
        for mode, key in (("malign", "multi_malign"), ("benign", "multi_benign")):
            for x in r[key]:
                if len(x) < 6:
                    continue
                k = (r["model"], mode, x[6])   # agrupa pelo ORCAMENTO, nao pelo KL obtido
                cur = best_var.get(k)
                better = cur is None or (x[5] > cur[5] if mode == "malign" else x[5] < cur[5])
                if better:
                    best_var[k] = x
    paired = {"malign": [], "benign": []}
    by_corpus: dict = {}
    for (model, mode, b), x in sorted(best_var.items()):
        kh, nh2 = honest_near_kl(model, x[1])
        if math.isnan(kh):
            continue
        c = corpus_of(model)
        by_corpus.setdefault(c, {"malign": [], "benign": [], "points": []})
        by_corpus[c][mode].append(x[5] / kh)
        by_corpus[c]["points"].append({"model": model_of(model), "mode": mode, "budget": b,
                                       "kl": x[1], "ratio": x[5] / kh})
        if c != "mix":
            continue   # o agregado historico e o do corpus do paper; os outros saem em corpus_summary
        paired[mode].append(x[5] / kh)
        print(f"    {model[:22]:22s} {x[1]:7.4f} {mode:8s} {x[5]:7.3f} {kh:14.3f} (n={nh2}) "
              f"{x[5]/kh:7.2f}x")
    for mode, v in paired.items():
        if v:
            se = np.std(v, ddof=1) / math.sqrt(len(v)) if len(v) > 1 else 0.0
            print(f"    {mode:8s}: {np.mean(v):.2f}x  IC95 [{np.mean(v) - 1.96*se:.2f}, "
                  f"{np.mean(v) + 1.96*se:.2f}]  min {min(v):.2f} max {max(v):.2f}  (n={len(v)})")
    out["paired_by_kl"] = paired          # so o corpus do paper (mix): e o numero publicado
    out["paired_by_corpus"] = by_corpus

    print("\n  comparacao no MESMO conjunto de teste (tokens que o ataque nunca viu):")
    tt_m, tt_b = [], []
    for r in rows:
        if not r["multi_malign"]:
            continue
        kht, nt = honest_test(r["model"])
        if math.isnan(kht):
            continue
        am = max(x[5] for x in r["multi_malign"] if len(x) > 5) if any(len(x) > 5 for x in r["multi_malign"]) else None
        ab = min(x[5] for x in r["multi_benign"] if len(x) > 5) if any(len(x) > 5 for x in r["multi_benign"]) else None
        if am:
            tt_m.append(am / kht)
        if ab:
            tt_b.append(ab / kht)
        r["kappa_honest_test"] = kht
        print(f"    {r['model'][:24]:24s} honesto(teste) {kht:.3f} (n={nt}) | maligno(teste) "
              f"{am if am else float('nan'):.3f} = {am/kht if am else float('nan'):.2f}x | "
              f"benigno(teste) {ab if ab else float('nan'):.3f} = {ab/kht if ab else float('nan'):.2f}x")
    if tt_m:
        print(f"    {'agregado':24s} maligno {np.mean(tt_m):.2f}x [{min(tt_m):.2f}, {max(tt_m):.2f}] | "
              f"benigno {np.mean(tt_b):.2f}x [{min(tt_b):.2f}, {max(tt_b):.2f}] (n={len(tt_m)})")
        out["test_set_comparison"] = {"malign": tt_m, "benign": tt_b}

    mm = [max(x[0] for x in r["multi_malign"]) / r["kappa_honest"] for r in rows if r["multi_malign"]]
    ss = [max(x[0] for x in r["single_layer"]) / r["kappa_honest"] for r in rows if r["single_layer"]]
    bb = [min(x[0] for x in r["multi_benign"]) / r["kappa_honest"] for r in rows if r["multi_benign"]]
    tt = [min(r["temperature"]) / r["kappa_honest"] for r in rows if r["temperature"]]
    cf = [x[3] for r in rows for x in r["multi_malign"]]
    ini = [x[4] / r["kappa_honest"] for r in rows for x in r["multi_malign"] if x[4]]
    print()
    for label, v in (("delta aleatorio de baixo rank (sem otimizar)", ini),
                     ("ataque de 1 camada, rank completo", ss),
                     ("ataque multicamada maligno (melhor de 3 orcamentos)", mm),
                     ("ataque multicamada BENIGNO (controle positivo)", bb),
                     ("temperatura (referencia analitica)", tt)):
        if v:
            print(f"  {label:52s}: kappa/kappa_honesto {np.mean(v):.2f}x  "
                  f"[{min(v):.2f}, {max(v):.2f}]  (n={len(v)})")
    if cf:
        print(f"  {'fracao do teto do oraculo, ataque maligno':52s}: {np.mean(cf):.3f} "
              f"[{min(cf):.3f}, {max(cf):.3f}]")
    print("\n  fator de anisotropia A = E|dgap| / sqrt(2 KL)  (isotropico ~ 1; honestos ~ 3):")
    for r in rows:
        an = anisotropy(r["model"])
        if not an or not an["attack"]:
            continue
        hv = [x["A"] for x in an["honest"].values()]
        av = {k: x["A"] for k, x in an["attack"].items()}
        r["anisotropy"] = an
        out["models"][r["model"]]["anisotropy"] = an
        print(f"    {r['model'][:24]:24s} honestos A {np.median(hv):.2f} [{min(hv):.2f},{max(hv):.2f}] "
              f"(n={len(hv)}) | ataque " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(av.items())))
        ts_h = np.median([x["tie_share_of_flips"] for x in an["honest"].values()])
        print(f"    {'':24s} flips vindos de empates bf16: honestos {100*ts_h:.1f}% | ataque " +
              ", ".join(f"{k} {100*x['tie_share_of_flips']:.1f}%"
                        for k, x in sorted(an["attack"].items())))
        fn_h = np.median([x["frac_dgap_negative"] for x in an["honest"].values()])
        print(f"    {'':24s} fracao de dgap que FECHA o gap (50% = sinal aleatorio): "
              f"honestos {100*fn_h:.1f}% | ataque " +
              ", ".join(f"{k} {100*x['frac_dgap_negative']:.1f}%"
                        for k, x in sorted(an["attack"].items())))
    # --- decomposicao da folga de 5x ate o teto do oraculo
    # 1a ordem: flips = rho_c(0) * E|dgap| / 2. O fator 1/2 supoe sinal simetrico, logo
    # um ataque com sinal PERFEITAMENTE alinhado vale no maximo 2x a taxa honesta no
    # mesmo KL. O resto da folga exige escolher o KL token a token, o que uma
    # perturbacao FIXA de pesos nao pode fazer por construcao.
    fd_all = json.loads((AN / "flipdirs.json").read_text())
    law = json.loads((AN / "law.json").read_text())
    fr = []
    for mid, v in fd_all.items():
        ms = mid.split("/")[-1]
        pts = [r for r in law["rows"] if r["ref"] == ms and r["model"] == ms
               and r["family"] != "outro modelo"]
        for b, ceil in v["oracle_ceiling"].items():
            near_p = sorted(pts, key=lambda r: abs(r["kl"] - float(b)))[:3]
            if near_p and abs(near_p[0]["kl"] - float(b)) < 0.5 * float(b):
                fr.append(float(np.mean([r["flip"] for r in near_p])) / ceil)
    if fr:
        f0 = float(np.mean(fr))
        print(f"\n  decomposicao da folga ate o teto do oraculo (media de {len(fr)} pontos):")
        print(f"    compressores honestos                    : {f0:.3f} do teto  (folga {1/f0:.1f}x)")
        print(f"    limite de 1a ordem com sinal alinhado    : {min(1.0, 2*f0):.3f} do teto  "
              f"(acessivel a uma perturbacao fixa: no maximo 2x)")
        if mm:
            print(f"    melhor ataque medido                     : {f0*np.mean(mm):.3f} do teto  "
                  f"({np.mean(mm):.2f}x honesto)")
        print(f"    resto, que exige alocar KL token a token : {1/(2*f0):.1f}x  "
              f"(impossivel para perturbacao fixa)")
        out["ceiling_decomposition"] = {"honest_fraction": f0, "sign_aligned_bound": min(1.0, 2 * f0),
                                        "allocation_factor": 1 / (2 * f0), "n": len(fr)}
    out["fp32_check"] = fp32_check()
    out["seed_check"] = seed_check()
    out["corpus_summary"] = corpus_summary(by_corpus, out["fp32_check"], out["seed_check"])
    er = [(r["model"], r["eff_rank"], r["H"], r["gain_rank1"]) for r in rows if r["eff_rank"]]
    if er:
        print("\n  geometria que explica a assimetria (direcoes de flip u_t = W_U[i1] - W_U[i2]):")
        for m, e, h, g in er:
            print(f"    {m[:24]:24s} rank efetivo {e:6.1f} de {h:5d} | ganho de um ataque rank-1 {g:.1f}x")
    (AN / "adversarial.json").write_text(json.dumps(out, indent=1))
    print(f"\n-> {AN / 'adversarial.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
