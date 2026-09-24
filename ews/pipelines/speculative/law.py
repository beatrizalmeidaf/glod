#!/usr/bin/env python3
"""C7/P3: a lei como PREVISOR da decodificacao especulativa.

Pergunta do paper: da para dizer quantos tokens um draft vai emplacar SEM
instanciar o sistema especulativo, so com dois escalares medidos em teacher
forcing (KL e TV por token) contra o alvo?

Tres previsores, do mais fraco (usa somente a lei) ao mais forte (usa a sequencia):
  L  lei    : a_g = 1 - kappa*sqrt(KL)   (kappa do ALVO, dos compressores honestos)
              a_s = 1 - TV               (identidade de Leviathan, exata)
  I  iid    : a_g = 1 - flip (concordancia medida em TF), rodadas i.i.d.
  S  seq    : simula as rodadas sobre a sequencia REAL de concordancia do TF
              (captura rajadas e a diluicao dos tokens propostos depois da
              primeira rejeicao, que o modelo i.i.d. ignora)

Comparacoes feitas em cima das MESMAS grandezas medidas: taxa de aceitacao por
token proposto (accepted/proposed, que inclui a diluicao) e tokens por rodada.
Tudo restrito ao dominio dos prompts usados na decodificacao (GSM8K).

    python ews_spec_law.py --k 4
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from ews.paths import OUT as ROOT
AN = ROOT / "analysis"


def load(p: Path):
    return torch.load(p, map_location="cpu", mmap=True, weights_only=False)


def kappa_honest(model_slug: str) -> tuple[float, int]:
    law = json.loads((AN / "law.json").read_text())
    ks = [r["flip"] / math.sqrt(r["kl"]) for r in law["rows"]
          if r["ref"] == model_slug and r["model"] == model_slug
          and r.get("family") != "outro modelo" and 1e-3 < r["kl"] < 0.25]
    return (float(np.median(ks)) if ks else float("nan")), len(ks)


def simulate(agree: list[bool], lens: list[int], k: int) -> dict:
    """Rodadas de especulativa sobre a sequencia de concordancia do teacher forcing."""
    emitted = rounds = accepted = proposed = 0
    off = 0
    for n in lens:
        seq = agree[off:off + n]
        off += n
        t = 0
        while t < n:
            acc = 0
            while acc < min(k, n - t) and seq[t + acc]:
                acc += 1
            step = min(acc + 1, n - t)
            emitted += step
            accepted += acc
            proposed += k          # o alvo verifica k propostas por rodada, aceitas ou nao
            rounds += 1
            t += step
    return {"tokens_per_round": emitted / max(rounds, 1),
            "accept_rate": accepted / max(proposed, 1)}


def iid(a: float, k: int) -> dict:
    tpr = (1 - a ** (k + 1)) / (1 - a) if a < 1 else k + 1
    # accepted/proposed com k propostas por rodada: E[aceitos] = sum_{j=1..k} a^j
    acc = sum(a ** j for j in range(1, k + 1))
    return {"tokens_per_round": tpr, "accept_rate": acc / k}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=int, default=4)
    args = p.parse_args(argv)
    rows = []
    for d in sorted((ROOT / "closedloop").iterdir()):
        rs = d.name
        ref_p = ROOT / rs / rs / "bf16.pt"
        if not ref_p.exists():
            continue
        ref = load(ref_p)
        corpus = json.loads((ROOT / "corpora" / f"{rs}.json").read_text())
        gsm = [r["source"] == "gsm8k" for r in corpus["records"]]
        mask = torch.tensor([g for r, gl in zip(gsm, corpus["gen_ids"]) for g in [r] * len(gl)])
        lens = [len(g) for g, ok in zip(corpus["gen_ids"], gsm) if ok]
        kap, nk = kappa_honest(rs)
        for f in sorted(d.glob("spec_*.json")):
            j = json.loads(f.read_text())
            name = f.stem[len("spec_"):]
            dm = name.split("_")[0] if "_" in name and name.startswith(("Qwen", "gemma", "OLMo", "Mistral", "Phi")) else rs
            cfg = name.split("_")[-1]
            # o draft nao comprimido foi gravado como raw.pt para drafts de outro modelo
            # e como bf16.pt para o proprio alvo; aceitar os dois nomes
            cands = ([ROOT / rs / dm / "raw.pt", ROOT / rs / dm / "bf16.pt"]
                     if cfg == "raw" else [ROOT / rs / dm / f"{cfg}.pt"])
            tf = next((c for c in cands if c.exists()), None)
            if tf is None:
                print(f"  (sem scoring TF para {rs} <- {name}: {[c.name for c in cands]})")
                continue
            c = load(tf)
            agree = (c["top1"] == ref["top1"])[mask].tolist()
            kl = c["kl"][mask].double().mean().item()
            tv = c["tv"][mask].double().mean().item()
            flip = 1 - float(np.mean(agree))
            a_law = max(0.0, 1 - kap * math.sqrt(kl))
            row = {"model": rs, "draft": name, "draft_model": dm,
                   "cross_model": dm != rs, "draft_config": cfg,
                   "kl": kl, "tv": tv, "flip_tf": flip,
                   "kappa_target": kap, "n_kappa": nk,
                   "meas_greedy_accept": j["greedy"]["accept_rate"],
                   "meas_greedy_tpr": j["greedy"]["tokens_per_round"],
                   "meas_sample_accept": j["sample"]["accept_rate"],
                   "meas_sample_tpr": j["sample"]["tokens_per_round"],
                   "meas_sample_accept_prob": j["sample"]["mean_accept_prob"],
                   "L_greedy": iid(a_law, args.k), "I_greedy": iid(1 - flip, args.k),
                   "S_greedy": simulate(agree, lens, args.k),
                   "L_sample": iid(1 - tv, args.k)}
            rows.append(row)
    if not rows:
        print("nenhum resultado de especulativa encontrado")
        return 1

    def stats(pred, meas):
        pred, meas = np.array(pred), np.array(meas)
        r2 = 1 - ((meas - pred) ** 2).sum() / ((meas - meas.mean()) ** 2).sum()
        return r2, float(np.abs(pred - meas).mean()), float(np.abs(pred / meas - 1).mean())

    print(f"n = {len(rows)} pares (alvo, draft) em {len({r['model'] for r in rows})} alvos, k = {args.k}\n")
    print(f"{'alvo':22s} {'draft':22s} {'KL':>7s} {'TV':>6s} | greedy aceita: "
          f"{'medido':>7s} {'lei':>7s} {'iid':>7s} {'seq':>7s} | tokens/rodada: {'medido':>7s} {'lei':>7s} {'seq':>7s}")
    for r in sorted(rows, key=lambda r: (r["model"], r["kl"])):
        print(f"{r['model'][:22]:22s} {r['draft'][:22]:22s} {r['kl']:7.4f} {r['tv']:6.3f} | "
              f"{r['meas_greedy_accept']:19.3f} {r['L_greedy']['accept_rate']:7.3f} "
              f"{r['I_greedy']['accept_rate']:7.3f} {r['S_greedy']['accept_rate']:7.3f} | "
              f"{r['meas_greedy_tpr']:22.3f} {r['L_greedy']['tokens_per_round']:7.3f} "
              f"{r['S_greedy']['tokens_per_round']:7.3f}")
    print()
    out = {"k": args.k, "rows": rows, "fits": {}}
    for label, key, getp in (
        ("greedy aceita/proposta   lei (so KL + kappa)", "meas_greedy_accept", lambda r: r["L_greedy"]["accept_rate"]),
        ("greedy aceita/proposta   iid (flip medido)  ", "meas_greedy_accept", lambda r: r["I_greedy"]["accept_rate"]),
        ("greedy aceita/proposta   seq (sequencia TF) ", "meas_greedy_accept", lambda r: r["S_greedy"]["accept_rate"]),
        ("greedy tokens/rodada     lei (so KL + kappa)", "meas_greedy_tpr", lambda r: r["L_greedy"]["tokens_per_round"]),
        ("greedy tokens/rodada     seq (sequencia TF) ", "meas_greedy_tpr", lambda r: r["S_greedy"]["tokens_per_round"]),
        ("amostra aceita/proposta  1-TV (identidade)  ", "meas_sample_accept", lambda r: r["L_sample"]["accept_rate"]),
        ("amostra E[min(1,p/q)]    1-TV (identidade)  ", "meas_sample_accept_prob", lambda r: 1 - r["tv"]),
        ("amostra tokens/rodada    1-TV (identidade)  ", "meas_sample_tpr", lambda r: r["L_sample"]["tokens_per_round"]),
    ):
        ok = [r for r in rows if not math.isnan(getp(r)) and r[key] is not None]
        r2, mae, mape = stats([getp(r) for r in ok], [r[key] for r in ok])
        out["fits"][label.strip()] = {"r2": r2, "mae": mae, "mape": mape, "n": len(ok)}
        print(f"  {label}: R2 {r2:6.3f} | MAE {mae:.4f} | erro relativo medio {100*mape:5.2f}%  (n={len(ok)})")
    (AN / "spec_law.json").write_text(json.dumps(out, indent=1))
    print(f"\n-> {AN / 'spec_law.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def decision_table(k: int = 4) -> None:
    """A lei ESCOLHE o draft certo? Previsao (so KL + custo de bytes) x relogio medido.

    Chamado por: python -c "import ews.pipelines.speculative.law as S; S.decision_table()"
    """
    from scipy.stats import spearmanr
    for d in sorted((ROOT / "spec_bench").iterdir()):
        f = d / "results.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        rows = [(n, v) for n, v in r["drafts"].items() if "skipped" not in v]
        if len(rows) < 3:
            continue
        print(f"\n=== alvo {r['target']} ({r['target_tokens_per_s']:.1f} tok/s sozinho, k={k})")
        print(f"  {'draft':28s} {'KL':>7s} {'tpr prev':>9s} {'tpr med':>8s} {'custo':>6s} "
              f"{'speedup prev':>12s} {'speedup med':>11s}")
        pred, meas = [], []
        for n, v in sorted(rows, key=lambda x: -x[1]["prediction"]["pred_tokens_per_round_greedy"]):
            p = v["prediction"]
            c = v["draft_bytes_ratio"]
            sp_pred = p["pred_tokens_per_round_greedy"] / (k * c + 1)
            pred.append(sp_pred)
            meas.append(v["greedy"]["speedup_wallclock"])
            print(f"  {n[:28]:28s} {p['kl']:7.4f} {p['pred_tokens_per_round_greedy']:9.3f} "
                  f"{v['greedy']['tokens_per_round']:8.3f} {c:6.2f} {sp_pred:12.2f} "
                  f"{v['greedy']['speedup_wallclock']:11.2f}x")
        best_pred = max(range(len(pred)), key=lambda i: pred[i])
        best_meas = max(range(len(meas)), key=lambda i: meas[i])
        print(f"  Spearman(previsto, medido) = {spearmanr(pred, meas)[0]:+.2f} | "
              f"escolha da lei = escolha certa? {'SIM' if best_pred == best_meas else 'NAO'} "
              f"(perda {100*(meas[best_meas]-meas[best_pred])/meas[best_meas]:.1f}% do melhor)")
