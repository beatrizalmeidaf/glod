#!/usr/bin/env python3
"""Concordancia entre juizes (LLM x LLM, e humano x LLM quando houver anotacao).

    python3 scripts/judge_agreement.py                       # Qwen x Mistral nos 800 flips
    python3 scripts/judge_agreement.py --human human_labels.csv   # + humano nos 100 itens cegos

Rotulo binario: 'different' quando p_different > 0.5 (media das duas ordens de
apresentacao), como no paper. 'unsure' do humano fica fora do kappa.
"""
import argparse, csv, json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AN = Path(os.environ.get("GLOD_RESULTS", "/local/user_beatrizalmeida/ews_results/fid")) / "analysis"


def kappa(a, b):
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0, po, n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--human"); args = ap.parse_args()
    q = json.load(open(AN / "semantic_judge.json")); m = json.load(open(AN / "semantic_judge_mistral.json"))
    out = {}
    a, b = [], []
    for kind in ("mag", "wanda"):
        for x, y in zip(q["items"][kind], m["items"][kind]):
            a.append(x["p_different"] > 0.5); b.append(y["p_different"] > 0.5)
    k, po, n = kappa(a, b)
    out["qwen_vs_mistral"] = {"kappa": k, "agreement": po, "n": n}
    for name, J in (("qwen", q), ("mistral", m)):
        out[name] = {kind: sum(x["p_different"] > 0.5 for x in J["items"][kind]) / len(J["items"][kind]) for kind in ("mag", "wanda")}
    if args.human:
        key = {r["id"]: r for r in json.load(open(ROOT / "results/human_eval/key_DO_NOT_OPEN_BEFORE_ANNOTATING.json"))}
        hum = {r["id"]: r["label"] for r in csv.DictReader(open(args.human)) if r["label"] in ("same", "different")}
        for name, J in (("qwen", q), ("mistral", m)):
            h, jj = [], []
            for i, lab in hum.items():
                r = key[i]; h.append(lab == "different"); jj.append(J["items"][r["kind"]][r["k"]]["p_different"] > 0.5)
            kk, po, n = kappa(h, jj); out[f"human_vs_{name}"] = {"kappa": kk, "agreement": po, "n": n}
        # proporcao "muda o conteudo" por metodo nos mesmos itens, humano e juizes, e a
        # diferenca magnitude - Wanda com bootstrap (itens reamostrados dentro de cada metodo)
        import random
        rng = random.Random(0)
        by = {"mag": [], "wanda": []}
        for i, lab in hum.items():
            r = key[i]
            by[r["kind"]].append((lab == "different", q["items"][r["kind"]][r["k"]]["p_different"] > 0.5,
                                  m["items"][r["kind"]][r["k"]]["p_different"] > 0.5))
        sub = {}
        for j, name in enumerate(("human", "qwen", "mistral")):
            sh = {kind: sum(x[j] for x in v) / len(v) for kind, v in by.items()}
            boots = []
            for _ in range(10000):
                bm = [rng.choice(by["mag"])[j] for _ in by["mag"]]; bw = [rng.choice(by["wanda"])[j] for _ in by["wanda"]]
                boots.append(sum(bm) / len(bm) - sum(bw) / len(bw))
            boots.sort()
            allv = [x[j] for v in by.values() for x in v]
            sub[name] = {**sh, "all": sum(allv) / len(allv), "diff": sh["mag"] - sh["wanda"], "lo": boots[250], "hi": boots[9749]}
        out["human_subset"] = {"n": {k: len(v) for k, v in by.items()}, "n_unsure": sum(1 for r in csv.DictReader(open(args.human)) if r["label"] == "unsure"), **sub}
    print(json.dumps(out, indent=1))
    (AN / "judge_agreement.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
