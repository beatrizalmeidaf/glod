#!/usr/bin/env python3
"""D2: buscas de compressao selecionadas em MMLU bruto x invariante a permutacao.

Cada configuracao e pontuada nas 4 permutacoes ciclicas das alternativas
(logits das letras, 0-shot ou 5-shot). Metricas:
  raw        acuracia na ordem original (o que a literatura reporta)
  inv        argmax da media dos log-probs por CONTEUDO nas 4 permutacoes
  pc         correcao de prior em vista unica (prior estimado sem rotulo em metade dos itens)
  nll        NLL do gabarito na ordem original (criterio de perfis de camada)
  prior      distribuicao media sobre as letras

Buscas (no split de calibracao):
  loo        remove cada camada isoladamente; ranqueia por raw, nll e inv;
             remove as k melhores de cada ranking
  mixed      N mapas aleatorios (25% das camadas em 3 bits, resto 4 bits);
             escolhe o melhor por raw e o melhor por inv
Avaliacao final no split de teste (disjunto) com as mesmas metricas.

    python ews_d2_search.py --model Qwen/Qwen3-14B --device cuda:1
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time

import torch

from ews.core import compressors as C
from ews.core.elastic_depth import make_elastic_depth
from ews.core.model_loader import load_model
from ews.core.scoring import score_choices
from ews.paths import CACHE_DIR as DEFAULT_CACHE_DIR
from ews.paths import OUT, slug
from ews.paths import MMLU_PT_CSV

LOGGER = logging.getLogger("ews.d2")
LETTERS = "ABCD"


def load_items(dataset: str, split: str, n: int, seed: int, cache_dir: str) -> list[dict]:
    if dataset == "mmlu_en":
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", "all", cache_dir=f"{cache_dir}/datasets")[split]
        items = [{"q": r["question"], "opts": list(r["choices"]), "gold": int(r["answer"]),
                  "subject": r["subject"]} for r in ds]
    else:
        import pandas as pd
        df = pd.read_csv(MMLU_PT_CSV).dropna(subset=["Question", "A", "B", "C", "D", "Answer"])
        df = df[df["Answer"].astype(str).str.strip().str.upper().isin(list(LETTERS))]
        df = df.sample(frac=1.0, random_state=7)
        half = len(df) // 2
        df = df.iloc[:half] if split == "validation" else df.iloc[half:]
        items = [{"q": r["Question"], "opts": [r[c] for c in LETTERS], "gold": LETTERS.index(str(r["Answer"]).strip().upper()),
                  "subject": r["Subject"]} for _, r in df.iterrows()]
    items = [it for it in items if len(it["opts"]) == 4]
    random.Random(seed).shuffle(items)
    return items[:n]


def fmt(it: dict, shift: int, lang: str) -> str:
    shown = [None] * 4
    for j in range(4):
        shown[(j + shift) % 4] = it["opts"][j]
    if lang == "en":
        body = f"Question: {it['q']}\n" + "".join(f"{LETTERS[i]}. {shown[i]}\n" for i in range(4)) + "Answer:"
    else:
        body = f"Pergunta: {it['q']}\n" + "".join(f"{LETTERS[i]}) {shown[i]}\n" for i in range(4)) + "Resposta:"
    return body


def build_prompts(items, shots, lang):
    prefix = ""
    if shots:
        blocks = [fmt(s, 0, lang) + " " + LETTERS[s["gold"]] for s in shots]
        prefix = "\n\n".join(blocks) + "\n\n"
    return {s: [prefix + fmt(it, s, lang) for it in items] for s in range(4)}


def choice_ids(tok):
    ids = []
    for c in LETTERS:
        enc = tok.encode(" " + c, add_special_tokens=False)
        ids.append(enc[-1])
    assert len(set(ids)) == 4, ids
    return ids


def evaluate(loaded, prompts, gold, cids, batch) -> dict:
    L = torch.stack([score_choices(loaded, prompts[s], cids, batch_size=batch, show_progress=False)[0]
                     for s in range(4)]).double()                        # [4, N, 4] ordem de letra
    idx = torch.tensor([[(j + s) % 4 for j in range(4)] for s in range(4)])
    Cn = torch.stack([L[s][:, idx[s]] for s in range(4)]).log_softmax(-1)  # por conteudo
    g = torch.tensor(gold)
    raw = (Cn[0].argmax(-1) == g)
    inv = (Cn.mean(0).argmax(-1) == g)
    N = len(gold)
    half = torch.arange(N) % 2 == 0
    lp0 = L[0].log_softmax(-1)
    pc = torch.empty(N, dtype=torch.bool)
    for fit, app in ((half, ~half), (~half, half)):
        prior = L[:, fit].softmax(-1).mean((0, 1)).log()
        pc[app] = (lp0[app] - prior).argmax(-1) == g[app]
    nll = -Cn[0].gather(-1, g[:, None]).squeeze(-1)
    return {"raw": raw, "inv": inv, "pc": pc, "nll": nll, "prior": L.softmax(-1).mean((0, 1)),
            "pred_raw": Cn[0].argmax(-1), "pred_inv": Cn.mean(0).argmax(-1)}


def summary(r: dict) -> dict:
    return {"raw": r["raw"].double().mean().item(), "inv": r["inv"].double().mean().item(),
            "pc": r["pc"].double().mean().item(), "nll": r["nll"].mean().item(),
            "prior": [round(x, 4) for x in r["prior"].tolist()]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--dataset", default="mmlu_en", choices=["mmlu_en", "mmlu_pt"])
    p.add_argument("--shots", type=int, default=0)
    p.add_argument("--n-calib", type=int, default=400)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--n-mixed", type=int, default=40)
    p.add_argument("--k-skip", type=int, nargs="+", default=[2, 4])
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    torch.set_grad_enabled(False)
    tag = f"{slug(args.model)}__{args.dataset}_{args.shots}shot"
    out_path = OUT / "d2" / f"{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(out_path.read_text()) if out_path.exists() else {"calib": {}, "test": {}}

    lang = "en" if args.dataset == "mmlu_en" else "pt"
    calib_items = load_items(args.dataset, "validation", args.n_calib, 0, args.cache_dir)
    test_items = load_items(args.dataset, "test", args.n_test, 1, args.cache_dir)
    shots = load_items(args.dataset, "dev" if lang == "en" else "validation", 5, 2, args.cache_dir)[:args.shots] \
        if args.shots else []
    loaded = load_model(args.model, role="cfg", device=args.device, dtype="bfloat16", cache_dir=args.cache_dir)
    bank = C.WeightBank(loaded.decoder)
    ctrl = make_elastic_depth(loaded.model)
    cids = choice_ids(loaded.tokenizer)
    P_cal = build_prompts(calib_items, shots, lang)
    P_test = build_prompts(test_items, shots, lang)
    g_cal = [it["gold"] for it in calib_items]
    g_test = [it["gold"] for it in test_items]
    nl = bank.num_layers
    dirty = {"w": False}  # restaurar 20+ GiB da CPU a cada config de skip domina o LOO

    def run(cfg: dict, split: str) -> dict:
        key = json.dumps(cfg, sort_keys=True)
        if key in state[split]:
            return state[split][key]
        if dirty["w"]:
            bank.restore()
            dirty["w"] = False
        bits = cfg.get("bits", {})
        if bits:
            dirty["w"] = True
            groups: dict[int, list[int]] = {}
            for layer, b in bits.items():
                groups.setdefault(int(b), []).append(int(layer))
            for b, layers in groups.items():
                bank.map(lambda w, t, b=b: C.rtn(w, b), layers=layers)
        with ctrl.skipping(cfg.get("skip", [])):
            r = evaluate(loaded, P_cal if split == "calib" else P_test, g_cal if split == "calib" else g_test,
                         cids, args.batch)
        s = summary(r)
        if split == "test":
            s["raw_vec"] = r["raw"].int().tolist(); s["inv_vec"] = r["inv"].int().tolist()
            s["pc_vec"] = r["pc"].int().tolist()
            s["pred_raw"] = r["pred_raw"].tolist(); s["pred_inv"] = r["pred_inv"].tolist()
        state[split][key] = s
        out_path.write_text(json.dumps(state))
        return s

    t0 = time.time()
    base = run({}, "calib")
    LOGGER.info("bf16 calib: %s", base)
    # ---- LOO de camadas
    loo = {l: run({"skip": [l]}, "calib") for l in range(nl)}
    LOGGER.info("LOO pronto (%.0fs)", time.time() - t0)
    picks = {}
    for k in args.k_skip:
        for crit, keyf in (("raw", lambda l: -loo[l]["raw"]), ("nll", lambda l: loo[l]["nll"]),
                           ("inv", lambda l: -loo[l]["inv"])):
            picks[f"skip{k}_by_{crit}"] = {"skip": sorted(sorted(range(1, nl), key=keyf)[:k])}
    # ---- precisao mista aleatoria
    rng = random.Random(0)
    mixed = []
    for i in range(args.n_mixed):
        low = rng.sample(range(nl), nl // 4)
        cfg = {"bits": {str(l): (3 if l in low else 4) for l in range(nl)}}
        mixed.append((cfg, run(cfg, "calib")))
    u4 = {"bits": {str(l): 4 for l in range(nl)}}
    run(u4, "calib")
    best_raw = max(mixed, key=lambda x: x[1]["raw"])[0]
    best_inv = max(mixed, key=lambda x: x[1]["inv"])[0]
    picks.update({"mixed_by_raw": best_raw, "mixed_by_inv": best_inv, "u4": u4, "bf16": {}})
    LOGGER.info("buscas prontas (%.0fs); avaliando %d configs no teste", time.time() - t0, len(picks))
    state["picks"] = picks
    for name, cfg in picks.items():
        s = run(cfg, "test")
        LOGGER.info("[test] %-16s raw %.4f inv %.4f pc %.4f prior %s", name, s["raw"], s["inv"], s["pc"], s["prior"])
    out_path.write_text(json.dumps(state))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
