#!/usr/bin/env python3
"""Por que magnitude e Wanda, com a mesma taxa de flips, diferem em 38 pontos no GSM8K?

A taxonomia por regra (flip-kinds) nao separa um flip inocuo ("that" -> "which") de
um fatal ("5" -> "3", "Sarah" -> "Alex"). Aqui um juiz LLM decide, em circuito
fechado, se o PRIMEIRO ponto em que a geracao comprimida se afasta da densa muda o
conteudo matematico do raciocinio.

  gen    regera os 400 problemas do `crack part2` na Gemma-3-4B: densa, magnitude e
         Wanda nos MESMOS controles do part2.json (KL 0.10), agora guardando o texto.
         Confere que as acuracias reproduzem as do part2 antes de seguir.
  judge  para cada problema que diverge, mostra ao juiz o enunciado, o prefixo comum
         e as duas continuacoes (ate a proxima quebra de linha, no maximo 40 tokens)
         e pergunta se dizem a mesma coisa matematicamente. Cada par e julgado nas
         duas ordens (A/B e B/A); o veredito e a media das duas probabilidades e a
         concordancia entre ordens e reportada como consistencia do juiz.

O juiz padrao e Qwen2.5-72B-Instruct, local; ele nao e da familia do modelo julgado.

ATENCAO: o desenho de circuito fechado (gen + judge) e confundido. A primeira
divergencia cai quase sempre no token 0-2, na frase de abertura ("Here's how to
solve..." x "Let $g$ be..."), e depois dela as trajetorias nao tem mais prefixo
comum. O veredito mede o estilo de abertura, nao erro matematico. O desenho valido e:

  flips     teacher forcing no corpus de referencia (metade GSM8K): amostra flips
            fora de empates bf16, com o MESMO prefixo nos dois modelos, e deixa o
            modelo comprimido continuar alguns tokens a partir do proprio token
            (a continuacao da referencia e o proprio corpus).
  judge-tf  o juiz compara a continuacao da referencia com a do modelo comprimido.

    python -m glod semantic-judge gen   --device cuda:0
    python -m glod semantic-judge judge --judge-gpus 0 1 2
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys

import numpy as np
import torch

from glod.paths import CACHE_DIR as DEFAULT_CACHE_DIR
from glod.paths import OUT, slug

LOGGER = logging.getLogger("glod.semantic_judge")
METHODS = ("mag", "wanda")
QUESTION = re.compile(r"user\n(.*?)<end_of_turn>", re.S)


def question_text(prompt: str) -> str:
    """Enunciado sem o template do chat: da instrucao ate o primeiro marcador de fim de
    turno. Na Gemma coincide com o grupo de QUESTION; serve tambem a Qwen, OLMo, Phi, Llama."""
    if (m := QUESTION.search(prompt)):
        return m.group(1).strip()
    i = prompt.find("Solve the following problem")
    body = prompt[i:] if i >= 0 else prompt
    cuts = [k for k in (body.find(x) for x in ("<|", "<end_of_turn>", "[/INST]", "</s>")) if k > 0]
    return body[:min(cuts)].strip() if cuts else body.strip()


def out_dir(model: str):
    d = OUT / "semantic" / slug(model)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------ gen
def stage_gen(args) -> None:
    from glod.core import compressors as C
    from glod.corpora.token_oracle import generate_greedy
    from glod.pipelines.tasks.closedloop import gsm8k_disjoint, parse_answer
    from glod.pipelines.tasks.crack_gsm8k import setup

    part2 = json.loads((OUT / "crack" / slug(args.model) / "part2.json").read_text())
    corpus, _ref, loaded, bank, _ctrl, calib_fn = setup(args)
    used = {r["prompt"] for r in corpus.records}
    # mesma semente e mesmo filtro do crack part2: os mesmos 400 problemas
    prompts = gsm8k_disjoint(loaded.tokenizer, args.n_task, used, args.cache_dir, seed=31337)

    def run() -> dict:
        recs = [dict(r) for r in prompts]
        gen = generate_greedy(loaded, recs, max_new_tokens=args.max_new_tokens, batch_size=args.batch)
        ok = []
        for rec in gen.records:
            a = parse_answer(rec["completion"])
            ok.append(a is not None and abs(a - float(rec["gold"])) < 1e-6)
        return {"acc": sum(ok) / len(ok), "correct": ok, "gen_ids": gen.gen_ids,
                "completions": [r["completion"] for r in gen.records],
                "answers": [parse_answer(r["completion"]) for r in gen.records]}

    res = {"prompts": [r["prompt"] for r in prompts], "gold": [r["gold"] for r in prompts]}
    bank.restore()
    res["base"] = run()
    for kind in METHODS:
        key = f"{kind}|{args.target}"
        knob = part2[key]["knob"]
        bank.restore()
        with torch.no_grad():
            if kind == "mag":
                bank.map(lambda w_, t: C.magnitude_prune(w_, knob))
            else:
                C.calibrated_compress(loaded.model, bank, loaded.decoder.layers, calib_fn(), "wanda",
                                      sparsity=knob, batch_size=args.calib_batch)
        res[kind] = run() | {"knob": knob}
        LOGGER.info("[%s] acc %.4f (part2: %.4f)", kind, res[kind]["acc"], part2[key]["acc"])
    bank.restore()
    # reproducao: se a acuracia nao bate com a do part2, as geracoes nao sao as do paper
    res["reproduces_part2"] = {
        "base": abs(res["base"]["acc"] - part2["base"]["acc"]) < 0.0126,
        **{k: abs(res[k]["acc"] - part2[f"{k}|{args.target}"]["acc"]) < 0.0126 for k in METHODS}}
    (out_dir(args.model) / "gens.json").write_text(json.dumps(res))
    LOGGER.info("reproduz o part2: %s", res["reproduces_part2"])


# ------------------------------------------------------------------ judge
PROMPT = (
    "You compare two continuations of the same partial solution to a math word problem.\n\n"
    "Problem:\n{q}\n\nPartial solution so far:\n{prefix}\n\n"
    "Continuation A:\n{a}\n\nContinuation B:\n{b}\n\n"
    "Do A and B express the same mathematical content -- the same quantities, operations, "
    "entities and logical steps -- differing at most in wording or formatting? "
    "Answer with exactly one word: SAME or DIFFERENT.")


def divergence(ref: list[int], cmp: list[int], tok, max_tokens: int):
    n = min(len(ref), len(cmp))
    i = next((j for j in range(n) if ref[j] != cmp[j]), None)
    if i is None:
        if len(ref) == len(cmp):
            return None
        i = n

    def cont(ids):
        text = tok.decode(ids[i:i + max_tokens], skip_special_tokens=True)
        # ate a proxima quebra de linha que venha depois do token divergente
        k = text.find("\n", 1)
        return text if k < 0 else text[:k]
    return {"pos": i, "rel_pos": i / max(len(ref), 1),
            "prefix": tok.decode(ref[:i], skip_special_tokens=True),
            "ref": cont(ref), "cmp": cont(cmp)}


def load_judge(args):
    """Devolve p_different(q, prefix, a, b): P(DIFFERENT) contra SAME no primeiro token."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    jtok = AutoTokenizer.from_pretrained(args.judge, cache_dir=args.cache_dir)
    mem = {i: args.judge_mem for i in args.judge_gpus}
    judge = AutoModelForCausalLM.from_pretrained(args.judge, cache_dir=args.cache_dir, torch_dtype=torch.bfloat16,
                                                 device_map="auto", max_memory=mem).eval()
    ids_same = jtok.encode("SAME", add_special_tokens=False)[0]
    ids_diff = jtok.encode("DIFFERENT", add_special_tokens=False)[0]
    assert ids_same != ids_diff

    @torch.no_grad()
    def p_different(q, prefix, a, b) -> float:
        msgs = [{"role": "user", "content": PROMPT.format(q=q, prefix=prefix or "(empty)", a=a, b=b)}]
        text = jtok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = jtok(text, return_tensors="pt").to(judge.device)
        logits = judge(**enc).logits[0, -1].float()
        two = torch.stack([logits[ids_same], logits[ids_diff]]).softmax(0)
        return float(two[1])
    return p_different


def stage_judge(args) -> None:
    from transformers import AutoTokenizer

    g = json.loads((out_dir(args.model) / "gens.json").read_text())
    if not all(g["reproduces_part2"].values()):
        LOGGER.warning("as geracoes NAO reproduzem o part2: %s", g["reproduces_part2"])
    gtok = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    p_different = load_judge(args)

    out = {"judge": args.judge, "items": {}}
    for kind in METHODS:
        items = []
        for i, prompt in enumerate(g["prompts"]):
            d = divergence(g["base"]["gen_ids"][i], g[kind]["gen_ids"][i], gtok, args.cont_tokens)
            if d is None:
                continue
            m = QUESTION.search(prompt)
            q = m.group(1).strip() if m else prompt
            p_ab = p_different(q, d["prefix"], d["ref"], d["cmp"])
            p_ba = p_different(q, d["prefix"], d["cmp"], d["ref"])
            items.append({"i": i, **d, "p_ab": p_ab, "p_ba": p_ba, "p_different": 0.5 * (p_ab + p_ba),
                          "consistent": (p_ab > 0.5) == (p_ba > 0.5),
                          "base_correct": g["base"]["correct"][i], "cmp_correct": g[kind]["correct"][i],
                          "answer_changed": g["base"]["answers"][i] != g[kind]["answers"][i]})
        out["items"][kind] = items
        LOGGER.info("[%s] %d divergencias julgadas", kind, len(items))

    rng = np.random.default_rng(0)

    def summary(items, n_total):
        diff = np.array([x["p_different"] > 0.5 for x in items])
        chg = np.array([x["answer_changed"] for x in items])
        lost = np.array([x["base_correct"] and not x["cmp_correct"] for x in items])
        return {"n_problems": n_total, "n_diverged": len(items),
                "share_different": float(diff.mean()), "n_different": int(diff.sum()),
                "consistency": float(np.mean([x["consistent"] for x in items])),
                "answer_changed_if_different": float(chg[diff].mean()) if diff.any() else None,
                "answer_changed_if_same": float(chg[~diff].mean()) if (~diff).any() else None,
                "lost_if_different": float(lost[diff].mean()) if diff.any() else None,
                "lost_if_same": float(lost[~diff].mean()) if (~diff).any() else None,
                "different_per_problem": float(diff.sum() / n_total)}

    n = len(g["prompts"])
    out["summary"] = {k: summary(out["items"][k], n) for k in METHODS}
    # diferenca mag - wanda na fracao de problemas cuja primeira divergencia muda o conteudo,
    # por bootstrap sobre os problemas (a unidade pareada: o mesmo enunciado nos dois metodos)
    flag = {k: np.zeros(n) for k in METHODS}
    for k in METHODS:
        for x in out["items"][k]:
            flag[k][x["i"]] = x["p_different"] > 0.5
    boot = []
    for _ in range(4000):
        s = rng.integers(0, n, n)
        boot.append(flag["mag"][s].mean() - flag["wanda"][s].mean())
    out["diff_mag_minus_wanda"] = {"per_problem": float(flag["mag"].mean() - flag["wanda"].mean()),
                                   "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5))}
    (OUT / "analysis" / "semantic_judge_closedloop.json").write_text(json.dumps(out, indent=1))
    for k in METHODS:
        v = out["summary"][k]
        print(f"{k:6s} diverge {v['n_diverged']}/{n} | primeira divergencia muda o conteudo: "
              f"{v['n_different']} ({100 * v['share_different']:.1f}% das divergencias) | consistencia "
              f"{100 * v['consistency']:.1f}% | resposta muda: {v['answer_changed_if_different']} se DIFFERENT, "
              f"{v['answer_changed_if_same']} se SAME")
    d = out["diff_mag_minus_wanda"]
    print(f"mag - wanda, por problema: {d['per_problem']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]")


# ------------------------------------------------------------------ flips (teacher forcing)
def stage_flips(args) -> None:
    from glod.core import compressors as C
    from glod.core.fidelity import score_fidelity
    from glod.pipelines.tasks.crack_gsm8k import gsm_mask, setup

    part2 = json.loads((OUT / "crack" / slug(args.model) / "part2.json").read_text())
    corpus, ref, loaded, bank, _ctrl, calib_fn = setup(args)
    tok = loaded.tokenizer
    offs = np.cumsum([0] + [len(g) for g in corpus.gen_ids])
    seq_of = np.repeat(np.arange(len(corpus.gen_ids)), np.diff(offs))
    mask = gsm_mask(corpus)
    tie = ref["topv"][:, 0] == ref["topv"][:, 1]
    rng = np.random.default_rng(0)
    res = {}
    for kind in METHODS:
        key = f"{kind}|{args.target}"
        knob = part2[key]["knob"]
        bank.restore()
        with torch.no_grad():
            if kind == "mag":
                bank.map(lambda w_, t: C.magnitude_prune(w_, knob))
            else:
                C.calibrated_compress(loaded.model, bank, loaded.decoder.layers, calib_fn(), "wanda",
                                      sparsity=knob, batch_size=args.calib_batch)
        s = score_fidelity(loaded, corpus, ref=ref, batch_size=8)
        flip = (s["top1"] != ref["top1"]) & mask & ~tie
        n_flip = int(flip.sum())
        pos = rng.choice(flip.nonzero().flatten().numpy(), size=min(args.n_flips, n_flip), replace=False)
        # continuacao local do modelo comprimido a partir do mesmo prefixo, em lote
        items = []
        tok.padding_side = "left"
        for b0 in range(0, len(pos), args.batch):
            chunk = pos[b0:b0 + args.batch]
            seqs = []
            for p_ in chunk:
                i = int(seq_of[p_]); t = int(p_ - offs[i])
                seqs.append((i, t, corpus.prompt_ids[i] + corpus.gen_ids[i][:t]))
            width = max(len(x[2]) for x in seqs)
            ids = torch.full((len(seqs), width), tok.pad_token_id)
            att = torch.zeros((len(seqs), width), dtype=torch.long)
            for r, (_, _, x) in enumerate(seqs):
                ids[r, width - len(x):] = torch.tensor(x); att[r, width - len(x):] = 1
            with torch.no_grad():
                out = loaded.model.generate(input_ids=ids.to(loaded.device), attention_mask=att.to(loaded.device),
                                            max_new_tokens=args.cont_tokens, do_sample=False, top_p=None,
                                            top_k=None, pad_token_id=tok.pad_token_id)
            for r, (i, t, _) in enumerate(seqs):
                cmp_ids = out[r, width:].tolist()

                def cut(text):
                    k = text.find("\n", 1)
                    return text if k < 0 else text[:k]
                items.append({"seq": i, "t": t, "rel_pos": t / max(len(corpus.gen_ids[i]), 1),
                              "prefix": tok.decode(corpus.gen_ids[i][:t], skip_special_tokens=True),
                              "ref": cut(tok.decode(corpus.gen_ids[i][t:t + args.cont_tokens],
                                                    skip_special_tokens=True)),
                              "cmp": cut(tok.decode(cmp_ids, skip_special_tokens=True)),
                              "cmp_first_matches_flip": cmp_ids[0] == int(s["top1"][int(chunk[r])])})
        res[kind] = {"knob": knob, "n_gsm_positions": int(mask.sum()),
                     "flip_rate": float(((s["top1"] != ref["top1"]) & mask).sum() / mask.sum()),
                     "n_flips_nontie": n_flip, "items": items}
        LOGGER.info("[%s] %d flips fora de empates; %d amostrados", kind, n_flip, len(items))
    bank.restore()
    res["questions"] = [question_text(r["prompt"]) for r in corpus.records]
    (out_dir(args.model) / "flips_tf.json").write_text(json.dumps(res))


def stage_judge_tf(args) -> None:
    f = json.loads((out_dir(args.model) / "flips_tf.json").read_text())
    p_different = load_judge(args)
    out = {"judge": args.judge, "items": {}, "summary": {}}
    for kind in METHODS:
        items = []
        for x in f[kind]["items"]:
            q = f["questions"][x["seq"]]
            p_ab = p_different(q, x["prefix"], x["ref"], x["cmp"])
            p_ba = p_different(q, x["prefix"], x["cmp"], x["ref"])
            items.append(x | {"p_ab": p_ab, "p_ba": p_ba, "p_different": 0.5 * (p_ab + p_ba),
                              "consistent": (p_ab > 0.5) == (p_ba > 0.5)})
        out["items"][kind] = items
        diff = np.array([x["p_different"] > 0.5 for x in items])
        out["summary"][kind] = {
            "n": len(items), "share_different": float(diff.mean()),
            "consistency": float(np.mean([x["consistent"] for x in items])),
            "flip_rate": f[kind]["flip_rate"],
            "different_flip_rate": float(f[kind]["flip_rate"] * diff.mean()),
            "cmp_first_matches_flip": float(np.mean([x["cmp_first_matches_flip"] for x in items])),
            "rel_pos_median": float(np.median([x["rel_pos"] for x in items]))}
    rng = np.random.default_rng(0)
    a = np.array([x["p_different"] > 0.5 for x in out["items"]["mag"]], dtype=float)
    b = np.array([x["p_different"] > 0.5 for x in out["items"]["wanda"]], dtype=float)
    boot = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(4000)]
    out["diff_mag_minus_wanda"] = {"share": float(a.mean() - b.mean()),
                                   "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5))}
    (OUT / "analysis" / args.out_name).write_text(json.dumps(out, indent=1))
    for k in METHODS:
        v = out["summary"][k]
        print(f"{k:6s} n={v['n']} flips que mudam o conteudo: {100 * v['share_different']:.1f}% | "
              f"consistencia {100 * v['consistency']:.1f}% | taxa de flips {v['flip_rate']:.4f} -> "
              f"flips que mudam o conteudo por token {v['different_flip_rate']:.4f}")
    d = out["diff_mag_minus_wanda"]
    print(f"mag - wanda (fracao): {d['share']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["gen", "judge", "flips", "judge-tf"])
    p.add_argument("--model", default="google/gemma-3-4b-it")
    p.add_argument("--judge", default="Qwen/Qwen2.5-72B-Instruct")
    p.add_argument("--target", type=float, default=0.1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--judge-gpus", type=int, nargs="+", default=[0, 1])
    p.add_argument("--judge-mem", default="78GiB", help="memoria maxima do juiz por GPU (GPUs compartilhadas)")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--n-task", type=int, default=400)
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--calib-batch", type=int, default=8)
    p.add_argument("--cont-tokens", type=int, default=40)
    p.add_argument("--n-flips", type=int, default=400, help="flips amostrados por metodo")
    p.add_argument("--out-name", default="semantic_judge.json",
                   help="arquivo em analysis/ (judge-tf); um segundo juiz grava em outro nome")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    torch.set_grad_enabled(False)
    {"gen": stage_gen, "judge": stage_judge, "flips": stage_flips, "judge-tf": stage_judge_tf}[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
