#!/usr/bin/env python3
"""P1: ataque adversarial MULTICAMADA a lei flips ~ kappa*sqrt(KL).

O ataque de uma camada (`glod adv-single`) so podia mexer no down_proj final,
que entra nos logits por um caminho quase linear. Aqui o atacante recebe o poder
maximo compativel com a restricao da tese ("perturbacao FIXA de peso"):

  * um delta de baixo rank dW_l = A_l B_l em TODAS as nn.Linear de TODAS as
    camadas do decoder (q,k,v,o,gate,up,down), somado a saida da projecao
    (identico a somar A_l B_l ao peso, mas sem copiar o modelo);
  * gradiente pelo modelo INTEIRO (checkpointing de ativacao), nao por um cache
    da ultima camada;
  * objetivo que maximiza log kappa = log(flips) - 1/2 log(KL) diretamente,
    portanto invariante a escala - a bisseccao posterior nao tira o ataque do
    otimo;
  * avaliacao honesta: a direcao e reescalada por bisseccao para casar o KL alvo,
    materializada nos pesos em bf16 e pontuada pelo pipeline normal no corpus
    inteiro, com split treino/teste por sequencia (o ataque nunca viu os tokens
    de teste).

Reporta kappa contra o kappa dos compressores honestos do mesmo modelo e contra
o teto do atacante oraculo (que pode escolher o KL token a token).

    python -m glod adv-multi --model google/gemma-3-1b-it --device cuda:2
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from glod.core import compressors as C
from glod.core.fidelity import score_fidelity, subset_index
from glod.core.model_loader import load_model, resolve_decoder
from glod.corpora.token_oracle import GeneratedCorpus, teacher_forcing_batch
from glod.paths import CACHE_DIR as DEFAULT_CACHE_DIR
from glod.paths import OUT, add_corpus_arg, ref_slug
from glod.pipelines.fidelity.build_grid import use_fp32_logits

LOGGER = logging.getLogger("glod.advmulti")


# ------------------------------------------------------------------ o ataque
class LowRankAttack:
    """dW = A B por modulo, aplicado como hook na saida da projecao."""

    def __init__(self, targets, rank: int, device, init: float = 1e-3, seed: int = 0):
        g = torch.Generator(device=device).manual_seed(seed)
        self.params: list[torch.Tensor] = []
        self.mods: list[nn.Linear] = []
        self.AB: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.handles: list = []
        self.diag: tuple | None = None
        self.scale = 1.0
        self.full = rank <= 0
        for t in targets:
            d_out, d_in = t.module.weight.shape
            if self.full:   # rank completo: TODO peso livre (atacante maximo)
                A = (init * torch.randn(d_out, d_in, device=device, dtype=torch.float32, generator=g))
                A.requires_grad_(True)
                self.params.append(A)
                self.AB.append((A, None))
            else:
                r = min(rank, d_out, d_in)
                A = (init * torch.randn(d_out, r, device=device, dtype=torch.float32, generator=g))
                B = (init * torch.randn(r, d_in, device=device, dtype=torch.float32, generator=g))
                A.requires_grad_(True); B.requires_grad_(True)
                self.params += [A, B]
                self.AB.append((A, B))
            self.mods.append(t.module)

    def add_output_scale(self, norm_mod: nn.Module, device) -> None:
        """Delta diagonal no peso da norma final: e a direcao de TEMPERATURA.

        Sem isso o atacante nao pode nem imitar temperatura, porque o RMSNorm final
        remove a escala global do estado oculto: so o peso da norma (ou o lm_head)
        controla a escala dos logits. O controle positivo do experimento depende disso.
        """
        d = next(norm_mod.parameters()).shape[0]
        g = torch.zeros(d, device=device, dtype=torch.float32).requires_grad_(True)
        self.params.append(g)
        self.diag = (norm_mod, g)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.params)

    def attach(self, *, detached: bool) -> None:
        self.remove()
        for mod, (A, B) in zip(self.mods, self.AB):
            a = A.detach() if detached else A
            b = None if B is None else (B.detach() if detached else B)

            def hook(_m, inp, out, a=a, b=b):
                x = inp[0].to(torch.float32)
                d = (x @ a.T) if b is None else ((x @ b.T) @ a.T)
                return out + (d * self.scale).to(out.dtype)

            self.handles.append(mod.register_forward_hook(hook))
        if self.diag is not None:
            nm, g = self.diag
            gd = g.detach() if detached else g

            def dhook(_m, _inp, out, gd=gd):
                return out * (1.0 + self.scale * gd).to(out.dtype)

            self.handles.append(nm.register_forward_hook(dhook))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def param_norm(self) -> float:
        return float(sum(p.pow(2).sum().item() for p in self.params)) ** 0.5

    def diag_share(self) -> float:
        """Fracao da norma do ataque que esta no delta da norma final (escala de saida)."""
        if self.diag is None:
            return 0.0
        tot = sum(p.pow(2).sum().item() for p in self.params)
        return float(self.diag[1].pow(2).sum().item() / max(tot, 1e-30))

    @torch.no_grad()
    def renorm(self, target: float) -> None:
        """Fixa a norma dos parametros do ataque: otimizacao na esfera.

        Necessario porque o KL medio e hipersensivel a escala global (expoente local
        ~8): deixar o Adam crescer a norma livremente tira o ataque do orcamento em
        poucos passos, e perseguir o KL com passo multiplicativo nao acompanha.
        """
        cur = self.param_norm()
        if cur > 0:
            for p_ in self.params:
                p_.mul_(target / cur)

    @torch.no_grad()
    def materialize(self, bank: C.WeightBank, scale: float) -> dict:
        """Escreve W0 + scale*A B em bf16 (a perturbacao de verdade) e mede a norma."""
        num = den = 0.0
        if self.diag is not None:   # o delta da norma final entra como peso tambem
            nm, g = self.diag
            w = next(nm.parameters())
            bank.extra["final_norm"] = (w.data, w.data.clone().cpu())
            w.data.copy_((w.data.float() * (1.0 + scale * g.detach())).to(w.dtype))
        by_mod = {id(t.module): t for t in bank.targets}
        for mod, (A, B) in zip(self.mods, self.AB):
            t = by_mod[id(mod)]
            W0 = t.original.to(mod.weight.device).float()
            dW = scale * (A.detach() if B is None else A.detach() @ B.detach())
            num += dW.pow(2).sum().item(); den += W0.pow(2).sum().item()
            mod.weight.copy_((W0 + dW).to(mod.weight.dtype))
            del W0, dW
        return {"rel_frobenius": math.sqrt(num / den)}


# --------------------------------------------------------------- referencias
def honest_kappa(model_slug: str, ref: dict) -> dict:
    """kappa dos compressores honestos do mesmo modelo, no mesmo corpus."""
    law = json.loads((OUT / "analysis" / "law.json").read_text())
    pts = [(r["config"], r["kl"], r["flip"]) for r in law["rows"]
           if r["ref"] == model_slug and r["model"] == model_slug
           and r.get("family") != "outro modelo" and 1e-3 < r["kl"] < 0.25]
    if not pts:   # modelo fora do law.json: recalcula dos proprios .pt
        for f in sorted((OUT / model_slug / model_slug).glob("*.pt")):
            if f.stem == "bf16":
                continue
            c = torch.load(f, map_location="cpu", mmap=True, weights_only=False)
            if "kl" not in c:
                continue
            kl = c["kl"].double().mean().item()
            fl = (c["top1"] != ref["top1"]).double().mean().item()
            if 1e-3 < kl < 0.25:
                pts.append((f.stem, kl, fl))
    ks = sorted(fl / math.sqrt(kl) for _, kl, fl in pts)
    return {"kappa_honest": float(np.median(ks)) if ks else float("nan"),
            "kappa_honest_p90": float(np.percentile(ks, 90)) if ks else float("nan"),
            "n_honest": len(pts), "points": pts}


def oracle_ceiling(ref: dict, budgets) -> dict:
    v = ref["topv"].double()
    p1, p2 = v[:, 0].exp(), v[:, 1].exp()
    m = (p1 + p2) / 2
    cost = p1 * (p1 / m).log() + p2 * (p2 / m).log()
    cum = torch.cumsum(torch.sort(cost).values, 0) / len(cost)
    return {str(b): float((cum <= b).double().mean()) for b in budgets}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="google/gemma-3-1b-it")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--rank", type=int, default=8, help="0 ou negativo = rank completo por modulo")
    p.add_argument("--targets", default="all", choices=["all", "mlp", "down_o", "down"])
    p.add_argument("--kl-budget", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    p.add_argument("--modes", nargs="+", default=["malign", "benign"],
                   help="malign = maximiza flips a KL fixo; benign = minimiza (controle positivo)")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr-rel", type=float, default=0.01,
                   help="passo do Adam relativo ao rms dos parametros do ataque (Adam nao e invariante a escala)")
    p.add_argument("--lam", type=float, default=10.0, help="peso da penalidade log-KL = log-orcamento")
    p.add_argument("--tau", type=float, default=0.03, help="largura final do sigmoide do proxy de flip (nats)")
    p.add_argument("--tau-start", type=float, default=0.3, help="largura inicial (anelada geometricamente)")
    p.add_argument("--output-scale", action=argparse.BooleanOptionalAction, default=True,
                   help="da ao atacante um delta na norma final (direcao de temperatura); "
                        "--no-output-scale restringe o ataque ao mesmo espaco dos compressores")
    p.add_argument("--project", action="store_true", default=True,
                   help="reescala o delta a cada passo para o KL do minilote bater o orcamento")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--seqs-per-step", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-seqs", type=int, default=32, help="lote fixo para medir kappa sem ruido")
    p.add_argument("--gen-trunc", type=int, default=128, help="tokens gerados usados por sequencia no treino")
    p.add_argument("--topk", type=int, default=128)
    p.add_argument("--score-batch", type=int, default=4)
    p.add_argument("--probe-seqs", type=int, default=96)
    p.add_argument("--tag", default="multi")
    p.add_argument("--seed", type=int, default=0,
                   help="semente do init do delta e dos minilotes/validacao (0 = rodadas antigas)")
    add_corpus_arg(p)
    p.add_argument("--logits-fp32", action="store_true",
                   help="lm_head em float32: remove o piso de empates exatos da grade do bf16, "
                        "que infla kappa dos honestos justamente no KL baixo")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    ms = ref_slug(args.model, corpus=args.corpus)
    # o corpus greedy e sempre o do bf16; so a PONTUACAO muda com fp32 (como em `glod grid`)
    corpus = GeneratedCorpus.from_dict(json.loads((OUT / "corpora" / f"{ms}.json").read_text()))
    rs = ms + ("__fp32" if args.logits_fp32 else "")
    out_dir = OUT / "adversarial" / rs
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_full = torch.load(OUT / rs / rs / "bf16.pt", map_location="cpu", mmap=True, weights_only=False)
    loaded = load_model(args.model, role="cfg", device=args.device, dtype="bfloat16", cache_dir=args.cache_dir)
    if args.logits_fp32:
        use_fp32_logits(loaded.model)
    dec = resolve_decoder(loaded.model)
    bank = C.WeightBank(dec)
    for prm in loaded.model.parameters():
        prm.requires_grad_(False)
    keep_pat = {"all": None, "mlp": ("gate_proj", "up_proj", "down_proj"),
                "down_o": ("down_proj", "o_proj"), "down": ("down_proj",)}[args.targets]
    targets = [t for t in bank.targets if keep_pat is None or any(k in t.name for k in keep_pat)]
    LOGGER.info("%s: %d modulos atacados (%s), rank %d", args.model, len(targets), args.targets, args.rank)

    # split por sequencia com PASSO FIXO, nao por prefixo: o corpus e ordenado por
    # dominio (GSM8K e depois MMLU) e um split por prefixo treinaria num dominio e
    # mediria no outro - e kappa difere entre dominios (~+0.10 em MMLU).
    n_seq = len(corpus.gen_ids)
    every = max(2, int(round(1 / max(1e-9, 1 - args.train_frac))))
    is_test = [i % every == 0 for i in range(n_seq)]
    train_seqs = [i for i in range(n_seq) if not is_test[i]]
    offs = np.cumsum([0] + [len(g) for g in corpus.gen_ids])
    tok_train = torch.zeros(int(offs[-1]), dtype=torch.bool)
    for i in train_seqs:
        tok_train[int(offs[i]):int(offs[i + 1])] = True

    # ------------------------------------------------- cache de referencia (treino)
    dev = loaded.device
    pad = loaded.tokenizer.pad_token_id
    gen_tr = {i: corpus.gen_ids[i][: args.gen_trunc] for i in train_seqs}
    cache: dict[int, dict] = {}
    with torch.no_grad():
        for s in range(0, len(train_seqs), args.seqs_per_step):
            sel = train_seqs[s:s + args.seqs_per_step]
            gens = [gen_tr[i] for i in sel]
            ids, mask, pos, kp = teacher_forcing_batch([corpus.prompt_ids[i] for i in sel], gens, pad)
            lg = loaded.model(input_ids=ids.to(dev), attention_mask=mask.to(dev), position_ids=pos.to(dev),
                              use_cache=False, logits_to_keep=kp).logits
            for row, i in enumerate(sel):
                n = len(gen_tr[i])
                lp = lg[row, kp - n:, :].float().log_softmax(-1)
                v, ix = lp.topk(args.topk, dim=-1)
                cache[i] = {"topv": v.half().cpu(), "topi": ix.int().cpu(),
                            "i1": lp.argmax(-1).cpu()}
    LOGGER.info("cache de referencia: %d sequencias de treino (de %d), %d tokens de treino",
                len(cache), n_seq, sum(len(g) for g in gen_tr.values()))

    atk = LowRankAttack(targets, args.rank, dev, seed=args.seed)
    LOGGER.info("parametros do ataque: %.2fM (%.4f%% do modelo)", atk.n_params() / 1e6,
                100 * atk.n_params() / bank.total_params())
    loaded.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    loaded.model.train()   # so para o checkpointing; nao ha dropout ativo nesses modelos

    results = {}
    f = out_dir / f"results_{args.tag}.json"
    if f.exists():
        results = json.loads(f.read_text())
    hon = honest_kappa(rs, ref_full)
    ceil = oracle_ceiling(ref_full, args.kl_budget)
    LOGGER.info("[%s] kappa honesto (mediana de %d configs): %.3f | teto oraculo %s", rs,
                hon["n_honest"], hon["kappa_honest"], {k: round(v, 4) for k, v in ceil.items()})

    for mode, budget in [(m, b) for b in args.kl_budget for m in args.modes]:
        sign = 1.0 if mode == "malign" else -1.0
        key = f"{mode}_multi@{budget}"
        if key in results:
            LOGGER.info("[%s] ja feito, pulando", key)
            continue
        # reinicia o ataque para cada orcamento
        atk.remove()
        atk = LowRankAttack(targets, args.rank, dev, seed=args.seed)
        if args.output_scale:
            atk.add_output_scale(dec.norm, dev)
        atk.scale = 1.0
        atk.attach(detached=False)
        rng = np.random.default_rng(args.seed)

        def measure(sel, chunk: int = 4) -> tuple[float, float]:
            """KL medio e taxa de flips DURA num conjunto fixo (sem gradiente, sem ruido de lote)."""
            ks, hs = [], []
            with torch.no_grad():
                for c0 in range(0, len(sel), chunk):
                    part = sel[c0:c0 + chunk]
                    gens = [gen_tr[i] for i in part]
                    ids, mask, pos, kp = teacher_forcing_batch([corpus.prompt_ids[i] for i in part], gens, pad)
                    lg = loaded.model(input_ids=ids.to(dev), attention_mask=mask.to(dev),
                                      position_ids=pos.to(dev), use_cache=False, logits_to_keep=kp).logits
                    for row, i in enumerate(part):
                        n = len(gen_tr[i])
                        q = lg[row, kp - n:, :].float().log_softmax(-1)
                        rv = cache[i]["topv"].to(dev).float(); ri = cache[i]["topi"].to(dev).long()
                        i1 = cache[i]["i1"].to(dev)
                        qv = q.gather(-1, ri)
                        P, Q = rv.exp(), qv.exp()
                        tP = (1 - P.sum(-1)).clamp(min=1e-8); tQ = (1 - Q.sum(-1)).clamp(min=1e-8)
                        ks.append((P * (rv - qv)).sum(-1) + tP * (tP.log() - tQ.log()))
                        hs.append((q.argmax(-1) != i1).float())
            return torch.cat(ks).mean().item(), torch.cat(hs).mean().item()

        def minibatch_kl(sel) -> float:
            return measure(sel)[0]

        val_seqs = [int(x) for x in rng.choice(train_seqs, size=min(args.eval_seqs, len(train_seqs)),
                                                replace=False)]
        # A escala inicial e calibrada por BISSECCAO em log no lote de VALIDACAO. Passo de
        # Newton (KL ~ s^2) nao serve: com 182 deltas que COMPOEM entre camadas o KL cresce
        # com expoente local ~8 em s, e o Newton oscila sem convergir.
        def set_scale(v: float) -> None:
            atk.scale = v

        lo_c, hi_c = 1e-4, 1e4
        for _ in range(14):
            mid_c = math.sqrt(lo_c * hi_c)
            set_scale(mid_c)
            if measure(val_seqs)[0] < budget:
                lo_c = mid_c
            else:
                hi_c = mid_c
        set_scale(1.0)
        with torch.no_grad():
            for p_ in atk.params:
                p_.mul_(math.sqrt(lo_c * hi_c) ** (0.5 if not atk.full else 1.0))
        # raio da esfera em que a otimizacao acontece (ver renorm)
        tnorm = atk.param_norm()
        rms = float(np.sqrt(np.mean([p_.pow(2).mean().item() for p_ in atk.params])))
        lr = args.lr_rel * rms
        kl0, fl0 = measure(val_seqs)
        LOGGER.info("  [%s %.3f] init calibrado: KL %.4f | kappa inicial (delta aleatorio) %.3f | "
                    "rms %.3g | lr %.3g", mode, budget, kl0, fl0 / math.sqrt(max(kl0, 1e-12)), rms, lr)
        opt = torch.optim.Adam(atk.params, lr=lr)
        ema = None
        best = {"kappa": -1.0 if sign > 0 else 1e9, "step": -1, "state": None}
        t0 = time.time()
        hist, traj = [], []
        torch.set_grad_enabled(True)
        for step in range(args.steps):
            frac = step / max(args.steps - 1, 1)
            tau = args.tau_start * (args.tau / args.tau_start) ** frac
            sel = [int(x) for x in rng.choice(train_seqs, size=args.seqs_per_step, replace=False)]
            gens = [gen_tr[i] for i in sel]
            ids, mask, pos, kp = teacher_forcing_batch([corpus.prompt_ids[i] for i in sel], gens, pad)
            lg = loaded.model(input_ids=ids.to(dev), attention_mask=mask.to(dev), position_ids=pos.to(dev),
                              use_cache=False, logits_to_keep=kp).logits
            kls, flips, hards = [], [], []
            for row, i in enumerate(sel):
                n = len(gen_tr[i])
                q = lg[row, kp - n:, :].float().log_softmax(-1)
                rv = cache[i]["topv"].to(dev).float()
                ri = cache[i]["topi"].to(dev).long()
                i1 = cache[i]["i1"].to(dev)
                qv = q.gather(-1, ri)
                P, Q = rv.exp(), qv.exp()
                tailP = (1 - P.sum(-1)).clamp(min=1e-8)
                tailQ = (1 - Q.sum(-1)).clamp(min=1e-8)
                kls.append((P * (rv - qv)).sum(-1) + tailP * (tailP.log() - tailQ.log()))
                other = q.scatter(-1, i1[:, None], -1e4).max(-1).values
                gap = q.gather(-1, i1[:, None]).squeeze(-1) - other
                flips.append(torch.sigmoid(-gap / tau))
                hards.append((gap < 0).float())
            kl = torch.cat(kls).mean().clamp(min=1e-10)
            fl = torch.cat(flips).mean().clamp(min=1e-8)
            hard = torch.cat(hards).mean().item()   # flips de verdade: comparavel entre passos
            # maximiza log kappa, com penalidade fraca para ficar no regime do orcamento
            # malign: maximiza log kappa = log(flips) - 1/2 log(KL); benign: minimiza.
            # o controle benigno e essencial: se o gradiente consegue derrubar kappa
            # muito abaixo da taxa honesta, a falha em subi-lo nao e falha de otimizacao.
            loss = sign * (-fl.log() + 0.5 * kl.log()) + args.lam * (kl.log() - math.log(budget)) ** 2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            atk.renorm(tnorm)          # otimizacao na esfera de norma fixa
            ema = kl.item() if ema is None else 0.8 * ema + 0.2 * kl.item()
            if args.project:
                # projeta na superficie KL = orcamento usando a MEDIA MOVEL do KL.
                # O expoente NAO e 1/2: com 182 deltas que compoem entre camadas, o KL
                # medio cresce com expoente local ~8 na escala global (e a media e
                # dominada por poucos tokens de KL explosivo), logo sqrt(budget/KL)
                # chuta ordens de magnitude. Passo multiplicativo pequeno e o certo.
                tnorm *= min(1.02, max(0.98, (budget / max(ema, 1e-12)) ** 0.125))
            # trajetoria e selecao medidas num lote FIXO: com 512 tokens de minilote a
            # contagem de flips tem ruido de ~40% e a selecao pegaria sorte, nao progresso
            if step % args.eval_every == 0 or step == args.steps - 1:
                kl_v, fl_v = measure(val_seqs)
                kap_v = fl_v / math.sqrt(max(kl_v, 1e-12))
                traj.append({"step": step, "kl_val": kl_v, "flip_val": fl_v, "kappa_val": kap_v})
                if 0.5 * budget < kl_v < 2 * budget and sign * kap_v > sign * best["kappa"]:
                    best = {"kappa": kap_v, "step": step,
                            "state": [(A.detach().clone(), None if B is None else B.detach().clone())
                                      for A, B in atk.AB]}
                tnorm *= min(1.25, max(0.8, (budget / max(kl_v, 1e-12)) ** 0.125))
                atk.renorm(tnorm)
                ema = budget
            if step % 50 == 0 or step == args.steps - 1:
                kappa_hard = hard / math.sqrt(kl.item())
                hist.append({"step": step, "kl": kl.item(), "ema": ema, "flip_proxy": fl.item(),
                             "flip_hard": hard, "kappa_hard": kappa_hard, "tau": tau})
                v = traj[-1] if traj else {"kl_val": float("nan"), "kappa_val": float("nan")}
                LOGGER.info("  [%s %.3f] passo %3d: lote KL %.4f (ema %.4f) flips %.4f | VALIDACAO "
                            "KL %.4f KAPPA %.3f | tau %.3f", mode, budget, step, kl.item(), ema, hard,
                            v["kl_val"], v["kappa_val"], tau)
        torch.set_grad_enabled(False)
        atk.remove()
        if best["state"] is not None:
            LOGGER.info("  usando o melhor iterado: passo %d (kappa na validacao %.3f)", best["step"], best["kappa"])
            with torch.no_grad():
                for (A, B), (A_b, B_b) in zip(atk.AB, best["state"]):
                    A.copy_(A_b)
                    if B is not None:
                        B.copy_(B_b)
            best["state"] = None

        # ---------------------------------------- avaliacao honesta a KL casado
        loaded.model.gradient_checkpointing_disable()
        loaded.model.eval()
        atk.attach(detached=True)
        probe, keep_idx = _probe(corpus, args.probe_seqs)
        ref_probe = {k: ref_full[k][keep_idx] for k in ("top1", "topv", "topi")}

        def kl_at(s: float) -> float:
            atk.scale = s
            r = score_fidelity(loaded, probe, ref=ref_probe, batch_size=args.score_batch)
            return r["kl"].double().mean().item()

        lo, hi = 1e-6, 1.0
        while kl_at(hi) < budget and hi < 1e6:
            lo, hi = hi, hi * 4
        while kl_at(lo) > budget and lo > 1e-12:
            hi, lo = lo, lo / 4
        for _ in range(28):
            mid = math.sqrt(lo * hi)
            if kl_at(mid) < budget:
                lo = mid
            else:
                hi = mid
        s_hat = math.sqrt(lo * hi)
        # o delta e aplicado EXATO (o hook soma A B a saida da projecao, identico a
        # W <- W + A B em fp32). Materializar em bf16 e reportado a parte: um delta de
        # norma relativa ~1e-4 fica abaixo do passo de arredondamento do bf16 (~4e-3)
        # e seria destruido pelo armazenamento, o que mediria o bf16, nao o ataque.
        # o KL medio por token e de cauda pesada, logo KL(s) nao e uma lei de potencia
        # limpa: em vez de insistir em casar o alvo, medimos no corpus inteiro na escala
        # do probe e reportamos kappa no KL OBTIDO (a lei diz que kappa nao depende do KL,
        # e e assim que cada compressor honesto entra na figura: um ponto (KL, flips)).
        atk.scale = s_hat
        sub = subset_index(corpus.n_tokens, 2048)
        res = score_fidelity(loaded, corpus, ref=ref_full, batch_size=args.score_batch, subset=sub)
        atk.remove()
        norms = atk.materialize(bank, s_hat)
        res_bf16 = score_fidelity(loaded, corpus, ref=ref_full, batch_size=args.score_batch)
        bank.restore()
        fl16 = (res_bf16["top1"] != ref_full["top1"]).double().mean().item()
        kl16 = res_bf16["kl"].double().mean().item()
        norms["bf16_materialized"] = {"flip": fl16, "kl": kl16, "kappa": fl16 / max(kl16, 1e-12) ** 0.5}

        # guarda a saida de fidelidade do ataque: permite medir o fator de anisotropia
        # A = E|dgap| / sqrt(2 KL) com a mesma conta dos compressores honestos (cmd_prop)
        torch.save({k: res[k] for k in ("top1", "kl", "tv", "at_ref", "logprob", "entropy")},
                   # a tag entra no nome: variantes (sementes, sem norma, rank completo) no mesmo
                   # diretorio sobrescreviam o fid do ataque principal usado na anisotropia
                   out_dir / (f"fid_{mode}_{budget}.pt" if args.tag == "multi"
                              else f"fid_{args.tag}_{mode}_{budget}.pt"))
        flip_t = (res["top1"] != ref_full["top1"])
        kl_t = res["kl"].double()
        row = {"mode": f"{mode}_multi", "budget": budget, "scale": s_hat, "rank": args.rank,
               "targets": args.targets, "n_modules": len(targets), "attack_params": atk.n_params(),
               "steps": args.steps, "seconds": time.time() - t0, "history": hist,
               "best_step": best["step"], "best_kappa_val": best["kappa"], "traj": traj,
               "output_scale": bool(args.output_scale), "seed": args.seed, "diag_share": atk.diag_share(), **norms}
        for name, m in (("all", torch.ones_like(tok_train)), ("train", tok_train), ("test", ~tok_train)):
            fl = flip_t[m].double().mean().item()
            kl = kl_t[m].mean().item()
            row[name] = {"flip": fl, "kl": kl, "kappa": fl / max(kl, 1e-12) ** 0.5}
        row["kappa_val_initial"] = fl0 / math.sqrt(max(kl0, 1e-12))
        row.update(hon | {"oracle_ceiling": ceil})
        row["kappa_over_honest"] = row["all"]["kappa"] / hon["kappa_honest"]
        row["ceiling_fraction"] = row["all"]["flip"] / max(ceil[str(budget)], 1e-9)
        results[key] = row
        LOGGER.info("[%s] CORPUS: flip %.4f | KL %.4f | kappa %.3f = %.2fx honesto (%.3f) | "
                    "fracao do teto %.3f | kappa treino %.3f teste %.3f | |dW|/|W| %.4f (%.0fs)", key,
                    row["all"]["flip"], row["all"]["kl"], row["all"]["kappa"], row["kappa_over_honest"],
                    hon["kappa_honest"], row["ceiling_fraction"], row["train"]["kappa"], row["test"]["kappa"],
                    norms["rel_frobenius"], row["seconds"])
        f.write_text(json.dumps(results, indent=1))
        loaded.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        loaded.model.train()
    return 0


def _probe(corpus: GeneratedCorpus, n: int):
    """Subcorpus com passo fixo (o corpus e ordenado por dominio; pegar os primeiros enviesa)."""
    step = max(1, len(corpus.gen_ids) // n)
    sel = list(range(0, len(corpus.gen_ids), step))[:n]
    offs = np.cumsum([0] + [len(g) for g in corpus.gen_ids])
    keep = torch.cat([torch.arange(offs[i], offs[i + 1]) for i in sel])
    sub = GeneratedCorpus(records=[corpus.records[i] for i in sel],
                          prompt_ids=[corpus.prompt_ids[i] for i in sel],
                          gen_ids=[corpus.gen_ids[i] for i in sel])
    return sub, keep


if __name__ == "__main__":
    raise SystemExit(main())
