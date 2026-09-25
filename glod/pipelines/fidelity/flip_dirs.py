#!/usr/bin/env python3
"""Por que nenhuma perturbacao FIXA de peso consegue ser muito mais eficaz que ruido.

Para virar o token t, a perturbacao precisa deslocar o gap na direcao
    u_t = W_U[i1(t)] - W_U[i2(t)]     (diferenca das linhas de unembedding do top-1/top-2)
Uma perturbacao fixa produz um deslocamento de gap  <u_t, d h_t>, com d h_t = M h_t
para um mapa linear fixo M. Se as direcoes u_t (normalizadas) forem quase ortogonais
entre tokens, nenhum M fixo consegue se alinhar com todas: o ganho maximo de um
ataque rank-1 e limitado por E|<u_t, v>| do melhor v, que medimos aqui, junto com
o rank efetivo (participacao) do conjunto {u_t}.

    python -m glod flip-dirs --model Qwen/Qwen3-4B
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import torch
from safetensors import safe_open

from glod.paths import OUT as ROOT
from glod.paths import CACHE_DIR as CACHE


def unembedding(model_id: str) -> torch.Tensor:
    pat = f"{CACHE}/models--{model_id.replace('/', '--')}/snapshots/*/"
    path = glob.glob(pat)[0]
    idx_f = glob.glob(path + "*.index.json")
    keys = ("lm_head.weight", "model.embed_tokens.weight", "language_model.lm_head.weight",
            "model.language_model.embed_tokens.weight", "language_model.model.embed_tokens.weight")
    if idx_f:
        wm = json.load(open(idx_f[0]))["weight_map"]
        for k in keys:
            if k in wm:
                with safe_open(path + wm[k], "pt") as fh:
                    return fh.get_tensor(k)
    for f in sorted(glob.glob(path + "*.safetensors")):
        with safe_open(f, "pt") as fh:
            for k in keys:
                if k in fh.keys():
                    return fh.get_tensor(k)
    raise KeyError("unembedding nao encontrado")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--tokens", type=int, default=8192)
    p.add_argument("--rank-sweep", type=int, nargs="*", default=[2048, 8192, 32768],
                   help="checa se o rank efetivo converge com o numero de tokens")
    args = p.parse_args(argv)
    slug = args.model.split("/")[-1]
    ref = torch.load(ROOT / slug / slug / "bf16.pt", map_location="cpu", mmap=True, weights_only=False)
    Wu = unembedding(args.model).float()
    g = torch.Generator().manual_seed(0)
    sel = torch.randperm(len(ref["topi"]), generator=g)[: args.tokens]
    i1 = ref["topi"][sel][:, 0].long()
    i2 = ref["topi"][sel][:, 1].long()
    U = Wu[i1] - Wu[i2]                                   # [T, H] direcoes de flip
    U = U / U.norm(dim=1, keepdim=True).clamp(min=1e-6)
    # melhor direcao unica (ataque rank-1 ideal): maximiza E|<u_t, v>|
    v = torch.linalg.svd(U, full_matrices=False)[2][0]
    cos_best = (U @ v).abs()
    # rank efetivo (participacao) do espectro de U^T U
    s = torch.linalg.svdvals(U) ** 2
    part = (s.sum() ** 2 / (s ** 2).sum()).item()
    iso = 1.0 / np.sqrt(U.shape[1]) * np.sqrt(2 / np.pi)  # E|<u, v>| para v aleatorio
    print(f"{args.model}: T={U.shape[0]} H={U.shape[1]}")
    print(f"  E|cos| com a melhor direcao unica  : {cos_best.mean():.4f}  (mediana {cos_best.median():.4f})")
    print(f"  E|cos| com direcao aleatoria (iso) : {iso:.4f}")
    print(f"  ganho maximo de um ataque rank-1   : {cos_best.mean().item()/iso:.1f}x em |dgap| por unidade de norma")
    print(f"  rank efetivo (participacao) de U   : {part:.1f} de {U.shape[1]} dimensoes")
    conv = {}
    for nt in args.rank_sweep:
        s_ = torch.randperm(len(ref["topi"]), generator=torch.Generator().manual_seed(1))[:nt]
        Us = Wu[ref["topi"][s_][:, 0].long()] - Wu[ref["topi"][s_][:, 1].long()]
        Us = Us / Us.norm(dim=1, keepdim=True).clamp(min=1e-6)
        sv = torch.linalg.svdvals(Us) ** 2
        conv[nt] = (sv.sum() ** 2 / (sv ** 2).sum()).item()
    print("  convergencia do rank efetivo      : " + ", ".join(f"T={k}: {v:.1f}" for k, v in conv.items()))
    print(f"  fracao da energia nos 16 primeiros : {(s[:16].sum()/s.sum()).item():.3f}; nos 128: {(s[:128].sum()/s.sum()).item():.3f}")
    # --- teto teorico: KL minimo por token para virar o top-1 (projecao de KL no
    # conjunto {q: q(i2) >= q(i1)}), que da q(i1)=q(i2)=(p1+p2)/2 e custo
    #   c_t = p1 ln(2p1/(p1+p2)) + p2 ln(2p2/(p1+p2)).
    # Um atacante ORACULO (que muda cada token de forma independente, o que uma
    # perturbacao fixa de peso nao pode) gasta o orcamento nos tokens mais baratos.
    v2 = ref["topv"].double()
    p1 = v2[:, 0].exp(); p2 = v2[:, 1].exp(); m = (p1 + p2) / 2
    cost = p1 * (p1 / m).log() + p2 * (p2 / m).log()
    cs = torch.sort(cost).values
    cum = torch.cumsum(cs, 0) / len(cs)
    print("  teto do atacante oraculo (flips maximos para um KL medio dado):")
    law = json.loads((ROOT / "analysis" / "law.json").read_text())
    pts = [r for r in law["rows"] if r["ref"] == slug and r["model"] == slug and r["family"] != "outro modelo"]
    for budget in (0.01, 0.02, 0.05, 0.1, 0.2):
        f_max = float((cum <= budget).double().mean())
        near = sorted(pts, key=lambda r: abs(r["kl"] - budget))[:3]
        obs = float(np.mean([r["flip"] for r in near])) if near else float("nan")
        print(f"    KL={budget:.2f}: teto {100*f_max:6.2f}%  | compressores honestos {100*obs:5.2f}%  "
              f"| fracao do teto {obs/max(f_max,1e-9):.3f}")
    out = {"model": args.model, "eff_rank_convergence": conv,
           "oracle_ceiling": {str(b): float((cum <= b).double().mean()) for b in (0.01, 0.02, 0.05, 0.1, 0.2)}, "cos_best_mean": cos_best.mean().item(), "iso": float(iso),
           "gain_rank1": cos_best.mean().item() / float(iso), "eff_rank": part, "H": U.shape[1],
           "energy_top16": (s[:16].sum() / s.sum()).item(), "energy_top128": (s[:128].sum() / s.sum()).item()}
    f = ROOT / "analysis" / "flipdirs.json"
    all_out = json.loads(f.read_text()) if f.exists() else {}
    all_out[args.model] = out
    f.write_text(json.dumps(all_out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
