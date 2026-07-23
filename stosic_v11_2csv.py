from __future__ import annotations

"""
https://github.com/gajaka/luces-pvs-theories
"""

"""
stosic_v11_2csv.py — 7-node krug (K=7 / prilagodjenje 7/39) — Displacement concavity / entropy excess (7/39)

Izvor (Stosić / LUCES):
  luces-pvs-theories-main/displacement_concavity.pvs
  — H_bound = (1-t)H(μ₀)+t·H(μ₁)
  — excess = H(physical_path) − H_bound
  — thm_excess_detects_hybrid: excess > 0 ⇒ hybrid (ne-geodezijski)
  — thm_pure_excess_nonpos: čista putanja ⇒ excess ≤ 0

Mapiranje na 7/39:
  μ = draw_to_prob; physical mid = linear_interp (tačka po tačka), t=1/2
  excess_i = H(mid) − H_bound  (≥0 po ax_entropy_concavity_of_mixing)
  skor += excess_i na brojevima izvlačenja i+1  (jačina hybrid detektora)
  next = top 7; ceo CSV; bez randoma; stop ako uzastopni/AP
"""

from typing import List

import numpy as np

from stosic_v1_2csv import CSV_LOTO, CSV_PLUS, EPS, draw_to_prob, load_draws
from stosic_v2_2csv import top7_from_freq
from stosic_v5_2csv import shannon_H
from stosic_v10_2csv import is_degenerate


def predict_next(draws: np.ndarray) -> List[int]:
    probs = np.vstack([draw_to_prob(d) for d in draws])
    t = 0.5  # SEED/(SEED+SEED)
    skor = np.zeros(probs.shape[1], dtype=np.float64)

    for i in range(len(probs) - 1):
        mu0, mu1 = probs[i], probs[i + 1]
        h0, h1 = shannon_H(mu0), shannon_H(mu1)
        h_bound = (1.0 - t) * h0 + t * h1
        mid = (1.0 - t) * mu0 + t * mu1
        mid = np.clip(mid, EPS, None)
        mid = mid / mid.sum()
        excess = shannon_H(mid) - h_bound
        # T3: excess > 0 detektuje hybrid; težina = excess
        if excess > 0.0:
            w = float(excess)
            for n in draws[i + 1]:
                skor[int(n) - 1] += w

    if float(skor.sum()) <= 0:
        for d in draws:
            for n in d:
                skor[int(n) - 1] += 1.0

    combo = top7_from_freq(skor)
    if is_degenerate(combo):
        nu = np.zeros_like(skor)
        for d in draws:
            for n in d:
                nu[int(n) - 1] += 1.0
        combo = top7_from_freq(nu)
    return combo


def main():
    next_loto = predict_next(load_draws(CSV_LOTO))
    next_loto_plus = predict_next(load_draws(CSV_PLUS))
    if is_degenerate(next_loto):
        raise SystemExit("degenerisan next_loto (uzastopni/AP) — zaustavljen pre ispisa")
    if is_degenerate(next_loto_plus):
        raise SystemExit("degenerisan next_loto_plus (uzastopni/AP) — zaustavljen pre ispisa")
    print("next_loto:      ", next_loto)
    print("next_loto_plus: ", next_loto_plus)


if __name__ == "__main__":
    main()



"""
next_loto:       [8, 16, 21, 22, 23, 35, 38]
next_loto_plus:  [2, 8, 11, 23, 26, 32, 34]
"""



"""
v11: displacement_concavity — skor težina = entropy excess (hybrid detector).
"""



"""
21 teorija

fisher_voronoi → v1, v2
dual_observability → v3
v4 se pozivao na W₂/stabilnost — slabo / nije strogo
entropy_along_geodesic → v5
velocity_asymmetry (+ delom lie_generator_structure) → v6
brenier_uniqueness (+ delom rank_orientation) → v7

kantorovich_duality
cyclical_monotonicity
displacement_interpolation
displacement_concavity
wasserstein_metric (strogo)
transport_structure
transport_structure_v2
transport_stability
stability_of_maps
monge_kantorovich_equivalence
lie_generator_structure (pun T10)
fisher_boundary
hybrid_observability
tangent_bundle
global_optimality
"""



"""
Kratko, o repou:

21 PVS teorija — sve su prošle kroz v1–v22 (neke ranije labavo: naročito v3/v4; rank_orientation je ušao uz Brenier u v7).
Repo je o spektralnom OT / LUCES (ESP32), ne o lotou — 7/39 je naša mapa, ne Stosićev domen.
Najčistije jezgro oko Fisher–Voronoi, Brenier/CM, W₂, T10 (lie_generator_structure). global_optimality je samo aksiomi + lema (bez teorema).
Empirija u PVS-u (bootovi, κ, Monge fraction) ne prenosi se automatski na CSV — samo struktura ideja.
"""
