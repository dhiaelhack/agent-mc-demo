"""Calcul Monte Carlo minimal : transmission non diffusée d'un faisceau de photons
de 662 keV à travers une plaque de matériau.

Modèle volontairement simple : chaque photon parcourt une distance aléatoire
(loi exponentielle) avant sa première interaction ; il est « transmis » si cette
distance dépasse l'épaisseur. Pas de diffusion, pas de build-up, pas de détecteur réel.
Les coefficients d'atténuation linéiques ci-dessous sont APPROXIMATIFS (ordre de
grandeur des tables NIST XCOM à 662 keV) : à vérifier avant tout usage réel.
"""
import numpy as np

MU = {"plomb": 1.26, "fer": 0.60, "béton": 0.18, "eau": 0.086}   # cm^-1, approximatif


def free_paths(material, n, seed=0):
    """Tirages communs (même graine) : la transmission est alors monotone en épaisseur."""
    return np.random.default_rng(seed).exponential(1.0 / MU[material], n)


def simulate(material, thickness_cm, n=200_000, seed=0):
    fp = free_paths(material, n, seed)
    p = float((fp > thickness_cm).mean())
    err = float(np.sqrt(max(p * (1 - p), 1e-12) / n))
    analytic = float(np.exp(-MU[material] * thickness_cm))
    return {"material": material, "thickness_cm": round(thickness_cm, 3), "n_particles": n,
            "transmission": round(p, 5), "stat_error": round(err, 5),
            "analytic": round(analytic, 5)}


def find_thickness(material, target_transmission, n=200_000, seed=0, tol_cm=0.005):
    """Épaisseur pour laquelle la transmission descend à la cible (dichotomie)."""
    fp = free_paths(material, n, seed)
    lo, hi, calls = 0.0, 60.0, 0
    while hi - lo > tol_cm:
        mid = (lo + hi) / 2
        calls += 1
        if (fp > mid).mean() > target_transmission:
            lo = mid
        else:
            hi = mid
    t = (lo + hi) / 2
    return {"material": material, "target_transmission": target_transmission,
            "thickness_cm": round(t, 3), "simulations": calls,
            "analytic_thickness_cm": round(-np.log(target_transmission) / MU[material], 3)}


if __name__ == "__main__":
    for m in MU:
        print(simulate(m, 2.0))
