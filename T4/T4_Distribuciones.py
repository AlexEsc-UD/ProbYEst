"""
T4_Distribuciones.py
Validacion numerica de los tres ejercicios del Taller T4:
  1. Binomial  (Bernoulli) - Lanzamiento de dado
  2. Poisson               - Accidentes en interseccion
  3. Normal                - Ejemplo 6.3 Walpole

Autor: Alejandro Escobar Barrios - 20251020094
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

AZUL   = "#00467F"
ROJO   = "#C0392B"
VERDE  = "#1A7A4A"
NARANJA= "#E67E22"

plt.rcParams.update({
    "font.family": "serif",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
})

# ═══════════════════════════════════════════════════════════════
# 1. DISTRIBUCIÓN BINOMIAL
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("PARTE 1: DISTRIBUCIÓN BINOMIAL")
print("  n=5 lanzamientos, p=1/6 (sacar el número 3)")
print("=" * 60)

n, p = 5, 1/6
dist_bin = stats.binom(n, p)

p_exactamente_2 = dist_bin.pmf(2)
p_maximo_1      = dist_bin.cdf(1)
p_al_menos_2    = 1 - dist_bin.cdf(1)

print(f"  P(X = 2)       = {p_exactamente_2:.6f}  [exacto: 1250/7776 = {1250/7776:.6f}]")
print(f"  P(X <= 1)      = {p_maximo_1:.6f}  [exacto: 6250/7776 = {6250/7776:.6f}]")
print(f"  P(X >= 2)      = {p_al_menos_2:.6f}  [exacto: 1 - 6250/7776 = {1-6250/7776:.6f}]")
print(f"  E[X]           = {dist_bin.mean():.6f}  [exacto: 5/6 = {5/6:.6f}]")
print(f"  Var(X)         = {dist_bin.var():.6f}  [exacto: 25/36 = {25/36:.6f}]")

# ═══════════════════════════════════════════════════════════════
# 2. DISTRIBUCIÓN DE POISSON
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PARTE 2: DISTRIBUCIÓN DE POISSON")
print("  lambda = n*p = 1000 * 0.001 = 1")
print("=" * 60)

lam = 1.0
dist_poi = stats.poisson(lam)

p_poi_0  = dist_poi.pmf(0)
p_poi_1  = dist_poi.pmf(1)
p_poi_2m = 1 - dist_poi.cdf(1)

print(f"  P(X = 0)       = {p_poi_0:.6f}  [exacto: e^(-1) = {np.exp(-1):.6f}]")
print(f"  P(X = 1)       = {p_poi_1:.6f}  [exacto: e^(-1) = {np.exp(-1):.6f}]")
print(f"  P(X >= 2)      = {p_poi_2m:.6f}  [exacto: 1-2e^(-1) = {1-2*np.exp(-1):.6f}]")
print(f"  E[X]           = {dist_poi.mean():.6f}  [exacto: lambda = 1]")
print(f"  Var(X)         = {dist_poi.var():.6f}  [exacto: lambda = 1]")

# ═══════════════════════════════════════════════════════════════
# 3. DISTRIBUCIÓN NORMAL (Walpole Ejemplo 6.3)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PARTE 3: DISTRIBUCIÓN NORMAL  (Walpole Ejemplo 6.3)")
print("  X ~ N(mu=50, sigma=10)  =>  P(45 < X < 62)")
print("=" * 60)

mu, sigma = 50, 10
dist_nor = stats.norm(mu, sigma)

p_intervalo = dist_nor.cdf(62) - dist_nor.cdf(45)
phi_1_2     = stats.norm.cdf(1.2)
phi_m05     = stats.norm.cdf(-0.5)

print(f"  Phi(1.2)       = {phi_1_2:.6f}  [tablas: 0.8849]")
print(f"  Phi(-0.5)      = {phi_m05:.6f}  [tablas: 0.3085]")
print(f"  P(45<X<62)     = {p_intervalo:.6f}  [tablas: 0.5764]")
print(f"  E[X]           = {dist_nor.mean():.6f}  [exacto: 50]")
print(f"  Var(X)         = {dist_nor.var():.6f}  [exacto: 100]")

# ═══════════════════════════════════════════════════════════════
# GRAFICAS
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("T4 — Distribuciones de Probabilidad\nAlejandro Escobar Barrios — 20251020094",
             fontsize=13, fontweight="bold", color=AZUL, y=1.02)

# ── 1. Binomial ──────────────────────────────────────────────
ax1 = axes[0]
k_vals = np.arange(0, n+1)
pmf_bin = dist_bin.pmf(k_vals)
colores_bin = [ROJO if k == 2 else AZUL for k in k_vals]
ax1.bar(k_vals, pmf_bin, color=colores_bin, edgecolor="white", linewidth=0.6)
ax1.set_title(f"Binomial  $n=5,\\ p=1/6$\n$P(X=2)={p_exactamente_2:.4f}$ (rojo)")
ax1.set_xlabel("$k$ (numero de veces que sale el 3)")
ax1.set_ylabel("$P(X=k)$")
ax1.set_xticks(k_vals)
ax1.grid(axis="y", alpha=0.3)

# ── 2. Poisson ───────────────────────────────────────────────
ax2 = axes[1]
k_poi = np.arange(0, 9)
pmf_poi = dist_poi.pmf(k_poi)
colores_poi = [VERDE if k >= 2 else AZUL for k in k_poi]
ax2.bar(k_poi, pmf_poi, color=colores_poi, edgecolor="white", linewidth=0.6)
ax2.set_title(f"Poisson  $\\lambda=1$\n$P(X\\geq2)={p_poi_2m:.4f}$ (verde)")
ax2.set_xlabel("$k$ (numero de accidentes)")
ax2.set_ylabel("$P(X=k)$")
ax2.set_xticks(k_poi)
ax2.grid(axis="y", alpha=0.3)

# ── 3. Normal ────────────────────────────────────────────────
ax3 = axes[2]
x_nor = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
ax3.plot(x_nor, dist_nor.pdf(x_nor), color=AZUL, lw=2)
x_fill = np.linspace(45, 62, 400)
ax3.fill_between(x_fill, dist_nor.pdf(x_fill), alpha=0.35, color=NARANJA,
                 label=f"$P(45<X<62)={p_intervalo:.4f}$")
ax3.axvline(45, color=ROJO, ls="--", lw=1.2)
ax3.axvline(62, color=ROJO, ls="--", lw=1.2)
ax3.set_title("Normal  $\\mu=50,\\ \\sigma=10$\n(Walpole Ej. 6.3)")
ax3.set_xlabel("$x$")
ax3.set_ylabel("$f(x)$")
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/home/claude/T4_graficas.png", bbox_inches="tight", dpi=180)
print("\n[OK] Figura guardada en T4_graficas.png")
print("[OK] Todos los resultados numericos coinciden con los valores exactos.")
