"""
T3 – Aplicación 1: Probabilidad Conjunta Discreta
==================================================
Contexto : Defectos de producción en turno mañana (X) y turno tarde (Y)
Variables: X, Y ∈ {0, 1, 2}
Herramienta: NumPy + Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401

# ── 0. Configuración de estilo ───────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 120,
})
AZUL = "#00467F"
NARANJA = "#E07B39"
VERDE = "#2E8B57"

# ════════════════════════════════════════════════════════════════════════════
# 1. TABLA DE PROBABILIDAD CONJUNTA
# ════════════════════════════════════════════════════════════════════════════
# Filas → X = {0,1,2}   Columnas → Y = {0,1,2}
P = np.array([
    [0.30, 0.10, 0.05],   # X=0
    [0.15, 0.20, 0.05],   # X=1
    [0.05, 0.05, 0.05],   # X=2
])

x_vals = np.array([0, 1, 2])
y_vals = np.array([0, 1, 2])

# ════════════════════════════════════════════════════════════════════════════
# 2. VERIFICACIONES BÁSICAS
# ════════════════════════════════════════════════════════════════════════════
suma_total = P.sum()
print("=" * 55)
print("  T3 – App 1: Distribución Conjunta Discreta")
print("=" * 55)
print(f"\n  Suma total de probabilidades : {suma_total:.4f}  {'✓' if abs(suma_total-1)<1e-9 else '✗'}")
print(f"  Todas las entradas ≥ 0       : {'✓' if (P >= 0).all() else '✗'}")

# ════════════════════════════════════════════════════════════════════════════
# 3. MARGINALES
# ════════════════════════════════════════════════════════════════════════════
pX = P.sum(axis=1)   # suma sobre columnas (Y)
pY = P.sum(axis=0)   # suma sobre filas (X)

print("\n── Distribuciones Marginales ─────────────────────")
print("  x      p_X(x)")
for x, px in zip(x_vals, pX):
    print(f"  {x}      {px:.4f}")
print()
print("  y      p_Y(y)")
for y, py in zip(y_vals, pY):
    print(f"  {y}      {py:.4f}")

# ════════════════════════════════════════════════════════════════════════════
# 4. INDEPENDENCIA
# ════════════════════════════════════════════════════════════════════════════
P_ind = np.outer(pX, pY)       # producto cartesiano de marginales
independientes = np.allclose(P, P_ind, atol=1e-9)
print("\n── Independencia ─────────────────────────────────")
print(f"  P(x,y) == p_X(x)·p_Y(y) para todo (x,y): {independientes}")
print("  Tabla producto de marginales:")
for i in range(3):
    for j in range(3):
        print(f"    P_ind({i},{j}) = {P_ind[i,j]:.4f}  vs  p({i},{j}) = {P[i,j]:.4f}"
              + ("  ✓" if abs(P[i,j]-P_ind[i,j]) < 1e-9 else "  ✗"))

# ════════════════════════════════════════════════════════════════════════════
# 5. ESPERANZAS Y VARIANZAS
# ════════════════════════════════════════════════════════════════════════════
EX  = np.dot(x_vals, pX)
EX2 = np.dot(x_vals**2, pX)
VarX = EX2 - EX**2

EY  = np.dot(y_vals, pY)
EY2 = np.dot(y_vals**2, pY)
VarY = EY2 - EY**2

print("\n── Momentos ──────────────────────────────────────")
print(f"  E[X]    = {EX:.4f}")
print(f"  Var(X)  = {VarX:.4f}")
print(f"  E[Y]    = {EY:.4f}")
print(f"  Var(Y)  = {VarY:.4f}")

# ════════════════════════════════════════════════════════════════════════════
# 6. COVARIANZA Y CORRELACIÓN
# ════════════════════════════════════════════════════════════════════════════
EXY = sum(x_vals[i]*y_vals[j]*P[i,j]
          for i in range(3) for j in range(3))
CovXY = EXY - EX*EY
rho   = CovXY / np.sqrt(VarX * VarY)

print(f"\n  E[XY]        = {EXY:.4f}")
print(f"  Cov(X,Y)     = {CovXY:.4f}")
print(f"  ρ(X,Y)       = {rho:.4f}")

# ════════════════════════════════════════════════════════════════════════════
# 7. DISTRIBUCIÓN CONDICIONAL  Y | X=1
# ════════════════════════════════════════════════════════════════════════════
x_cond = 1
P_YgX1 = P[x_cond, :] / pX[x_cond]
EY_gX1 = np.dot(y_vals, P_YgX1)

print(f"\n── Distribución Condicional  Y | X={x_cond} ─────────────")
for y, p in zip(y_vals, P_YgX1):
    print(f"  P(Y={y} | X={x_cond}) = {p:.4f}")
print(f"  E[Y | X={x_cond}]       = {EY_gX1:.4f}")
print("=" * 55)

# ════════════════════════════════════════════════════════════════════════════
# 8. FIGURAS
# ════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(16, 5))
fig.suptitle("T3 – App 1: Distribución de Probabilidad Conjunta Discreta",
             fontsize=14, fontweight="bold", color=AZUL, y=1.01)

# ── Fig 1: Barras 3D ─────────────────────────────────────────────────────
ax1 = fig.add_subplot(131, projection="3d")
xpos, ypos = np.meshgrid(x_vals, y_vals, indexing="ij")
xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros_like(xpos, dtype=float)
dz   = P.flatten()

colors = plt.cm.Blues(dz / dz.max())
ax1.bar3d(xpos - 0.3, ypos - 0.3, zpos,
          0.6, 0.6, dz, color=colors, alpha=0.85, edgecolor="white")
ax1.set_xlabel("X (turno mañana)", labelpad=8)
ax1.set_ylabel("Y (turno tarde)", labelpad=8)
ax1.set_zlabel("P(X=x, Y=y)")
ax1.set_title("Distribución Conjunta\nP(X, Y)", pad=10)
ax1.set_xticks([0, 1, 2])
ax1.set_yticks([0, 1, 2])

# ── Fig 2: Marginales ────────────────────────────────────────────────────
ax2 = fig.add_subplot(132)
width = 0.3
ax2.bar(x_vals - width/2, pX, width, label="$p_X(x)$",
        color=AZUL, alpha=0.85)
ax2.bar(y_vals + width/2, pY, width, label="$p_Y(y)$",
        color=NARANJA, alpha=0.85)
ax2.set_xlabel("Valor de la variable")
ax2.set_ylabel("Probabilidad")
ax2.set_title("Distribuciones Marginales")
ax2.set_xticks([0, 1, 2])
ax2.legend()
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax2.set_ylim(0, 0.65)
for xi, px in zip(x_vals, pX):
    ax2.text(xi - width/2, px + 0.01, f"{px:.2f}", ha="center", fontsize=9)
for yi, py in zip(y_vals, pY):
    ax2.text(yi + width/2, py + 0.01, f"{py:.2f}", ha="center", fontsize=9)

# ── Fig 3: Condicional Y | X=1 ──────────────────────────────────────────
ax3 = fig.add_subplot(133)
ax3.bar(y_vals, P_YgX1, color=VERDE, alpha=0.85, width=0.5)
ax3.axhline(EY_gX1, color="red", ls="--", lw=1.5,
            label=f"$E[Y|X=1]={EY_gX1:.3f}$")
ax3.set_xlabel("y")
ax3.set_ylabel("P(Y=y | X=1)")
ax3.set_title(f"Distribución Condicional\n$P(Y=y \\mid X=1)$")
ax3.set_xticks([0, 1, 2])
ax3.legend(fontsize=9)
ax3.set_ylim(0, 0.65)
for yi, p in zip(y_vals, P_YgX1):
    ax3.text(yi, p + 0.01, f"{p:.3f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("T3_App1_Discreta_graficas.png", bbox_inches="tight", dpi=150)
plt.show()
print("\n  Figura guardada: T3_App1_Discreta_graficas.png")
