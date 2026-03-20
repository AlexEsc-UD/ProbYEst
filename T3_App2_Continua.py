"""
T3 – Aplicación 2: Distribución de Probabilidad Conjunta Continua
==================================================================
Contexto  : Reactor químico – tiempo de reacción (X) y temperatura (Y)
Densidad  : f(x,y) = (x + y) / 2,   (x,y) ∈ [0,1] × [0,1]
Herramienta: NumPy + SciPy + Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from scipy import integrate

# ── Configuración de estilo ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 120,
})
AZUL    = "#00467F"
NARANJA = "#E07B39"
VERDE   = "#2E8B57"
ROJO    = "#CC3333"

# ════════════════════════════════════════════════════════════════════════════
# 1. DEFINICIÓN DE LA DENSIDAD
# ════════════════════════════════════════════════════════════════════════════
def f(x, y):
    """Densidad conjunta f(x,y) = x+y en [0,1]×[0,1]."""
    return x + y

def fX(x):
    """Marginal de X: f_X(x) = (2x+1)/4."""
    return x + 0.5

def fY(y):
    """Marginal de Y: f_Y(y) = (2y+1)/4."""
    return y + 0.5

def fY_gX(y, x=0.5):
    """Densidad condicional Y|X=x: f_{Y|X}(y|x) = (x+y)/2 / f_X(x)."""
    return f(x, y) / fX(x)

# ════════════════════════════════════════════════════════════════════════════
# 2. VERIFICACIONES NUMÉRICAS
# ════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  T3 – App 2: Distribución Conjunta Continua")
print("=" * 60)

# Integral doble total  (scipy.dblquad llama f(y,x) internamente)
total, _ = integrate.dblquad(lambda y, x: f(x, y), 0, 1,
                              lambda x: 0, lambda x: 1)
print(f"\n  ∫∫ f(x,y) dx dy = {total:.6f}  {'✓' if abs(total-1)<1e-6 else '✗'}")

# Marginales (integración numérica)
def marg_fX_num(x):
    val, _ = integrate.quad(lambda y: f(x, y), 0, 1)
    return val

def marg_fY_num(y):
    val, _ = integrate.quad(lambda x: f(x, y), 0, 1)
    return val

# ════════════════════════════════════════════════════════════════════════════
# 3. MOMENTOS  (integración numérica)
# ════════════════════════════════════════════════════════════════════════════
EX,  _ = integrate.dblquad(lambda y,x: x * f(x,y), 0, 1, 0, 1)
EX2, _ = integrate.dblquad(lambda y,x: x**2 * f(x,y), 0, 1, 0, 1)
EY,  _ = integrate.dblquad(lambda y,x: y * f(x,y), 0, 1, 0, 1)
EY2, _ = integrate.dblquad(lambda y,x: y**2 * f(x,y), 0, 1, 0, 1)
EXY, _ = integrate.dblquad(lambda y,x: x*y * f(x,y), 0, 1, 0, 1)

VarX  = EX2 - EX**2
VarY  = EY2 - EY**2
CovXY = EXY - EX*EY
rho   = CovXY / np.sqrt(VarX * VarY)

# Valores exactos (analíticos) para comparación
EX_exact   = 7/12
VarX_exact = 11/144
EXY_exact  = 1/3
Cov_exact  = -1/144
rho_exact  = -1/11

print(f"\n── Momentos ─────────────────────────────────────────")
print(f"  E[X]    numérico = {EX:.6f}   exacto = {EX_exact:.6f}")
print(f"  Var(X)  numérico = {VarX:.6f}   exacto = {VarX_exact:.6f}")
print(f"  E[Y]    numérico = {EY:.6f}   exacto = {EX_exact:.6f}")
print(f"  Var(Y)  numérico = {VarY:.6f}   exacto = {VarX_exact:.6f}")
print(f"  E[XY]   numérico = {EXY:.6f}   exacto = {EXY_exact:.6f}")
print(f"  Cov(X,Y)numérico = {CovXY:.6f}   exacto = {Cov_exact:.6f}")
print(f"  ρ(X,Y)  numérico = {rho:.6f}   exacto = {rho_exact:.6f}")

# ════════════════════════════════════════════════════════════════════════════
# 4. INDEPENDENCIA
# ════════════════════════════════════════════════════════════════════════════
# Si fueran independientes: f(x,y) = fX(x)*fY(y) para todo (x,y)
x_test, y_test = 0.3, 0.7
joint      = f(x_test, y_test)
product    = fX(x_test) * fY(y_test)
print(f"\n── Independencia ─────────────────────────────────────")
print(f"  f(0.3, 0.7)            = {joint:.6f}")
print(f"  f_X(0.3) · f_Y(0.7)   = {product:.6f}")
print(f"  Son independientes     : {np.isclose(joint, product)}")

# ════════════════════════════════════════════════════════════════════════════
# 5. PROBABILIDAD P(X≤0.5, Y≤0.5)
# ════════════════════════════════════════════════════════════════════════════
prob, _ = integrate.dblquad(lambda y, x: f(x, y), 0, 0.5,
                             lambda x: 0, lambda x: 0.5)
print(f"\n── Probabilidad ──────────────────────────────────────")
print(f"  P(X≤0.5, Y≤0.5) = {prob:.6f}  exacto = {1/8:.6f}")

# ════════════════════════════════════════════════════════════════════════════
# 6. ESPERANZA CONDICIONAL E[Y | X=0.5]
# ════════════════════════════════════════════════════════════════════════════
EYgX05, _ = integrate.quad(lambda y: y * fY_gX(y, x=0.5), 0, 1)
print(f"\n── Condicional ───────────────────────────────────────")
print(f"  E[Y | X=0.5] = {EYgX05:.6f}  exacto = {7/12:.6f}")
print("=" * 60)

# ════════════════════════════════════════════════════════════════════════════
# 7. FIGURAS
# ════════════════════════════════════════════════════════════════════════════
xx = np.linspace(0, 1, 100)
yy = np.linspace(0, 1, 100)
X, Y = np.meshgrid(xx, yy)
Z = f(X, Y)

fig = plt.figure(figsize=(16, 10))
fig.suptitle("T3 – App 2: Distribución de Probabilidad Conjunta Continua\n"
             r"$f(x,y)=\frac{1}{2}(x+y)$,   $(x,y)\in[0,1]\times[0,1]$",
             fontsize=13, fontweight="bold", color=AZUL)

# ── Figura 1: Superficie 3D ──────────────────────────────────────────────
ax1 = fig.add_subplot(221, projection="3d")
surf = ax1.plot_surface(X, Y, Z, cmap="Blues", alpha=0.9, edgecolor="none")
fig.colorbar(surf, ax=ax1, shrink=0.5, label="f(x,y)")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("f(x,y)")
ax1.set_title("Superficie de la densidad conjunta")
ax1.view_init(elev=30, azim=-50)

# ── Figura 2: Mapa de calor ──────────────────────────────────────────────
ax2 = fig.add_subplot(222)
cf = ax2.contourf(X, Y, Z, levels=20, cmap="Blues")
fig.colorbar(cf, ax=ax2, label="f(x,y)")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_title("Mapa de calor (contour) de f(x,y)")
ax2.set_aspect("equal")

# ── Figura 3: Marginales ─────────────────────────────────────────────────
ax3 = fig.add_subplot(223)
ax3.plot(xx, fX(xx), color=AZUL,    lw=2.5, label=r"$f_X(x) = \frac{2x+1}{4}$")
ax3.plot(yy, fY(yy), color=NARANJA, lw=2.5, ls="--",
         label=r"$f_Y(y) = \frac{2y+1}{4}$")
ax3.fill_between(xx, fX(xx), alpha=0.15, color=AZUL)
ax3.fill_between(yy, fY(yy), alpha=0.15, color=NARANJA)
ax3.set_xlabel("Valor de la variable")
ax3.set_ylabel("Densidad")
ax3.set_title("Densidades Marginales")
ax3.legend()
ax3.set_xlim(0, 1)

# ── Figura 4: Condicional Y | X=0.5 ─────────────────────────────────────
ax4 = fig.add_subplot(224)
y_plot = np.linspace(0, 1, 200)
ax4.plot(y_plot, fY_gX(y_plot, x=0.5), color=VERDE, lw=2.5,
         label=r"$f_{Y|X}(y \mid x=0.5) = 0.5 + y$")
ax4.axvline(EYgX05, color=ROJO, ls="--", lw=1.8,
            label=f"$E[Y|X=0.5]={EYgX05:.4f}$")
ax4.fill_between(y_plot, fY_gX(y_plot, x=0.5), alpha=0.15, color=VERDE)
ax4.set_xlabel("y")
ax4.set_ylabel(r"$f_{Y|X}(y \mid x=0.5)$")
ax4.set_title(r"Densidad Condicional $f_{Y|X}(y \mid X=0.5)$")
ax4.legend(fontsize=9)
ax4.set_xlim(0, 1)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("T3_App2_Continua_graficas.png", bbox_inches="tight", dpi=150)
plt.show()
print("\n  Figura guardada: T3_App2_Continua_graficas.png")
