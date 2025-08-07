#!/usr/bin/env python3
# gel_surface_3D_clip10.py
"""
Paysage 3-D de l’énergie libre : ΔG limité à 0-10 kJ mol⁻¹

• Coupe (« clipping ») stricte : dG[dG > 10] = 10
• Échelle couleur & axe Z verrouillés à 0-10
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D    # noqa: F401
from scipy.stats import gaussian_kde
from pathlib import Path

# ── RÉGLAGES UTILISATEUR ─────────────────────────────────────────
INFILE = '2Dproj_PC1_PC2.xvg'     # votre fichier PC1 PC2
TEMP   = 300.0                    # K
kB     = 0.008314                 # kJ mol⁻¹ K⁻¹
BINS   = 100                      # finesse du maillage
BWIDTH = 0.15                     # bande KDE
Z_MAX  = 10.0                     # plafond ΔG (kJ mol⁻¹)

# ── CHARGEMENT ───────────────────────────────────────────────────
pc1, pc2 = np.loadtxt(INFILE, comments=('#', '@')).T

# ── DENSITÉ → ΔG ─────────────────────────────────────────────────
kde   = gaussian_kde([pc1, pc2], bw_method=BWIDTH)
xi, yi = [np.linspace(v.min(), v.max(), BINS + 1) for v in (pc1, pc2)]
xc, yc = [0.5 * (v[:-1] + v[1:]) for v in (xi, yi)]
X, Y   = np.meshgrid(xc, yc)
rho    = kde([X.ravel(), Y.ravel()]).reshape(X.shape)
dG     = -kB * TEMP * np.log(rho / rho.max())

# ── CLIPPING STRICT ─────────────────────────────────────────────
dG = np.clip(dG, 0, Z_MAX)        # toutes les valeurs >10 deviennent 10

# ── FIGURE ───────────────────────────────────────────────────────
plt.rcParams.update({"font.size": 10, "axes.labelweight": "bold"})
fig = plt.figure(figsize=(6, 5), dpi=600)
ax  = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(
    X, Y, dG,
    cmap='jet', vmin=0, vmax=Z_MAX,
    rstride=1, cstride=1, edgecolor='k', linewidth=0.2
)
ax.set_zlim(0, Z_MAX)             # verrouillage axe Z
surf.set_clim(0, Z_MAX)           # verrouillage palette

# Étiquettes & titre
ax.set_xlabel('PC1', fontstyle='italic')
ax.set_ylabel('PC2', fontstyle='italic')
ax.set_zlabel('G (kJ mol$^{-1}$)')
ax.set_title('A', loc='left', fontsize=16, pad=10)
ax.view_init(elev=28, azim=-40)

# Barre de couleurs
cbar = fig.colorbar(surf, ax=ax, orientation='vertical',
                    fraction=0.045, pad=0.06, shrink=0.8)
cbar.set_ticks(np.arange(0, Z_MAX + 2, 2))
cbar.set_label('ΔG (kJ mol⁻¹)', weight='bold')

plt.tight_layout()

# ── ENREGISTREMENT ──────────────────────────────────────────────
out_png = f"GEL_surface_3D_{Path(INFILE).stem}_clip10.png"
plt.savefig(out_png)
print(f"Figure enregistrée : {out_png}")

plt.show()
