#!/usr/bin/env python3
"""
Render a square Gibbs Free-Energy Landscape for PC1 / PC2
– pixel style, red background, colour bar 0-10 kJ mol-1.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ── load PC1 & PC2 (two-column .xvg) ───────────────────────────
xy = np.loadtxt('2Dproj_PC1_PC2.xvg', comments=('#', '@'))
pc1, pc2 = xy.T

# ── constants & visual parameters ──────────────────────────────
T       = 300          # simulation temperature (K)  ← change if needed
kB      = 0.008314     # kJ mol⁻1 K⁻1
BINS    = 60           # grid resolution – lower = bigger squares
BW      = 0.15         # KDE bandwidth – lower = sharper details
Z_MAX   = 10.1         # upper ΔG limit; anything higher turns red

# ── probability density ρ(x,y) via KDE –>  ΔG = –kT ln(ρ/ρmax) ─
kde  = gaussian_kde([pc1, pc2], bw_method=BW)
xi   = np.linspace(pc1.min(), pc1.max(), BINS + 1)
yi   = np.linspace(pc2.min(), pc2.max(), BINS + 1)
xc   = 0.5 * (xi[:-1] + xi[1:])          # cell centres
yc   = 0.5 * (yi[:-1] + yi[1:])
X, Y = np.meshgrid(xc, yc)
rho  = kde([X.ravel(), Y.ravel()]).reshape(X.shape)
dG   = -kB * T * np.log(rho / rho.max())  # zero at global minimum

# ── plot ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 5))

cmap = plt.get_cmap('jet')               # blue → … → red
pcm  = ax.pcolormesh(xi, yi, dG,
                     cmap=cmap,
                     shading='auto',
                     vmin=0, vmax=Z_MAX) # values >Z_MAX become red
pcm.cmap.set_over('red')                 # hard-red background

# Axis / title
ax.set_xlabel('PC1');  ax.set_ylabel('PC2')
ax.set_title('Gibbs Energy Landscape', pad=15, fontsize=14)

# Horizontal colour-bar underneath
cbar = fig.colorbar(pcm, orientation='horizontal',
                    fraction=0.07, pad=0.18, extend='max')
cbar.set_label('G (kJ/mol)')
cbar.set_ticks([0, Z_MAX])

# Optional subtle grid like the example
ax.set_xticks(xi, minor=True)
ax.set_yticks(yi, minor=True)
ax.grid(which='minor', color='white', lw=0.3)

plt.tight_layout()
plt.savefig('GEL_square_style.png', dpi=300)
plt.show()
