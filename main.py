"""
Pomeron and rho-exchange residues in elastic pi-p scattering
=========================================================================
CSV FORMAT:
  plab_GeV,sigma_mb,sigma_err_mb
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok=True)

# =============================================
# PHYSICS CONSTANTS
# =============================================
mN   = 0.9383
mpi  = 0.1396
nu0  = 1.0

aP   = 1.08
arho = 0.55

aPp   = 0.25
arhop = 0.85

bP = 2.0

# =============================================
# KINEMATIC HELPERS
# =============================================
def plab_to_s(plab):
    E_pi = np.sqrt(plab**2 + mpi**2)
    return mN**2 + mpi**2 + 2.0 * mN * E_pi

def s_to_sqrts(s):
    return np.sqrt(s)

def s_to_nu(s):
    return (s - mN**2 - mpi**2) / (2.0 * mN)

def plab_to_sqrts(plab):
    return s_to_sqrts(plab_to_s(plab))

def plab_to_nu(plab):
    return s_to_nu(plab_to_s(plab))

# =============================================
# REGGE CROSS SECTIONS
# =============================================
def sigma_plus_regge(nu, betaP):
    return betaP * (nu / nu0) ** (aP - 1.0)

def sigma_minus_regge(nu, betarho):
    return betarho * (nu / nu0) ** (arho - 1.0)

def sigma_piplus_regge(nu, betaP, betarho):
    return sigma_plus_regge(nu, betaP) - sigma_minus_regge(nu, betarho)

def sigma_piminus_regge(nu, betaP, betarho):
    return sigma_plus_regge(nu, betaP) + sigma_minus_regge(nu, betarho)

# =============================================
# RESIDUE EXTRACTION
# =============================================
def extract_residues(nu_star, sigma_plus_star, sigma_minus_star):
    betaP = sigma_plus_star * (nu_star / nu0) ** (1.0 - aP)
    betarho = sigma_minus_star * (nu_star / nu0) ** (1.0 - arho)
    return betaP, betarho

# =============================================
# FORWARD ELASTIC SLOPE
# =============================================
def forward_slope(nu):
    return 2.0 * aPp * np.log(nu / nu0) + 2.0 * bP

# =============================================
# EMBEDDED DATA ARRAYS
# =============================================
_piplus = np.array([])

_piminus = np.array([])


# =============================================
# LOAD CSV DATA
# =============================================
DATA_DIR = "./data/csv/"

def load_or_use_embedded(filename, embedded_array):
    path = os.path.join(DATA_DIR, filename)

    if os.path.exists(path):
        print(f"  Loading from CSV: {path}")
        df = pd.read_csv(path)
        return np.column_stack([
            df["plab_GeV"].values,
            df["sigma_mb"].values,
            df["sigma_err_mb"].values
        ])
    else:
        print(f"  File '{filename}' not found - using embedded representative data.")
        return embedded_array

print("Loading data...")
pdg_piplus = load_or_use_embedded("pdg_piplus.csv", _piplus)
pdg_piminus = load_or_use_embedded("pdg_piminus.csv", _piminus)

# =============================================
# UNPACK DATA
# =============================================
hp_plab, hp_sig, hp_err = pdg_piplus.T
hm_plab, hm_sig, hm_err = pdg_piminus.T

# =============================================
# KINEMATIC CONVERSIONS
# =============================================
hp_sqrts = plab_to_sqrts(hp_plab)
hm_sqrts = plab_to_sqrts(hm_plab)

hp_nu = plab_to_nu(hp_plab)
hm_nu = plab_to_nu(hm_plab)


# =============================================
# TASK 1 - CROSS SECTIONS
# =============================================
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# PLOT 1 - Total Cross Section
ax = axes[0]
ax.set_title(
    "Measured total cross sections for elastic "
    r"$\pi^{+}p$ and $\pi^{-}p$ scattering",
    fontsize=10
)

ax.errorbar(
    hp_sqrts,
    hp_sig,
    yerr=hp_err,
    fmt='.',
    ms=4,
    color='navy',
    label=r'$\sigma^{\pi^+p}$ PDG',
    capsize=2
)

ax.errorbar(
    hm_sqrts,
    hm_sig,
    yerr=hm_err,
    fmt='s',
    ms=3,
    color='darkred',
    label=r'$\sigma^{\pi^-p}$ PDG',
    capsize=2
)

ax.set_xscale('log')
ax.set_xlabel(r'$\sqrt{s}$ [GeV]')
ax.set_ylabel(r'$\sigma_{\rm tot}$ [mb]')
ax.set_xlim(1, 10)
ax.set_ylim(0, 250)

ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)


# PLOT 2 - Isospin combination
# Interpolating pi− onto pi+ grid
hm_interp = np.interp(hp_sqrts, hm_sqrts, hm_sig)

sig_plus_pdg  = 0.5 * (hm_interp + hp_sig)
sig_minus_pdg = 0.5 * (hm_interp - hp_sig)
sqrts_pdg = hp_sqrts

ax2 = axes[1]

ax2.set_title(
    "Isospin combinations: Pomeron and rho-exchange separation\n"
    r"$\sigma^{(+)} = \frac{1}{2}(\sigma^{\pi^- p} + \sigma^{\pi^+ p})$   "
    r"$\sigma^{(-)} = \frac{1}{2}(\sigma^{\pi^- p} - \sigma^{\pi^+ p})$",
    fontsize=10
)

ax2.errorbar(
    sqrts_pdg,
    sig_plus_pdg,
    fmt='o',
    ms=3,
    color='navy',
    label=r'$\sigma^{(+)}$ PDG'
)

ax2.errorbar(
    sqrts_pdg,
    sig_minus_pdg,
    fmt='s',
    ms=3,
    color='darkred',
    label=r'$\sigma^{(-)}$ PDG'
)

ax2.set_yscale('log')

ax2.set_xlabel(r'$\sqrt{s}$ [GeV]')
ax2.set_ylabel(r'$\sigma$ [mb]')

ax2.set_xlim(1, 10)
ax2.set_ylim(0.05, 300)

ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(
    "outputs/fig1_cross_sections.png",
    dpi=150,
    bbox_inches='tight'
)

print("\nSaved outputs/fig1_cross_sections.png")


# =============================================
# TASK 3 - RESIDUE EXTRACTION
# =============================================
matching_sqrts = [2.0, 2.5, 3.0]

print("\n" + "="*60)
print("TASK 3 - Residue extraction")
print("="*60)

idx = np.argsort(sqrts_pdg)

sqrts_pdg = sqrts_pdg[idx]
sig_plus_pdg  = sig_plus_pdg[idx]
sig_minus_pdg  = sig_minus_pdg[idx]

results = []

for sqrts_star in matching_sqrts:

    s_star = sqrts_star**2
    nu_star = s_to_nu(s_star)
    sp_star = np.interp(sqrts_star, sqrts_pdg, sig_plus_pdg)
    sm_star = np.interp(sqrts_star, sqrts_pdg, sig_minus_pdg)

    bP, brho = extract_residues(nu_star, sp_star, sm_star)

    results.append((bP, brho))

    print(
        f"sqrt(s*)={sqrts_star:.1f} GeV   "
        f"beta_P={bP:.2f} mb   "
        f"beta_rho={brho:.2f} mb"
    )

betaP_mean = np.mean([r[0] for r in results])
betarho_mean = np.mean([r[1] for r in results])

print(f"\nMean beta_P   = {betaP_mean:.2f} mb")
print(f"Mean beta_rho = {betarho_mean:.2f} mb")


# =============================================
# TASK 4 - REGGE PREDICTIONS
# =============================================
print("\n" + "="*60)
print("TASK 4 - Regge predictions")
print("="*60)

sqrts_pred = np.array([5.0, 10.0, 20.0, 50.0])

nu_pred = s_to_nu(sqrts_pred**2)

sig_piplus_pred = sigma_piplus_regge(
    nu_pred,
    betaP_mean,
    betarho_mean
)

sig_piminus_pred = sigma_piminus_regge(
    nu_pred,
    betaP_mean,
    betarho_mean
)

print(
    f"\n{'sqrt(s)':>10} "
    f"{'pi+p pred':>12} "
    f"{'pi-p pred':>12}"
)

for i, sq in enumerate(sqrts_pred):
    print(
        f"{sq:>10.1f} "
        f"{sig_piplus_pred[i]:>12.2f} "
        f"{sig_piminus_pred[i]:>12.2f}"
    )

print(
    f"\n{'sqrt(s)':>10} "
    f"{'pi+p data':>12} "
    f"{'pi-p data':>12} "
    f"{'dev+':>10} "
    f"{'dev-':>10}"
)

for i, sq in enumerate(sqrts_pred):
    sig_p_data = np.interp(sq,hp_sqrts, hp_sig)
    sig_m_data = np.interp(sq, hm_sqrts, hm_sig)

    dev_p = ((sig_piplus_pred[i] - sig_p_data) / sig_p_data * 100)
    dev_m = ((sig_piminus_pred[i] - sig_m_data) / sig_m_data * 100)

    print(
        f"{sq:>10.1f} "
        f"{sig_p_data:>12.2f} "
        f"{sig_m_data:>12.2f} "
        f"{dev_p:>9.1f}% "
        f"{dev_m:>9.1f}%"
    )

fig2, ax3 = plt.subplots(figsize=(10, 6))

sqrts_curve = np.linspace(2.5, 200, 1000)
nu_curve = s_to_nu(sqrts_curve**2)

sig_pp_curve = sigma_piplus_regge(
    nu_curve,
    betaP_mean,
    betarho_mean
)

sig_pm_curve = sigma_piminus_regge(
    nu_curve,
    betaP_mean,
    betarho_mean
)

ax3.plot(sqrts_curve, sig_pp_curve, 'b-', lw=2, label=r'Regge $\pi^+p$')
ax3.plot(sqrts_curve, sig_pm_curve, 'r-', lw=2, label=r'Regge $\pi^-p$')

ax3.errorbar(
    hp_sqrts,
    hp_sig,
    yerr=hp_err,
    fmt='.',
    ms=4,
    color='navy',
    label=r'PDG $\pi^+p$',
    capsize=2
)

ax3.errorbar(
    hm_sqrts,
    hm_sig,
    yerr=hm_err,
    fmt='s',
    ms=3,
    color='darkred',
    label=r'PDG $\pi^-p$',
    capsize=2
)

ax3.set_xscale('log')

ax3.set_xlabel(r'$\sqrt{s}$ [GeV]')
ax3.set_ylabel(r'$\sigma_{\rm tot}$ [mb]')

ax3.legend()
ax3.grid(True, alpha=0.3, which='both')

ax3.set_xlim([1, 300])

plt.tight_layout()
plt.savefig(
    "outputs/fig2_regge_prediction.png",
    dpi=150,
    bbox_inches='tight'
)

print("\nSaved outputs/fig2_regge_prediction.png")


# =============================================
# TASK 5 - FORWARD ELASRIC SLOPE
# =============================================
print("\n" + "="*60)
print("TASK 5 - Forward elastic slope")
print("="*60)

sqrts_B = 19.0
nu_B = s_to_nu(sqrts_B**2)
B_pred = forward_slope(nu_B)

print(f"\nsqrt(s) = {sqrts_B} GeV")
print(f"nu      = {nu_B:.2f} GeV")
print(f"B(s)    = {B_pred:.2f} GeV^-2")

plt.show()