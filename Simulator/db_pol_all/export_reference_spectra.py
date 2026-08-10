#!/usr/bin/env python3

import json
from pathlib import Path

import camb
import numpy as np


# =========================================================
# Fiducial cosmology
# =========================================================

OMEGA_B = 0.050
OMEGA_C = 0.275
OMEGA_L = 0.675

H0 = 67.0
h = H0 / 100.0

TAU = 0.054
AS = 2.1e-9
NS = 0.965

LMAX = 3000

OMEGA_R = (
    4.165e-5
    / h**2
)


# =========================================================
# Output
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_FILE = (
    BASE_DIR
    / "reference_spectra.json"
)


# =========================================================
# Derived densities
# =========================================================

ombh2 = (
    OMEGA_B
    * h**2
)

omch2 = (
    OMEGA_C
    * h**2
)

Omega_k = (
    1.0
    - OMEGA_B
    - OMEGA_C
    - OMEGA_L
    - OMEGA_R
)


print()
print("=" * 70)
print("EXPORTING REFERENCE POLARIZATION SPECTRA")
print("=" * 70)

print(
    f"Omega_b      = {OMEGA_B:.3f}"
)

print(
    f"Omega_c      = {OMEGA_C:.3f}"
)

print(
    f"Omega_Lambda = {OMEGA_L:.3f}"
)

print(
    f"Omega_k      = {Omega_k:+.8f}"
)

print()


# =========================================================
# CAMB
# =========================================================

pars = camb.CAMBparams()

pars.set_cosmology(
    H0=H0,
    ombh2=ombh2,
    omch2=omch2,
    omk=Omega_k,
    tau=TAU,
    mnu=0.0
)

pars.InitPower.set_params(
    As=AS,
    ns=NS,
    r=0.0
)

pars.WantTensors = False

# Explicitly disable lensing
pars.Want_CMB_lensing = False
pars.DoLensing = False

pars.set_for_lmax(
    LMAX,
    lens_potential_accuracy=0
)

# Enforce again after set_for_lmax()
pars.Want_CMB_lensing = False
pars.DoLensing = False


results = camb.get_results(
    pars
)


cls = results.get_unlensed_scalar_cls(
    lmax=LMAX,
    CMB_unit="muK",
    raw_cl=False
)


# =========================================================
# Extract spectra
#
# CAMB columns:
#
# 0 = TT
# 1 = EE
# 2 = BB
# 3 = TE
#
# raw_cl=False means:
#
# D_l = l(l+1) C_l / 2pi
# =========================================================

ell = np.arange(
    cls.shape[0],
    dtype=int
)

TT = cls[:, 0]
EE = cls[:, 1]
TE = cls[:, 3]


# =========================================================
# Build JSON
# =========================================================

data = {

    "description":
        "Reference unlensed CMB spectra "
        "for browser polarization generation",

    "cosmology": {

        "omega_b":
            OMEGA_B,

        "omega_c":
            OMEGA_C,

        "omega_lambda":
            OMEGA_L,

        "omega_k":
            float(
                Omega_k
            ),

        "H0":
            H0,

        "tau":
            TAU,

        "As":
            AS,

        "ns":
            NS
    },

    "spectrum_convention":
        "Dl = ell(ell+1)Cl/(2pi)",

    "units":
        "microK^2",

    "lmax":
        LMAX,

    "ell":
        ell.tolist(),

    "TT":
        TT.tolist(),

    "EE":
        EE.tolist(),

    "TE":
        TE.tolist()
}


# =========================================================
# Write file
# =========================================================

with open(
    OUTPUT_FILE,
    "w"
) as f:

    json.dump(
        data,
        f,
        separators=(
            ",",
            ":"
        )
    )


size_mb = (
    OUTPUT_FILE.stat().st_size
    / 1024.0
    / 1024.0
)


print(
    f"Multipoles written: "
    f"{len(ell):,}"
)

print(
    f"ell range: "
    f"{ell[0]} ... {ell[-1]}"
)

print(
    f"File size: "
    f"{size_mb:.3f} MB"
)

print()
print(
    "Saved:"
)

print(
    OUTPUT_FILE
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)