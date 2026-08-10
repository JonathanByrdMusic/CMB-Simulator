#!/usr/bin/env python3

import json
from pathlib import Path


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "reference_spectra.json"


# =========================================================
# Sampling intervals to test
# =========================================================

SPACINGS = [1, 5, 10, 20]


# =========================================================
# Build one downsampled spectrum file
# =========================================================

def build_sampled_version(data, spacing):

    ell = data["ell"]
    TT = data["TT"]
    EE = data["EE"]
    TE = data["TE"]

    sampled_ell = []
    sampled_TT = []
    sampled_EE = []
    sampled_TE = []

    for i in range(0, len(ell), spacing):

        sampled_ell.append(ell[i])
        sampled_TT.append(TT[i])
        sampled_EE.append(EE[i])
        sampled_TE.append(TE[i])

    # Always include the final ell value.
    if sampled_ell[-1] != ell[-1]:

        sampled_ell.append(ell[-1])
        sampled_TT.append(TT[-1])
        sampled_EE.append(EE[-1])
        sampled_TE.append(TE[-1])

    output = {
        "description":
            "Downsampled reference unlensed CMB spectra "
            "for browser polarization-generation tests",

        "source":
            INPUT_FILE.name,

        "sampling": {
            "delta_ell": spacing,
            "original_points": len(ell),
            "sampled_points": len(sampled_ell)
        },

        "cosmology":
            data["cosmology"],

        "spectrum_convention":
            data["spectrum_convention"],

        "units":
            data["units"],

        "lmax":
            data["lmax"],

        "ell":
            sampled_ell,

        "TT":
            sampled_TT,

        "EE":
            sampled_EE,

        "TE":
            sampled_TE
    }

    return output


# =========================================================
# Main
# =========================================================

print()
print("=" * 70)
print("SPECTRAL SAMPLING TEST")
print("=" * 70)
print()

print("Reading:")
print(INPUT_FILE)
print()


with open(INPUT_FILE, "r") as f:
    reference = json.load(f)


print(
    "Original multipoles:",
    len(reference["ell"])
)

print(
    "ell range:",
    reference["ell"][0],
    "...",
    reference["ell"][-1]
)

print()


for spacing in SPACINGS:

    sampled = build_sampled_version(
        reference,
        spacing
    )

    output_file = (
        BASE_DIR
        / (
            "reference_spectra_dl"
            + str(spacing)
            + ".json"
        )
    )

    with open(output_file, "w") as f:

        json.dump(
            sampled,
            f,
            separators=(",", ":")
        )

    size_kb = (
        output_file.stat().st_size
        / 1024.0
    )

    print(
        "delta ell =",
        spacing
    )

    print(
        "  points:",
        sampled["sampling"]["sampled_points"]
    )

    print(
        "  size:",
        f"{size_kb:.1f} KB"
    )

    print(
        "  file:",
        output_file.name
    )

    print()


print("=" * 70)
print("DONE")
print("=" * 70)