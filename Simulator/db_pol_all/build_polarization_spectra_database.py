#!/usr/bin/env python3

import base64
import json
import time
from pathlib import Path

import camb
import numpy as np


# =========================================================
# Configuration
# =========================================================

TEST_MODE = False
TEST_MODEL_COUNT = 100

STEP = 0.025

OMEGA_B_VALUES = np.round(
    np.arange(0.025, 1.000 + STEP / 2, STEP),
    3
)

OMEGA_C_VALUES = np.round(
    np.arange(0.000, 1.000 + STEP / 2, STEP),
    3
)

OMEGA_L_VALUES = np.round(
    np.arange(0.000, 1.000 + STEP / 2, STEP),
    3
)


# =========================================================
# CAMB settings
# =========================================================

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
# Spectral sampling
# =========================================================

DELTA_ELL = 20

ELL_SAMPLES = np.arange(
    0,
    LMAX + 1,
    DELTA_ELL,
    dtype=np.int32
)

if ELL_SAMPLES[-1] != LMAX:
    ELL_SAMPLES = np.append(
        ELL_SAMPLES,
        LMAX
    )

N_ELL = len(ELL_SAMPLES)


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

CACHE_DIR = BASE_DIR / "spectra_cache"
SLICE_DIR = BASE_DIR / "spectra_slices"

FAILURE_FILE = BASE_DIR / "spectra_failures.json"
MANIFEST_FILE = SLICE_DIR / "manifest.json"

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SLICE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# Helpers
# =========================================================

def value_token(value):
    return (
        f"{value:.3f}"
        .replace(".", "p")
    )


def cache_filename(
    omega_b,
    omega_c,
    omega_l
):
    return (
        CACHE_DIR
        / (
            "Ob"
            + value_token(omega_b)
            + "_Oc"
            + value_token(omega_c)
            + "_Ol"
            + value_token(omega_l)
            + ".npz"
        )
    )


def build_camb_ee_spectrum(
    omega_b,
    omega_c,
    omega_l
):

    ombh2 = (
        omega_b
        * h**2
    )

    omch2 = (
        omega_c
        * h**2
    )

    omega_k = (
        1.0
        - omega_b
        - omega_c
        - omega_l
        - OMEGA_R
    )


    pars = camb.CAMBparams()

    pars.set_cosmology(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        omk=omega_k,
        tau=TAU,
        mnu=0.0
    )

    pars.InitPower.set_params(
        As=AS,
        ns=NS,
        r=0.0
    )

    pars.WantTensors = False

    # Explicitly disable lensing.
    pars.Want_CMB_lensing = False
    pars.DoLensing = False

    pars.set_for_lmax(
        LMAX,
        lens_potential_accuracy=0
    )

    # Enforce again after set_for_lmax().
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


    # CAMB column 1 = EE.
    #
    # raw_cl=False returns
    #
    # D_ell = ell(ell+1) C_ell / (2 pi)
    #
    ee_dl = cls[:, 1]


    sampled_dl = ee_dl[
        ELL_SAMPLES
    ]


    # Convert to C_ell now so the browser does not need
    # to repeat this conversion for every slice.
    sampled_cl = np.zeros(
        N_ELL,
        dtype=np.float64
    )


    for i, ell in enumerate(
        ELL_SAMPLES
    ):

        if ell >= 2:

            sampled_cl[i] = (
                sampled_dl[i]
                * 2.0
                * np.pi
                / (
                    ell
                    * (ell + 1.0)
                )
            )

        else:

            sampled_cl[i] = 0.0


    return (
        sampled_cl.astype(
            np.float32
        ),
        float(omega_k)
    )


def save_model(
    omega_b,
    omega_c,
    omega_l,
    ee_cl,
    omega_k
):

    filename = cache_filename(
        omega_b,
        omega_c,
        omega_l
    )


    np.savez_compressed(
        filename,
        omega_b=np.float32(
            omega_b
        ),
        omega_c=np.float32(
            omega_c
        ),
        omega_l=np.float32(
            omega_l
        ),
        omega_k=np.float32(
            omega_k
        ),
        ee_cl=ee_cl
    )


def load_model(
    omega_b,
    omega_c,
    omega_l
):

    filename = cache_filename(
        omega_b,
        omega_c,
        omega_l
    )


    with np.load(
        filename
    ) as data:

        return (
            data["ee_cl"]
            .astype(
                np.float32
            )
        )


def encode_float32(array):
    """
    Encode little-endian float32 bytes as Base64.
    """

    values = np.asarray(
        array,
        dtype="<f4"
    )

    return (
        base64.b64encode(
            values.tobytes()
        )
        .decode("ascii")
    )


# =========================================================
# Construct complete cosmology list
# =========================================================

ALL_MODELS = []

for omega_b in OMEGA_B_VALUES:

    for omega_c in OMEGA_C_VALUES:

        for omega_l in OMEGA_L_VALUES:

            ALL_MODELS.append(
                (
                    float(omega_b),
                    float(omega_c),
                    float(omega_l)
                )
            )


TOTAL_MODELS = len(
    ALL_MODELS
)


# =========================================================
# Select test models
#
# Spread them across the full flattened cube rather than
# simply taking the first 100.
# =========================================================

if TEST_MODE:

    indices = np.linspace(
        0,
        TOTAL_MODELS - 1,
        TEST_MODEL_COUNT,
        dtype=int
    )

    MODELS_TO_PROCESS = [
        ALL_MODELS[i]
        for i in indices
    ]

else:

    MODELS_TO_PROCESS = (
        ALL_MODELS
    )


# =========================================================
# Load existing failure log
# =========================================================

if FAILURE_FILE.exists():

    try:

        with open(
            FAILURE_FILE,
            "r"
        ) as f:

            failures = json.load(
                f
            )

    except Exception:

        failures = []

else:

    failures = []


# =========================================================
# Stage 1: CAMB cache
# =========================================================

print()
print("=" * 72)
print("POLARIZATION EE SPECTRUM DATABASE")
print("=" * 72)

print(
    f"Test mode: {TEST_MODE}"
)

print(
    f"Delta ell: {DELTA_ELL}"
)

print(
    f"Stored ell values: {N_ELL}"
)

print(
    f"Full cosmology cube: {TOTAL_MODELS:,}"
)

print(
    f"Models selected this run: "
    f"{len(MODELS_TO_PROCESS):,}"
)

print()


new_successes = 0
already_cached = 0
new_failures = 0

start_time = time.time()


for number, (
    omega_b,
    omega_c,
    omega_l
) in enumerate(
    MODELS_TO_PROCESS,
    start=1
):

    filename = cache_filename(
        omega_b,
        omega_c,
        omega_l
    )


    if filename.exists():

        already_cached += 1

        continue


    try:

        ee_cl, omega_k = (
            build_camb_ee_spectrum(
                omega_b,
                omega_c,
                omega_l
            )
        )


        save_model(
            omega_b,
            omega_c,
            omega_l,
            ee_cl,
            omega_k
        )


        new_successes += 1


    except Exception as exc:

        new_failures += 1

        failure = {
            "omega_b":
                omega_b,

            "omega_c":
                omega_c,

            "omega_l":
                omega_l,

            "error":
                str(exc)
        }

        failures.append(
            failure
        )

        print()
        print(
            "FAILED:",
            failure
        )


    if (
        number % 100 == 0
        or number == len(
            MODELS_TO_PROCESS
        )
    ):

        elapsed = (
            time.time()
            - start_time
        )

        print(
            f"{number:,} / "
            f"{len(MODELS_TO_PROCESS):,}"
            f"  | new {new_successes:,}"
            f"  | cached {already_cached:,}"
            f"  | failed {new_failures:,}"
            f"  | {elapsed / 60.0:.2f} min"
        )


with open(
    FAILURE_FILE,
    "w"
) as f:

    json.dump(
        failures,
        f,
        indent=2
    )


elapsed = (
    time.time()
    - start_time
)


print()
print("-" * 72)

print(
    f"New successful: {new_successes:,}"
)

print(
    f"Already cached: {already_cached:,}"
)

print(
    f"New failures: {new_failures:,}"
)

print(
    f"Elapsed: {elapsed / 60.0:.2f} min"
)

print("-" * 72)


# =========================================================
# Stop here in test mode.
#
# We want to verify the 100-model cache before creating
# production slice files.
# =========================================================

if TEST_MODE:

    cache_count = len(
        list(
            CACHE_DIR.glob(
                "*.npz"
            )
        )
    )

    cache_bytes = sum(
        p.stat().st_size
        for p in CACHE_DIR.glob(
            "*.npz"
        )
    )

    print()
    print(
        f"Cache files present: "
        f"{cache_count:,}"
    )

    print(
        f"Cache size: "
        f"{cache_bytes / 1024.0 / 1024.0:.2f} MB"
    )

    print()
    print(
        "TEST MODE COMPLETE"
    )

    print(
        "If the test looks good, set "
        "TEST_MODE = False and run again."
    )

    print()
    raise SystemExit(0)


# =========================================================
# Confirm full cache exists before packaging
# =========================================================

missing_models = []

for (
    omega_b,
    omega_c,
    omega_l
) in ALL_MODELS:

    if not cache_filename(
        omega_b,
        omega_c,
        omega_l
    ).exists():

        missing_models.append(
            (
                omega_b,
                omega_c,
                omega_l
            )
        )


if missing_models:

    print()
    print(
        "Cannot package slices:"
    )

    print(
        f"{len(missing_models):,} "
        "cosmologies are missing from the cache."
    )

    print()

    raise SystemExit(1)


# =========================================================
# Slice metadata
# =========================================================

ENCODING = {
    "type":
        "base64_float32_le",

    "quantity":
        "C_ell_EE",

    "dtype":
        "float32 little-endian",

    "values_per_model":
        N_ELL,

    "bytes_per_model":
        N_ELL * 4
}


def common_slice_header():
    return {
        "format":
            "cmb-polarization-ee-spectrum-slice",

        "format_version":
            1,

        "step":
            STEP,

        "lmax":
            LMAX,

        "delta_ell":
            DELTA_ELL,

        "ell":
            ELL_SAMPLES.tolist(),

        "spectrum_convention":
            "C_ell",

        "units":
            "microK^2",

        "encoding":
            ENCODING
    }


# =========================================================
# Stage 2A: slices varying Omega_b
# =========================================================

print()
print("=" * 72)
print("PACKAGING SPECTRAL SLICES")
print("=" * 72)
print()


written = 0


for omega_c in OMEGA_C_VALUES:

    for omega_l in OMEGA_L_VALUES:

        output_file = (
            SLICE_DIR
            / (
                "Ob_Oc"
                + f"{omega_c:.3f}"
                + "_Ol"
                + f"{omega_l:.3f}"
                + ".json"
            )
        )


        slice_data = (
            common_slice_header()
        )

        slice_data[
            "varying_parameter"
        ] = "omega_b"

        slice_data[
            "fixed_parameters"
        ] = {
            "omega_c":
                float(omega_c),

            "omega_l":
                float(omega_l)
        }


        models = {}


        for omega_b in OMEGA_B_VALUES:

            ee_cl = load_model(
                float(omega_b),
                float(omega_c),
                float(omega_l)
            )

            models[
                f"{omega_b:.3f}"
            ] = {
                "data":
                    encode_float32(
                        ee_cl
                    )
            }


        slice_data[
            "models"
        ] = models


        with open(
            output_file,
            "w"
        ) as f:

            json.dump(
                slice_data,
                f,
                separators=(
                    ",",
                    ":"
                )
            )


        written += 1


# =========================================================
# Stage 2B: slices varying Omega_c
# =========================================================

for omega_b in OMEGA_B_VALUES:

    for omega_l in OMEGA_L_VALUES:

        output_file = (
            SLICE_DIR
            / (
                "Ob"
                + f"{omega_b:.3f}"
                + "_Oc_Ol"
                + f"{omega_l:.3f}"
                + ".json"
            )
        )


        slice_data = (
            common_slice_header()
        )

        slice_data[
            "varying_parameter"
        ] = "omega_c"

        slice_data[
            "fixed_parameters"
        ] = {
            "omega_b":
                float(omega_b),

            "omega_l":
                float(omega_l)
        }


        models = {}


        for omega_c in OMEGA_C_VALUES:

            ee_cl = load_model(
                float(omega_b),
                float(omega_c),
                float(omega_l)
            )

            models[
                f"{omega_c:.3f}"
            ] = {
                "data":
                    encode_float32(
                        ee_cl
                    )
            }


        slice_data[
            "models"
        ] = models


        with open(
            output_file,
            "w"
        ) as f:

            json.dump(
                slice_data,
                f,
                separators=(
                    ",",
                    ":"
                )
            )


        written += 1


# =========================================================
# Stage 2C: slices varying Omega_Lambda
# =========================================================

for omega_b in OMEGA_B_VALUES:

    for omega_c in OMEGA_C_VALUES:

        output_file = (
            SLICE_DIR
            / (
                "Ob"
                + f"{omega_b:.3f}"
                + "_Oc"
                + f"{omega_c:.3f}"
                + "_Ol.json"
            )
        )


        slice_data = (
            common_slice_header()
        )

        slice_data[
            "varying_parameter"
        ] = "omega_l"

        slice_data[
            "fixed_parameters"
        ] = {
            "omega_b":
                float(omega_b),

            "omega_c":
                float(omega_c)
        }


        models = {}


        for omega_l in OMEGA_L_VALUES:

            ee_cl = load_model(
                float(omega_b),
                float(omega_c),
                float(omega_l)
            )

            models[
                f"{omega_l:.3f}"
            ] = {
                "data":
                    encode_float32(
                        ee_cl
                    )
            }


        slice_data[
            "models"
        ] = models


        with open(
            output_file,
            "w"
        ) as f:

            json.dump(
                slice_data,
                f,
                separators=(
                    ",",
                    ":"
                )
            )


        written += 1


# =========================================================
# Manifest
# =========================================================

slice_files = list(
    SLICE_DIR.glob(
        "*.json"
    )
)

slice_bytes = sum(
    p.stat().st_size
    for p in slice_files
)


manifest = {
    "format":
        "cmb-polarization-ee-spectrum-database",

    "format_version":
        1,

    "omega_b": {
        "minimum":
            float(
                OMEGA_B_VALUES[0]
            ),

        "maximum":
            float(
                OMEGA_B_VALUES[-1]
            ),

        "count":
            len(
                OMEGA_B_VALUES
            )
    },

    "omega_c": {
        "minimum":
            float(
                OMEGA_C_VALUES[0]
            ),

        "maximum":
            float(
                OMEGA_C_VALUES[-1]
            ),

        "count":
            len(
                OMEGA_C_VALUES
            )
    },

    "omega_lambda": {
        "minimum":
            float(
                OMEGA_L_VALUES[0]
            ),

        "maximum":
            float(
                OMEGA_L_VALUES[-1]
            ),

        "count":
            len(
                OMEGA_L_VALUES
            )
    },

    "cosmologies":
        TOTAL_MODELS,

    "delta_ell":
        DELTA_ELL,

    "ell":
        ELL_SAMPLES.tolist(),

    "values_per_model":
        N_ELL,

    "slice_files":
        written
}


with open(
    MANIFEST_FILE,
    "w"
) as f:

    json.dump(
        manifest,
        f,
        indent=2
    )


print(
    f"Slice files written: "
    f"{written:,}"
)

print(
    f"Spectral slice database size: "
    f"{slice_bytes / 1024.0 / 1024.0:.2f} MB"
)

print()

print(
    "Saved to:"
)

print(
    SLICE_DIR
)

print()

print("=" * 72)
print("COMPLETE")
print("=" * 72)