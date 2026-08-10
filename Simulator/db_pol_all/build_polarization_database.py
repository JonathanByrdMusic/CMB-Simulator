#!/usr/bin/env python3

import json
import math
import time
from datetime import datetime
from pathlib import Path

import camb
import numpy as np


# =========================================================
# USER SETTINGS
# =========================================================

# First test run:
#   100 = calculate 100 cosmologies spread across the cube.
#
# Final production run:
#   change this to None.
#
MAX_MODELS = None

H0 = 67.0
h = H0 / 100.0

TAU = 0.054
AS = 2.1e-9
NS = 0.965

LMAX = 3000

N = 256
PATCH_SIZE_DEG = 10.0
STICK_GRID = 14

RANDOM_SEED = 12345


# =========================================================
# PATHS
#
# Everything temporary stays beside this script.
# Finished website JSON files will be produced later.
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

CACHE_DIR = BASE_DIR / "cache"
FAILURE_DIR = BASE_DIR / "failures"
LOG_DIR = BASE_DIR / "logs"

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FAILURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LOG_FILE = (
    LOG_DIR
    / "build.log"
)


# =========================================================
# Logging
# =========================================================

def log(message=""):

    print(
        message,
        flush=True
    )

    with open(
        LOG_FILE,
        "a"
    ) as f:

        f.write(
            str(message)
            + "\n"
        )


# =========================================================
# Slider values
#
# Omega_b = 0 is intentionally omitted.
#
# Omega_c and Omega_Lambda include zero.
# =========================================================

omega_b_values = [
    round(
        0.025 * i,
        3
    )
    for i in range(
        1,
        41
    )
]

omega_c_values = [
    round(
        0.025 * i,
        3
    )
    for i in range(
        0,
        41
    )
]

omega_l_values = [
    round(
        0.025 * i,
        3
    )
    for i in range(
        0,
        41
    )
]


# =========================================================
# Construct all unique nonzero-baryon cosmologies
# =========================================================

all_cosmologies = []

for Omega_b in omega_b_values:

    for Omega_c in omega_c_values:

        for Omega_l in omega_l_values:

            all_cosmologies.append(
                (
                    Omega_b,
                    Omega_c,
                    Omega_l
                )
            )


TOTAL_COSMOLOGIES = len(
    all_cosmologies
)


# =========================================================
# For the 100-model test, sample evenly through the entire
# cube rather than taking the first 100 consecutive models.
#
# When MAX_MODELS = None, process the whole cube.
# =========================================================

if MAX_MODELS is None:

    cosmologies_to_process = (
        all_cosmologies
    )

else:

    sample_indices = np.linspace(
        0,
        TOTAL_COSMOLOGIES - 1,
        MAX_MODELS,
        dtype=int
    )

    # Remove any accidental duplicates while preserving order
    seen = set()
    unique_indices = []

    for index in sample_indices:

        index = int(index)

        if index not in seen:

            seen.add(index)
            unique_indices.append(
                index
            )

    cosmologies_to_process = [
        all_cosmologies[index]
        for index in unique_indices
    ]


# =========================================================
# Naming helpers
# =========================================================

def value_string(value):

    return (
        f"{value:.3f}"
        .replace(
            ".",
            "p"
        )
    )


def model_name(
    Omega_b,
    Omega_c,
    Omega_l
):

    return (
        "Ob"
        + value_string(Omega_b)
        + "_Oc"
        + value_string(Omega_c)
        + "_Ol"
        + value_string(Omega_l)
    )


# =========================================================
# Flat-sky Fourier grid
# =========================================================

patch_size_rad = np.radians(
    PATCH_SIZE_DEG
)

pixel_size_rad = (
    patch_size_rad
    / N
)

freq = np.fft.fftfreq(
    N,
    d=pixel_size_rad
)

kx = (
    2.0
    * np.pi
    * freq
)

ky = (
    2.0
    * np.pi
    * freq
)

KX, KY = np.meshgrid(
    kx,
    ky
)

ELL_GRID = np.sqrt(
    KX**2
    +
    KY**2
)

PHI = np.arctan2(
    KY,
    KX
)

COS_2PHI = np.cos(
    2.0
    * PHI
)

SIN_2PHI = np.sin(
    2.0
    * PHI
)


# =========================================================
# One fixed random universe
#
# Every cosmology gets exactly the same phases.
# =========================================================

rng = np.random.default_rng(
    RANDOM_SEED
)

noise1 = rng.normal(
    0.0,
    1.0,
    size=(N, N)
)

noise2 = rng.normal(
    0.0,
    1.0,
    size=(N, N)
)

G1 = np.fft.fft2(
    noise1
)

G2 = np.fft.fft2(
    noise2
)


# =========================================================
# Helper: interpolate C_l onto the Fourier grid
# =========================================================

def interpolate_cl(
    ell,
    cl
):

    values = np.interp(
        ELL_GRID.ravel(),
        ell,
        cl,
        left=0.0,
        right=0.0
    )

    return values.reshape(
        ELL_GRID.shape
    )


# =========================================================
# CAMB
#
# Returns unlensed TT, EE, TE.
#
# raw_cl=False means CAMB returns:
#
#     D_l = l(l+1) C_l / 2pi
#
# We convert D_l -> C_l before constructing Q/U.
# =========================================================

def calculate_spectra(
    Omega_b,
    Omega_c,
    Omega_l
):

    ombh2 = (
        Omega_b
        * h**2
    )

    omch2 = (
        Omega_c
        * h**2
    )

    Omega_r = (
        4.165e-5
        / h**2
    )

    Omega_k = (
        1.0
        - Omega_b
        - Omega_c
        - Omega_l
        - Omega_r
    )

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

    # Explicitly disable CMB lensing.
    pars.Want_CMB_lensing = False
    pars.DoLensing = False

    pars.set_for_lmax(
        LMAX,
        lens_potential_accuracy=0
    )

    # Enforce again after set_for_lmax().
    pars.Want_CMB_lensing = False
    pars.DoLensing = False

    camb_results = camb.get_results(
        pars
    )

    cls = (
        camb_results
        .get_unlensed_scalar_cls(
            lmax=LMAX,
            CMB_unit="muK",
            raw_cl=False
        )
    )

    ell = np.arange(
        cls.shape[0],
        dtype=float
    )

    return {
        "ell":
            ell,

        "TT":
            cls[:, 0],

        "EE":
            cls[:, 1],

        "TE":
            cls[:, 3],

        "Omega_k":
            Omega_k
    }


# =========================================================
# Convert CAMB spectra -> Q/U
# =========================================================

def generate_qu(
    spectra
):

    ell = spectra["ell"]

    Dl_TT = np.asarray(
        spectra["TT"],
        dtype=float
    )

    Dl_EE = np.asarray(
        spectra["EE"],
        dtype=float
    )

    Dl_TE = np.asarray(
        spectra["TE"],
        dtype=float
    )


    # -----------------------------------------------------
    # D_l -> C_l
    # -----------------------------------------------------

    Cl_TT = np.zeros_like(
        Dl_TT
    )

    Cl_EE = np.zeros_like(
        Dl_EE
    )

    Cl_TE = np.zeros_like(
        Dl_TE
    )

    valid_ell = (
        ell >= 2
    )

    conversion = (
        2.0
        * np.pi
        /
        (
            ell[valid_ell]
            *
            (
                ell[valid_ell]
                + 1.0
            )
        )
    )

    Cl_TT[valid_ell] = (
        Dl_TT[valid_ell]
        * conversion
    )

    Cl_EE[valid_ell] = (
        Dl_EE[valid_ell]
        * conversion
    )

    Cl_TE[valid_ell] = (
        Dl_TE[valid_ell]
        * conversion
    )


    # -----------------------------------------------------
    # Put spectra onto the flat-sky FFT grid
    # -----------------------------------------------------

    Ctt = interpolate_cl(
        ell,
        Cl_TT
    )

    Cee = interpolate_cl(
        ell,
        Cl_EE
    )

    Cte = interpolate_cl(
        ell,
        Cl_TE
    )


    # -----------------------------------------------------
    # Correlated T/E realization
    # -----------------------------------------------------

    E_fourier = np.zeros(
        (N, N),
        dtype=complex
    )

    valid = (
        Ctt > 0.0
    )

    correlated_part = np.zeros_like(
        Ctt
    )

    correlated_part[valid] = (
        Cte[valid]
        /
        np.sqrt(
            Ctt[valid]
        )
    )

    remaining_variance = (
        np.zeros_like(
            Cee
        )
    )

    remaining_variance[valid] = (
        Cee[valid]
        -
        (
            Cte[valid]**2
            /
            Ctt[valid]
        )
    )

    remaining_variance = (
        np.maximum(
            remaining_variance,
            0.0
        )
    )

    E_fourier[valid] = (
        correlated_part[valid]
        *
        G1[valid]
        +
        np.sqrt(
            remaining_variance[valid]
        )
        *
        G2[valid]
    )


    # -----------------------------------------------------
    # Pure E -> Q/U
    # -----------------------------------------------------

    Q_fourier = (
        E_fourier
        *
        COS_2PHI
    )

    U_fourier = (
        E_fourier
        *
        SIN_2PHI
    )

    Q = np.fft.ifft2(
        Q_fourier
    ).real

    U = np.fft.ifft2(
        U_fourier
    ).real

    Q -= np.mean(Q)
    U -= np.mean(U)

    return Q, U


# =========================================================
# Bin Q/U into the website's 14 x 14 stick grid
#
# We save RAW q/u values here.
#
# Do NOT convert them to stick lengths yet.
# The final global normalization must be calculated only
# after the entire cosmological cube exists.
# =========================================================

def bin_qu(
    Q,
    U
):

    cell_size = (
        N
        / STICK_GRID
    )

    q_cells = []
    u_cells = []
    x_cells = []
    y_cells = []

    for gy in range(
        STICK_GRID
    ):

        for gx in range(
            STICK_GRID
        ):

            x0 = int(
                gx
                * cell_size
            )

            x1 = int(
                (gx + 1)
                * cell_size
            )

            y0 = int(
                gy
                * cell_size
            )

            y1 = int(
                (gy + 1)
                * cell_size
            )

            q_mean = float(
                np.mean(
                    Q[
                        y0:y1,
                        x0:x1
                    ]
                )
            )

            u_mean = float(
                np.mean(
                    U[
                        y0:y1,
                        x0:x1
                    ]
                )
            )

            x = float(
                (x0 + x1)
                / 2.0
            )

            y = float(
                (y0 + y1)
                / 2.0
            )

            q_cells.append(
                q_mean
            )

            u_cells.append(
                u_mean
            )

            x_cells.append(
                x
            )

            y_cells.append(
                y
            )

    return (
        np.asarray(
            x_cells,
            dtype=np.float32
        ),
        np.asarray(
            y_cells,
            dtype=np.float32
        ),
        np.asarray(
            q_cells,
            dtype=np.float32
        ),
        np.asarray(
            u_cells,
            dtype=np.float32
        )
    )


# =========================================================
# Existing cache / failure counters
# =========================================================

existing_cache_files = list(
    CACHE_DIR.glob(
        "*.npz"
    )
)

existing_failure_files = list(
    FAILURE_DIR.glob(
        "*.json"
    )
)

log("")
log("=" * 72)
log("CMB POLARIZATION DATABASE BUILD")
log("=" * 72)

log(
    "Started: "
    + datetime.now().isoformat(
        timespec="seconds"
    )
)

log(
    f"Full nonzero-baryon cube: "
    f"{TOTAL_COSMOLOGIES:,} cosmologies"
)

if MAX_MODELS is None:

    log(
        "Mode: FULL PRODUCTION RUN"
    )

else:

    log(
        f"Mode: TEST RUN "
        f"({len(cosmologies_to_process)} "
        f"cosmologies spread across cube)"
    )

log(
    f"Existing cached models: "
    f"{len(existing_cache_files):,}"
)

log(
    f"Existing failure records: "
    f"{len(existing_failure_files):,}"
)

log("")


# =========================================================
# Main loop
# =========================================================

run_start = time.perf_counter()

new_successes = 0
new_failures = 0
already_cached = 0
already_failed = 0

new_attempt_times = []


for sequence_number, cosmology in enumerate(
    cosmologies_to_process,
    start=1
):

    (
        Omega_b,
        Omega_c,
        Omega_l
    ) = cosmology

    name = model_name(
        Omega_b,
        Omega_c,
        Omega_l
    )

    cache_file = (
        CACHE_DIR
        / f"{name}.npz"
    )

    failure_file = (
        FAILURE_DIR
        / f"{name}.json"
    )


    # -----------------------------------------------------
    # Resume support
    # -----------------------------------------------------

    if cache_file.exists():

        already_cached += 1

        log(
            f"[{sequence_number:>4}/"
            f"{len(cosmologies_to_process)}] "
            f"SKIP cached  {name}"
        )

        continue


    if failure_file.exists():

        already_failed += 1

        log(
            f"[{sequence_number:>4}/"
            f"{len(cosmologies_to_process)}] "
            f"SKIP failed  {name}"
        )

        continue


    # -----------------------------------------------------
    # Calculate this cosmology
    # -----------------------------------------------------

    log(
        f"[{sequence_number:>4}/"
        f"{len(cosmologies_to_process)}] "
        f"RUN          {name}"
    )

    model_start = (
        time.perf_counter()
    )

    try:

        spectra = calculate_spectra(
            Omega_b,
            Omega_c,
            Omega_l
        )

        Q, U = generate_qu(
            spectra
        )

        (
            x_cells,
            y_cells,
            q_cells,
            u_cells
        ) = bin_qu(
            Q,
            U
        )


        # -------------------------------------------------
        # Save the raw binned polarization field.
        #
        # This is our checkpoint.
        # -------------------------------------------------

        np.savez_compressed(
            cache_file,

            omega_b=np.float32(
                Omega_b
            ),

            omega_c=np.float32(
                Omega_c
            ),

            omega_l=np.float32(
                Omega_l
            ),

            omega_k=np.float32(
                spectra[
                    "Omega_k"
                ]
            ),

            x=x_cells,
            y=y_cells,
            q=q_cells,
            u=u_cells
        )


        elapsed = (
            time.perf_counter()
            - model_start
        )

        new_attempt_times.append(
            elapsed
        )

        new_successes += 1

        log(
            f"             SUCCESS "
            f"{elapsed:.2f} s"
        )


    except Exception as error:

        elapsed = (
            time.perf_counter()
            - model_start
        )

        new_attempt_times.append(
            elapsed
        )

        new_failures += 1

        failure_data = {

            "omega_b":
                Omega_b,

            "omega_c":
                Omega_c,

            "omega_lambda":
                Omega_l,

            "error":
                str(error),

            "timestamp":
                datetime.now().isoformat(
                    timespec="seconds"
                )
        }

        with open(
            failure_file,
            "w"
        ) as f:

            json.dump(
                failure_data,
                f,
                indent=2
            )

        log(
            f"             FAILED  "
            f"{elapsed:.2f} s"
        )

        log(
            "             "
            + str(error)
        )


# =========================================================
# Final report
# =========================================================

run_elapsed = (
    time.perf_counter()
    - run_start
)

all_cache_files = list(
    CACHE_DIR.glob(
        "*.npz"
    )
)

all_failure_files = list(
    FAILURE_DIR.glob(
        "*.json"
    )
)

log("")
log("=" * 72)
log("RUN COMPLETE")
log("=" * 72)

log(
    f"New successful models: "
    f"{new_successes:,}"
)

log(
    f"New failures: "
    f"{new_failures:,}"
)

log(
    f"Already cached in this test set: "
    f"{already_cached:,}"
)

log(
    f"Already marked failed in this test set: "
    f"{already_failed:,}"
)

log(
    f"Total cache now contains: "
    f"{len(all_cache_files):,}"
)

log(
    f"Total failure records: "
    f"{len(all_failure_files):,}"
)

log(
    f"Elapsed wall time this run: "
    f"{run_elapsed / 60.0:.2f} minutes"
)


# =========================================================
# Runtime estimate
# =========================================================

if new_attempt_times:

    average_seconds = float(
        np.mean(
            new_attempt_times
        )
    )

    median_seconds = float(
        np.median(
            new_attempt_times
        )
    )

    remaining = (
        TOTAL_COSMOLOGIES
        - len(all_cache_files)
        - len(all_failure_files)
    )

    estimated_seconds = (
        remaining
        * average_seconds
    )

    log("")
    log(
        f"Average time per new model: "
        f"{average_seconds:.2f} s"
    )

    log(
        f"Median time per new model: "
        f"{median_seconds:.2f} s"
    )

    log(
        f"Remaining unique cosmologies: "
        f"{remaining:,}"
    )

    log(
        "Estimated remaining time at "
        "this average:"
    )

    log(
        f"    "
        f"{estimated_seconds / 3600.0:.1f} hours"
    )

    log(
        f"    "
        f"{estimated_seconds / 86400.0:.1f} days"
    )


log("")
log(
    "Cache directory:"
)

log(
    str(
        CACHE_DIR
    )
)

log("")
log(
    "Finished: "
    + datetime.now().isoformat(
        timespec="seconds"
    )
)