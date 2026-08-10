#!/usr/bin/env python3

import base64
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np


# =========================================================
# USER SETTINGS
# =========================================================

# First run:
#     True  -> build only the three familiar reference slices
#
# Production run:
#     False -> build all 4,961 slice files
#
TEST_MODE = False

# If False, an existing completed slice file is skipped.
OVERWRITE_EXISTING_SLICES = False

STEP = 0.025

STICK_GRID = 14
VECTORS_PER_MODEL = STICK_GRID * STICK_GRID

FORMAT_VERSION = 2


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

CACHE_DIR = BASE_DIR / "cache"

OUTPUT_DIR = BASE_DIR / "slices"

LOG_DIR = BASE_DIR / "logs"

NORMALIZATION_FILE = (
    BASE_DIR
    / "polarization_normalization.json"
)

MANIFEST_FILE = (
    OUTPUT_DIR
    / "manifest.json"
)

LOG_FILE = (
    LOG_DIR
    / "slice_build.log"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
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
# Omega_b intentionally begins at 0.025.
#
# Omega_c and Omega_Lambda include zero.
# =========================================================

omega_b_values = [
    round(
        STEP * i,
        3
    )
    for i in range(
        1,
        41
    )
]

omega_c_values = [
    round(
        STEP * i,
        3
    )
    for i in range(
        0,
        41
    )
]

omega_l_values = [
    round(
        STEP * i,
        3
    )
    for i in range(
        0,
        41
    )
]


TOTAL_COSMOLOGIES = (
    len(omega_b_values)
    *
    len(omega_c_values)
    *
    len(omega_l_values)
)


# =========================================================
# Expected slice counts
# =========================================================

N_BARYON_SLICES = (
    len(omega_c_values)
    *
    len(omega_l_values)
)

N_CDM_SLICES = (
    len(omega_b_values)
    *
    len(omega_l_values)
)

N_LAMBDA_SLICES = (
    len(omega_b_values)
    *
    len(omega_c_values)
)

TOTAL_SLICES = (
    N_BARYON_SLICES
    +
    N_CDM_SLICES
    +
    N_LAMBDA_SLICES
)


# =========================================================
# Naming helpers
# =========================================================

def cache_value_string(value):

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
        + cache_value_string(
            Omega_b
        )
        + "_Oc"
        + cache_value_string(
            Omega_c
        )
        + "_Ol"
        + cache_value_string(
            Omega_l
        )
    )


def model_path(
    Omega_b,
    Omega_c,
    Omega_l
):

    return (
        CACHE_DIR
        /
        (
            model_name(
                Omega_b,
                Omega_c,
                Omega_l
            )
            + ".npz"
        )
    )


def json_key(value):

    return f"{value:.3f}"


# =========================================================
# Verify cache
# =========================================================

def verify_cache():

    cache_files = list(
        CACHE_DIR.glob(
            "*.npz"
        )
    )

    count = len(
        cache_files
    )

    log(
        f"Cache files found: "
        f"{count:,}"
    )

    log(
        f"Cache files expected: "
        f"{TOTAL_COSMOLOGIES:,}"
    )

    if count != TOTAL_COSMOLOGIES:

        raise RuntimeError(
            "Cache is incomplete. "
            f"Expected {TOTAL_COSMOLOGIES:,} "
            f"models but found {count:,}."
        )

    # Check one file to verify the expected grid.
    with np.load(
        cache_files[0]
    ) as data:

        q = data["q"]
        u = data["u"]

        if (
            q.size
            != VECTORS_PER_MODEL
            or
            u.size
            != VECTORS_PER_MODEL
        ):

            raise RuntimeError(
                "Unexpected polarization grid size. "
                f"Expected {VECTORS_PER_MODEL} "
                "Q/U samples per model."
            )

    return cache_files


# =========================================================
# GLOBAL NORMALIZATION
#
# We calculate:
#
#     P = sqrt(Q^2 + U^2)
#
# for every one of the 196 grid positions in every
# cosmology.
#
# The global 95th percentile of P becomes the ONE
# normalization scale used everywhere in the simulator.
# =========================================================

def calculate_global_normalization(
    cache_files
):

    expected_samples = (
        len(cache_files)
        *
        VECTORS_PER_MODEL
    )

    # About 53 MB as float32 for the complete database.
    amplitudes = np.empty(
        expected_samples,
        dtype=np.float32
    )

    cursor = 0

    start = time.perf_counter()

    log("")
    log(
        "Calculating global polarization "
        "normalization..."
    )

    for index, path in enumerate(
        cache_files,
        start=1
    ):

        with np.load(
            path
        ) as data:

            q = np.asarray(
                data["q"],
                dtype=np.float32
            )

            u = np.asarray(
                data["u"],
                dtype=np.float32
            )

        p = np.hypot(
            q,
            u
        )

        next_cursor = (
            cursor
            + p.size
        )

        amplitudes[
            cursor:next_cursor
        ] = p

        cursor = next_cursor

        if (
            index % 5000 == 0
            or
            index == len(cache_files)
        ):

            log(
                f"  scanned "
                f"{index:,} / "
                f"{len(cache_files):,} "
                f"models"
            )

    if cursor != expected_samples:

        raise RuntimeError(
            "Global normalization sample "
            "count does not match expectation."
        )

    global_p95 = float(
        np.percentile(
            amplitudes,
            95.0
        )
    )

    global_max = float(
        np.max(
            amplitudes
        )
    )

    mean_p = float(
        np.mean(
            amplitudes
        )
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    if (
        not np.isfinite(
            global_p95
        )
        or global_p95 <= 0.0
    ):

        raise RuntimeError(
            "Invalid global polarization "
            "normalization."
        )

    normalization = {

        "format_version":
            FORMAT_VERSION,

        "description":
            "Global normalization for "
            "the CMB polarization database",

        "method":
            "95th percentile of "
            "sqrt(Q^2 + U^2)",

        "global_p95":
            global_p95,

        "global_max":
            global_max,

        "mean_p":
            mean_p,

        "models":
            len(
                cache_files
            ),

        "vectors_per_model":
            VECTORS_PER_MODEL,

        "samples":
            expected_samples,

        "created":
            datetime.now().isoformat(
                timespec="seconds"
            )
    }

    with open(
        NORMALIZATION_FILE,
        "w"
    ) as f:

        json.dump(
            normalization,
            f,
            indent=2
        )

    log("")
    log(
        f"Global P95 = "
        f"{global_p95:.12g}"
    )

    log(
        f"Global maximum = "
        f"{global_max:.12g}"
    )

    log(
        f"Mean P = "
        f"{mean_p:.12g}"
    )

    log(
        f"Normalization scan took "
        f"{elapsed:.2f} seconds"
    )

    log("")
    log(
        "Saved normalization:"
    )

    log(
        str(
            NORMALIZATION_FILE
        )
    )

    return global_p95


# =========================================================
# Load or calculate global normalization
# =========================================================

def get_global_normalization(
    cache_files
):

    if NORMALIZATION_FILE.exists():

        try:

            with open(
                NORMALIZATION_FILE
            ) as f:

                data = json.load(
                    f
                )

            saved_models = int(
                data.get(
                    "models",
                    -1
                )
            )

            saved_vectors = int(
                data.get(
                    "vectors_per_model",
                    -1
                )
            )

            saved_p95 = float(
                data.get(
                    "global_p95",
                    0.0
                )
            )

            if (
                saved_models
                == len(cache_files)
                and
                saved_vectors
                == VECTORS_PER_MODEL
                and
                np.isfinite(
                    saved_p95
                )
                and
                saved_p95 > 0.0
            ):

                log("")
                log(
                    "Using existing global "
                    "normalization."
                )

                log(
                    f"Global P95 = "
                    f"{saved_p95:.12g}"
                )

                return saved_p95

        except Exception:

            pass

    return calculate_global_normalization(
        cache_files
    )


# =========================================================
# Compact model encoding
#
# Each model has 196 sticks.
#
# We do NOT write:
#
#     x
#     y
#     q
#     u
#     p
#     psi_rad
#     length_px
#     opacity
#
# 196 times as verbose JSON objects.
#
# Instead:
#
# byte 0 .. 195
#     normalized polarization amplitude
#
# byte 196 .. 391
#     polarization angle
#
# Amplitude:
#
#     0   -> P = 0
#     255 -> P >= global P95
#
# Angle:
#
#     0 .. 255 spans 0 .. pi
#
# Since polarization is headless,
# psi and psi + pi are equivalent.
# =========================================================

def encode_model(
    q,
    u,
    global_p95
):

    q = np.asarray(
        q,
        dtype=np.float64
    )

    u = np.asarray(
        u,
        dtype=np.float64
    )

    if (
        q.size
        != VECTORS_PER_MODEL
        or
        u.size
        != VECTORS_PER_MODEL
    ):

        raise RuntimeError(
            "Unexpected Q/U array length."
        )

    if (
        not np.all(
            np.isfinite(
                q
            )
        )
        or
        not np.all(
            np.isfinite(
                u
            )
        )
    ):

        raise RuntimeError(
            "Non-finite Q/U value found."
        )


    # -----------------------------------------------------
    # Polarization magnitude
    # -----------------------------------------------------

    p = np.hypot(
        q,
        u
    )

    normalized_p = np.clip(
        p
        / global_p95,
        0.0,
        1.0
    )

    amplitude_byte = np.rint(
        normalized_p
        * 255.0
    ).astype(
        np.uint8
    )


    # -----------------------------------------------------
    # Headless polarization angle
    #
    # psi = 1/2 atan2(U,Q)
    #
    # Map to [0, pi).
    # -----------------------------------------------------

    psi = (
        0.5
        * np.arctan2(
            u,
            q
        )
    )

    psi = np.mod(
        psi,
        np.pi
    )

    angle_byte = np.floor(
        psi
        / np.pi
        * 256.0
    )

    angle_byte = np.clip(
        angle_byte,
        0.0,
        255.0
    ).astype(
        np.uint8
    )


    # -----------------------------------------------------
    # Pack:
    #
    # [196 amplitude bytes][196 angle bytes]
    # -----------------------------------------------------

    payload = (
        amplitude_byte.tobytes()
        +
        angle_byte.tobytes()
    )

    encoded = base64.b64encode(
        payload
    ).decode(
        "ascii"
    )

    return encoded


# =========================================================
# Read and encode one cached cosmology
# =========================================================

def encoded_cosmology(
    Omega_b,
    Omega_c,
    Omega_l,
    global_p95
):

    path = model_path(
        Omega_b,
        Omega_c,
        Omega_l
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing cached model: {path}"
        )

    with np.load(
        path
    ) as data:

        q = data["q"]
        u = data["u"]

    return encode_model(
        q,
        u,
        global_p95
    )


# =========================================================
# Atomic JSON writer
#
# Write to .tmp first so interruption cannot leave behind
# a half-written slice that looks complete.
# =========================================================

def write_json_atomic(
    path,
    data
):

    temp_path = (
        path.with_suffix(
            path.suffix
            + ".tmp"
        )
    )

    with open(
        temp_path,
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

    os.replace(
        temp_path,
        path
    )


# =========================================================
# Standard metadata shared by every slice
# =========================================================

def slice_header(
    varying_parameter,
    fixed_parameters,
    global_p95
):

    return {

        "format":
            "cmb-polarization-slice",

        "format_version":
            FORMAT_VERSION,

        "varying_parameter":
            varying_parameter,

        "fixed_parameters":
            fixed_parameters,

        "step":
            STEP,

        "grid": {

            "rows":
                STICK_GRID,

            "columns":
                STICK_GRID,

            "count":
                VECTORS_PER_MODEL
        },

        "normalization": {

            "method":
                "global_95th_percentile",

            "p95":
                global_p95
        },

        "encoding": {

            "type":
                "base64_u8_amplitude_angle",

            "bytes_per_model":
                2
                * VECTORS_PER_MODEL,

            "layout":
                (
                    "first 196 bytes = amplitude; "
                    "next 196 bytes = angle"
                ),

            "amplitude_decode":
                "normalized_p = byte / 255",

            "angle_decode":
                "psi_rad = byte * pi / 256"
        },

        "models": {}
    }


# =========================================================
# Omega_b slice
#
# File:
#
#     Ob_Oc0.275_Ol0.675.json
#
# Models:
#
#     0.025 ... 1.000
# =========================================================

def build_omega_b_slice(
    Omega_c,
    Omega_l,
    global_p95
):

    filename = (
        "Ob_Oc"
        + f"{Omega_c:.3f}"
        + "_Ol"
        + f"{Omega_l:.3f}"
        + ".json"
    )

    path = (
        OUTPUT_DIR
        / filename
    )

    if (
        path.exists()
        and
        not OVERWRITE_EXISTING_SLICES
    ):

        return "skipped"

    data = slice_header(
        "omega_b",

        {
            "omega_c":
                Omega_c,

            "omega_lambda":
                Omega_l
        },

        global_p95
    )

    for Omega_b in omega_b_values:

        data["models"][
            json_key(
                Omega_b
            )
        ] = {

            "data":
                encoded_cosmology(
                    Omega_b,
                    Omega_c,
                    Omega_l,
                    global_p95
                )
        }

    write_json_atomic(
        path,
        data
    )

    return "written"


# =========================================================
# Omega_c slice
#
# File:
#
#     Ob0.050_Oc_Ol0.675.json
#
# Models:
#
#     0.000 ... 1.000
# =========================================================

def build_omega_c_slice(
    Omega_b,
    Omega_l,
    global_p95
):

    filename = (
        "Ob"
        + f"{Omega_b:.3f}"
        + "_Oc_Ol"
        + f"{Omega_l:.3f}"
        + ".json"
    )

    path = (
        OUTPUT_DIR
        / filename
    )

    if (
        path.exists()
        and
        not OVERWRITE_EXISTING_SLICES
    ):

        return "skipped"

    data = slice_header(
        "omega_c",

        {
            "omega_b":
                Omega_b,

            "omega_lambda":
                Omega_l
        },

        global_p95
    )

    for Omega_c in omega_c_values:

        data["models"][
            json_key(
                Omega_c
            )
        ] = {

            "data":
                encoded_cosmology(
                    Omega_b,
                    Omega_c,
                    Omega_l,
                    global_p95
                )
        }

    write_json_atomic(
        path,
        data
    )

    return "written"


# =========================================================
# Omega_Lambda slice
#
# File:
#
#     Ob0.050_Oc0.275_Ol.json
#
# Models:
#
#     0.000 ... 1.000
# =========================================================

def build_omega_l_slice(
    Omega_b,
    Omega_c,
    global_p95
):

    filename = (
        "Ob"
        + f"{Omega_b:.3f}"
        + "_Oc"
        + f"{Omega_c:.3f}"
        + "_Ol.json"
    )

    path = (
        OUTPUT_DIR
        / filename
    )

    if (
        path.exists()
        and
        not OVERWRITE_EXISTING_SLICES
    ):

        return "skipped"

    data = slice_header(
        "omega_l",

        {
            "omega_b":
                Omega_b,

            "omega_c":
                Omega_c
        },

        global_p95
    )

    for Omega_l in omega_l_values:

        data["models"][
            json_key(
                Omega_l
            )
        ] = {

            "data":
                encoded_cosmology(
                    Omega_b,
                    Omega_c,
                    Omega_l,
                    global_p95
                )
        }

    write_json_atomic(
        path,
        data
    )

    return "written"


# =========================================================
# Progress helper
# =========================================================

def report_progress(
    done,
    total,
    written,
    skipped,
    start_time
):

    elapsed = (
        time.perf_counter()
        - start_time
    )

    if done > 0:

        rate = (
            elapsed
            / done
        )

        remaining_seconds = (
            total
            - done
        ) * rate

    else:

        remaining_seconds = 0.0

    log(
        f"[{done:,}/{total:,}] "
        f"written={written:,} "
        f"skipped={skipped:,} "
        f"ETA={remaining_seconds / 60.0:.1f} min"
    )


# =========================================================
# TEST BUILD
#
# These are the three reference slices already used during
# development:
#
# Omega_b varying:
#     Oc = 0.275
#     Ol = 0.675
#
# Omega_c varying:
#     Ob = 0.050
#     Ol = 0.675
#
# Omega_Lambda varying:
#     Ob = 0.050
#     Oc = 0.275
# =========================================================

def build_test_slices(
    global_p95
):

    log("")
    log("=" * 72)
    log("BUILDING THREE REFERENCE SLICES")
    log("=" * 72)

    result = build_omega_b_slice(
        0.275,
        0.675,
        global_p95
    )

    log(
        "Omega_b reference slice: "
        + result
    )

    result = build_omega_c_slice(
        0.050,
        0.675,
        global_p95
    )

    log(
        "Omega_c reference slice: "
        + result
    )

    result = build_omega_l_slice(
        0.050,
        0.275,
        global_p95
    )

    log(
        "Omega_Lambda reference slice: "
        + result
    )


# =========================================================
# FULL PRODUCTION BUILD
# =========================================================

def build_all_slices(
    global_p95
):

    written = 0
    skipped = 0
    done = 0

    start = (
        time.perf_counter()
    )

    log("")
    log("=" * 72)
    log("BUILDING FULL SLICE DATABASE")
    log("=" * 72)

    log(
        f"Omega_b slices: "
        f"{N_BARYON_SLICES:,}"
    )

    log(
        f"Omega_c slices: "
        f"{N_CDM_SLICES:,}"
    )

    log(
        f"Omega_Lambda slices: "
        f"{N_LAMBDA_SLICES:,}"
    )

    log(
        f"Total slices: "
        f"{TOTAL_SLICES:,}"
    )

    log("")


    # -----------------------------------------------------
    # Vary Omega_b
    # -----------------------------------------------------

    log(
        "Building Omega_b slices..."
    )

    for Omega_c in omega_c_values:

        for Omega_l in omega_l_values:

            result = build_omega_b_slice(
                Omega_c,
                Omega_l,
                global_p95
            )

            done += 1

            if result == "written":

                written += 1

            else:

                skipped += 1

            if (
                done % 100 == 0
            ):

                report_progress(
                    done,
                    TOTAL_SLICES,
                    written,
                    skipped,
                    start
                )


    # -----------------------------------------------------
    # Vary Omega_c
    # -----------------------------------------------------

    log(
        "Building Omega_c slices..."
    )

    for Omega_b in omega_b_values:

        for Omega_l in omega_l_values:

            result = build_omega_c_slice(
                Omega_b,
                Omega_l,
                global_p95
            )

            done += 1

            if result == "written":

                written += 1

            else:

                skipped += 1

            if (
                done % 100 == 0
            ):

                report_progress(
                    done,
                    TOTAL_SLICES,
                    written,
                    skipped,
                    start
                )


    # -----------------------------------------------------
    # Vary Omega_Lambda
    # -----------------------------------------------------

    log(
        "Building Omega_Lambda slices..."
    )

    for Omega_b in omega_b_values:

        for Omega_c in omega_c_values:

            result = build_omega_l_slice(
                Omega_b,
                Omega_c,
                global_p95
            )

            done += 1

            if result == "written":

                written += 1

            else:

                skipped += 1

            if (
                done % 100 == 0
                or
                done == TOTAL_SLICES
            ):

                report_progress(
                    done,
                    TOTAL_SLICES,
                    written,
                    skipped,
                    start
                )


    elapsed = (
        time.perf_counter()
        - start
    )

    log("")
    log("=" * 72)
    log("SLICE BUILD COMPLETE")
    log("=" * 72)

    log(
        f"Written: "
        f"{written:,}"
    )

    log(
        f"Skipped: "
        f"{skipped:,}"
    )

    log(
        f"Total: "
        f"{done:,}"
    )

    log(
        f"Elapsed: "
        f"{elapsed / 60.0:.2f} minutes"
    )


# =========================================================
# Manifest
# =========================================================

def write_manifest(
    global_p95,
    test_mode
):

    json_files = [
        p
        for p
        in OUTPUT_DIR.glob(
            "*.json"
        )
        if p.name
        != "manifest.json"
    ]

    total_bytes = sum(
        p.stat().st_size
        for p in json_files
    )

    manifest = {

        "format":
            "CMB polarization database",

        "format_version":
            FORMAT_VERSION,

        "test_mode":
            test_mode,

        "step":
            STEP,

        "omega_b": {

            "minimum":
                min(
                    omega_b_values
                ),

            "maximum":
                max(
                    omega_b_values
                ),

            "count":
                len(
                    omega_b_values
                )
        },

        "omega_c": {

            "minimum":
                min(
                    omega_c_values
                ),

            "maximum":
                max(
                    omega_c_values
                ),

            "count":
                len(
                    omega_c_values
                )
        },

        "omega_lambda": {

            "minimum":
                min(
                    omega_l_values
                ),

            "maximum":
                max(
                    omega_l_values
                ),

            "count":
                len(
                    omega_l_values
                )
        },

        "cosmologies":
            TOTAL_COSMOLOGIES,

        "expected_full_slice_count":
            TOTAL_SLICES,

        "current_slice_count":
            len(
                json_files
            ),

        "grid":
            STICK_GRID,

        "vectors_per_model":
            VECTORS_PER_MODEL,

        "global_p95":
            global_p95,

        "encoding":
            "base64_u8_amplitude_angle",

        "current_database_bytes":
            total_bytes,

        "current_database_megabytes":
            (
                total_bytes
                / 1024.0
                / 1024.0
            ),

        "created":
            datetime.now().isoformat(
                timespec="seconds"
            )
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

    return (
        len(
            json_files
        ),
        total_bytes
    )


# =========================================================
# MAIN
# =========================================================

def main():

    overall_start = (
        time.perf_counter()
    )

    log("")
    log("=" * 72)
    log("CMB POLARIZATION SLICE PACKAGER")
    log("=" * 72)

    log(
        "Started: "
        + datetime.now().isoformat(
            timespec="seconds"
        )
    )

    log(
        f"Test mode: "
        f"{TEST_MODE}"
    )

    log(
        f"Expected cosmologies: "
        f"{TOTAL_COSMOLOGIES:,}"
    )

    log(
        f"Expected full slice count: "
        f"{TOTAL_SLICES:,}"
    )

    log("")

    cache_files = verify_cache()

    global_p95 = (
        get_global_normalization(
            cache_files
        )
    )

    if TEST_MODE:

        build_test_slices(
            global_p95
        )

    else:

        build_all_slices(
            global_p95
        )


    file_count, total_bytes = (
        write_manifest(
            global_p95,
            TEST_MODE
        )
    )

    overall_elapsed = (
        time.perf_counter()
        - overall_start
    )

    log("")
    log("=" * 72)
    log("PACKAGING RUN COMPLETE")
    log("=" * 72)

    log(
        f"Slice files currently present: "
        f"{file_count:,}"
    )

    log(
        f"Current slice database size: "
        f"{total_bytes / 1024.0 / 1024.0:.2f} MB"
    )

    log(
        f"Global P95: "
        f"{global_p95:.12g}"
    )

    log(
        f"Elapsed wall time: "
        f"{overall_elapsed / 60.0:.2f} minutes"
    )

    log("")
    log(
        "Output directory:"
    )

    log(
        str(
            OUTPUT_DIR
        )
    )

    log("")
    log(
        "Manifest:"
    )

    log(
        str(
            MANIFEST_FILE
        )
    )

    log("")
    log(
        "Finished: "
        + datetime.now().isoformat(
            timespec="seconds"
        )
    )


if __name__ == "__main__":

    main()