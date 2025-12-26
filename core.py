"""
core.py

Core utilities for loading the inhibitory control Neuropixels dataset and
running CEBRA analyses.

HOW TO POINT TO YOUR DATA
-------------------------
Option 1:
    Edit DEFAULT_DATA_ROOT below to the directory containing:

        firing rates/
            dbh dcz/
                free/
                tone/
                post/
            dbh saline/
                free/
                tone/
                post/

Option 2 (more flexible):
    Set an environment variable before running scripts:

        export INHIB_CONTROL_ROOT="/path/to/preprocessed time series"
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Sequence, Tuple

import h5py
import numpy as np
from cebra import CEBRA

# Optional plotting libs (only used in plotting helpers)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import plotly.graph_objects as go
from matplotlib import cm


# ======================================================================
# CONFIG
# ======================================================================

DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "INHIB_CONTROL_ROOT",
        "/Users/timothylantin/Desktop/scripts/cebra_test/data/inhibitory_control/preprocessed time series"  # <-- EDIT THIS
    )
)


def _ensure_root(root: os.PathLike | str | None) -> Path:
    """Return a Path for root, falling back to DEFAULT_DATA_ROOT."""
    if root is None:
        return DEFAULT_DATA_ROOT
    return Path(root)


# ======================================================================
# BASIC LOADING UTILITIES
# ======================================================================

def load_mat_data(mat_path: os.PathLike | str) -> np.ndarray:
    """
    Extract continuous numeric data from '#refs#' group in a .mat/HDF5 file.

    Returns
    -------
    X : np.ndarray, shape (T, F)
        Time x features, with 2D datasets concatenated along feature axis.
    """
    mat_path = Path(mat_path)
    arrays: List[np.ndarray] = []

    with h5py.File(mat_path, "r") as f:
        if "#refs#" not in f:
            raise ValueError(f"No '#refs#' group in {mat_path}")

        group = f["#refs#"]
        for key in group.keys():
            obj = group[key]
            if isinstance(obj, h5py.Dataset) and obj.dtype != object:
                arr = np.array(obj)
                if arr.ndim == 2:  # T x F
                    arrays.append(arr)

    if len(arrays) == 0:
        raise ValueError(f"No usable 2D datasets in {mat_path}")

    T = max(a.shape[0] for a in arrays)
    padded = []
    for a in arrays:
        if a.shape[0] < T:
            pad = np.zeros((T - a.shape[0], a.shape[1]))
            a = np.vstack([a, pad])
        padded.append(a)

    return np.concatenate(padded, axis=1)  # T x (sum F)


def session_length(mat_path: os.PathLike | str) -> int:
    """Return the max time length for 2D datasets in '#refs#' of a session."""
    mat_path = Path(mat_path)
    with h5py.File(mat_path, "r") as f:
        if "#refs#" not in f:
            return 0

        group = f["#refs#"]
        arrays = []
        for key in group.keys():
            obj = group[key]
            if isinstance(obj, h5py.Dataset) and obj.dtype != object:
                arr = np.array(obj)
                if arr.ndim == 2:
                    arrays.append(arr)

    return max(a.shape[0] for a in arrays) if arrays else 0


# ======================================================================
# SUCCESS / PUNISH TRIAL EXTRACTION
# ======================================================================

def extract_success_punish(mat_path):
    """
    Extract trial matrices from MATLAB v7.3 .mat tone files
    where:
        Session_TonePeriod_Train/Success  -> cell array of object refs
        Session_TonePeriod_Train/Punish   -> cell array of object refs

    Returns:
        success_trials : list of (T, N)
        punish_trials  : list of (T, N)
    """
    import h5py
    import numpy as np

    success_trials = []
    punish_trials = []

    with h5py.File(mat_path, "r") as f:

        # ---- locate the session group ----
        session_keys = [k for k in f.keys() if k.startswith("Session_")]
        if not session_keys:
            raise RuntimeError(f"No Session_* key in {mat_path}")

        session = f[session_keys[0]]

        if "Success" not in session or "Punish" not in session:
            raise RuntimeError(
                f"Expected 'Success' and 'Punish' in {session.name} of {mat_path}, "
                f"found: {list(session.keys())}"
            )

        # MATLAB cell arrays → (1, n_trials)
        success_cells = session["Success"][0]
        punish_cells  = session["Punish"][0]

        # ---- dereference SUCCESS trials ----
        for ref in success_cells:
            if isinstance(ref, h5py.Reference):
                arr = np.array(f[ref])
                success_trials.append(arr)

        # ---- dereference PUNISH trials ----
        for ref in punish_cells:
            if isinstance(ref, h5py.Reference):
                arr = np.array(f[ref])
                punish_trials.append(arr)

    return success_trials, punish_trials



def _pad_list_of_mats(
    mats: Sequence[np.ndarray],
    Fmax: int | None = None,
) -> Tuple[List[np.ndarray], int]:
    """
    Pad a list of 2D matrices on the feature dimension so they share Fmax.
    Returns padded list and Fmax.
    """
    mats = list(mats)
    if len(mats) == 0:
        return [], 0 if Fmax is None else Fmax

    if Fmax is None:
        Fmax = max(m.shape[1] for m in mats)

    padded: List[np.ndarray] = []
    for m in mats:
        if m.shape[1] < Fmax:
            pad = np.zeros((m.shape[0], Fmax - m.shape[1]))
            m = np.hstack([m, pad])
        padded.append(m)

    return padded, Fmax


# ======================================================================
# TONE SUCCESS/PUNISH LOADERS
# ======================================================================

#
# --- BACKWARD-COMPAT COMPATIBILITY SHIM ---
#
# Some notebooks call `extract_success_punish_trials`
# Others call `extract_success_punish`
#
# Both should resolve to the same underlying function.
#

def extract_success_punish_trials(*args, **kwargs):
    """Alias kept for older notebooks — forwards to extract_success_punish."""
    return extract_success_punish(*args, **kwargs)


def load_tone_success_punish_all(
    root: os.PathLike | str | None = None,
) -> Tuple[np.ndarray | None, np.ndarray | None, list[Path], list[Path]]:
    """
    Load ALL TONE-period Success and Punish trials across dbh dcz + dbh saline.

    Returns
    -------
    neural_tone_all_success : (T_success_total, F) or None
    neural_tone_all_punish  : (T_punish_total,  F) or None
    tone_mat_paths_dcz      : list of Paths
    tone_mat_paths_saline   : list of Paths
    """
    root = _ensure_root(root)

    tone_dcz_dir = root / "firing rates" / "dbh dcz" / "tone"
    tone_saline_dir = root / "firing rates" / "dbh saline" / "tone"

    tone_mat_paths_dcz = sorted(p for p in tone_dcz_dir.iterdir() if p.suffix == ".mat")
    tone_mat_paths_saline = sorted(p for p in tone_saline_dir.iterdir() if p.suffix == ".mat")

    all_success: List[np.ndarray] = []
    all_punish: List[np.ndarray] = []

    for mat_path in (tone_mat_paths_dcz + tone_mat_paths_saline):
        succ_trials, pun_trials = extract_success_punish_trials(mat_path)

        if succ_trials:
            succ_padded, _ = _pad_list_of_mats(succ_trials)
            all_success.extend(succ_padded)

        if pun_trials:
            pun_padded, _ = _pad_list_of_mats(pun_trials)
            all_punish.extend(pun_padded)

    if not all_success and not all_punish:
        raise ValueError("No trials found across all tone sessions.")

    all_mats = all_success + all_punish
    Fmax = max(m.shape[1] for m in all_mats)

    def _pad_to_Fmax(mats: Sequence[np.ndarray]) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for m in mats:
            if m.shape[1] < Fmax:
                pad = np.zeros((m.shape[0], Fmax - m.shape[1]))
                m = np.hstack([m, pad])
            out.append(m)
        return out

    all_success_final = _pad_to_Fmax(all_success)
    all_punish_final = _pad_to_Fmax(all_punish)

    neural_tone_all_success = np.vstack(all_success_final) if all_success_final else None
    neural_tone_all_punish = np.vstack(all_punish_final) if all_punish_final else None

    return (
        neural_tone_all_success,
        neural_tone_all_punish,
        tone_mat_paths_dcz,
        tone_mat_paths_saline,
    )


def load_tone_success_punish_train_test(
    root: os.PathLike | str | None = None,
    test_size: float = 0.2,
    random_seed: int = 0,
) -> Tuple[np.ndarray | None, np.ndarray | None,
           np.ndarray | None, np.ndarray | None,
           list[Path], list[Path], list[Path], list[Path]]:
    """
    SESSION-LEVEL train/test split for TONE-period Success and Punish trials.

    Returns
    -------
    train_success : (T_train_success, F) or None
    train_punish  : (T_train_punish,  F) or None
    test_success  : (T_test_success,  F) or None
    test_punish   : (T_test_punish,   F) or None

    train_paths_dcz, train_paths_saline, test_paths_dcz, test_paths_saline
    """
    root = _ensure_root(root)
    rng = np.random.default_rng(random_seed)

    tone_dcz_dir = root / "firing rates" / "dbh dcz" / "tone"
    tone_saline_dir = root / "firing rates" / "dbh saline" / "tone"

    tone_mat_paths_dcz = sorted(p for p in tone_dcz_dir.iterdir() if p.suffix == ".mat")
    tone_mat_paths_saline = sorted(p for p in tone_saline_dir.iterdir() if p.suffix == ".mat")

    def split_paths(paths: Sequence[Path]) -> Tuple[list[Path], list[Path]]:
        paths = np.array(paths, dtype=object)
        rng.shuffle(paths)
        n_test = max(1, int(len(paths) * test_size))
        return paths[n_test:].tolist(), paths[:n_test].tolist()

    train_paths_dcz, test_paths_dcz = split_paths(tone_mat_paths_dcz)
    train_paths_saline, test_paths_saline = split_paths(tone_mat_paths_saline)

    train_paths_all = train_paths_dcz + train_paths_saline
    test_paths_all = test_paths_dcz + test_paths_saline

    train_success_trials: List[np.ndarray] = []
    train_punish_trials: List[np.ndarray] = []
    test_success_trials: List[np.ndarray] = []
    test_punish_trials: List[np.ndarray] = []

    for mat_path in train_paths_all:
        succ, pun = extract_success_punish_trials(mat_path)
        train_success_trials.extend(succ)
        train_punish_trials.extend(pun)

    for mat_path in test_paths_all:
        succ, pun = extract_success_punish_trials(mat_path)
        test_success_trials.extend(succ)
        test_punish_trials.extend(pun)

    all_mats = train_success_trials + train_punish_trials + test_success_trials + test_punish_trials
    if len(all_mats) == 0:
        raise ValueError("No trials found across all sessions.")

    Fmax = max(m.shape[1] for m in all_mats)

    def pad_to_Fmax(mats: Sequence[np.ndarray]) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for m in mats:
            if m.shape[1] < Fmax:
                pad = np.zeros((m.shape[0], Fmax - m.shape[1]))
                m = np.hstack([m, pad])
            out.append(m)
        return out

    train_success_trials = pad_to_Fmax(train_success_trials)
    train_punish_trials = pad_to_Fmax(train_punish_trials)
    test_success_trials = pad_to_Fmax(test_success_trials)
    test_punish_trials = pad_to_Fmax(test_punish_trials)

    train_success = np.vstack(train_success_trials) if train_success_trials else None
    train_punish = np.vstack(train_punish_trials) if train_punish_trials else None
    test_success = np.vstack(test_success_trials) if test_success_trials else None
    test_punish = np.vstack(test_punish_trials) if test_punish_trials else None

    return (
        train_success,
        train_punish,
        test_success,
        test_punish,
        train_paths_dcz,
        train_paths_saline,
        test_paths_dcz,
        test_paths_saline,
    )


def load_tone_success_punish_trials_train_test(
    root: os.PathLike | str | None = None,
    test_size: float = 0.2,
    random_seed: int = 0,
) -> Tuple[list[np.ndarray], np.ndarray,
           list[np.ndarray], np.ndarray,
           list[Path], list[Path]]:
    """
    Trial-level train/test split for TONE Success/Punish, preserving sessions.

    Returns
    -------
    train_trials : list of np.ndarray (T_trial, F)
    train_labels : np.ndarray (n_train_trials,)  0 = Success, 1 = Punish
    test_trials  : list of np.ndarray (T_trial, F)
    test_labels  : np.ndarray (n_test_trials,)   0 = Success, 1 = Punish
    train_paths  : list of Paths
    test_paths   : list of Paths
    """
    root = _ensure_root(root)
    rng = np.random.default_rng(random_seed)

    tone_dcz_dir = root / "firing rates" / "dbh dcz" / "tone"
    tone_saline_dir = root / "firing rates" / "dbh saline" / "tone"

    tone_paths = sorted(p for p in tone_dcz_dir.iterdir() if p.suffix == ".mat") + \
                 sorted(p for p in tone_saline_dir.iterdir() if p.suffix == ".mat")

    tone_paths = np.array(tone_paths, dtype=object)
    rng.shuffle(tone_paths)
    n_test = max(1, int(len(tone_paths) * test_size))

    test_paths = tone_paths[:n_test].tolist()
    train_paths = tone_paths[n_test:].tolist()

    train_trials: List[np.ndarray] = []
    train_labels: List[int] = []
    test_trials: List[np.ndarray] = []
    test_labels: List[int] = []

    for p in train_paths:
        succ, pun = extract_success_punish_trials(p)
        for t in succ:
            train_trials.append(t)
            train_labels.append(0)
        for t in pun:
            train_trials.append(t)
            train_labels.append(1)

    for p in test_paths:
        succ, pun = extract_success_punish_trials(p)
        for t in succ:
            test_trials.append(t)
            test_labels.append(0)
        for t in pun:
            test_trials.append(t)
            test_labels.append(1)

    all_trials = train_trials + test_trials
    if not all_trials:
        raise ValueError("No trials found across train+test tone sessions.")

    Fmax = max(t.shape[1] for t in all_trials)

    def pad_trial(t: np.ndarray) -> np.ndarray:
        if t.shape[1] < Fmax:
            pad = np.zeros((t.shape[0], Fmax - t.shape[1]))
            return np.hstack([t, pad])
        return t

    train_trials = [pad_trial(t) for t in train_trials]
    test_trials = [pad_trial(t) for t in test_trials]

    return (
        train_trials,
        np.array(train_labels, dtype=int),
        test_trials,
        np.array(test_labels, dtype=int),
        train_paths,
        test_paths,
    )

# -----------------------------------------------------------------------------
# Helper: inspect structure of a tone .mat file (for debugging)
# -----------------------------------------------------------------------------
def inspect_tone_mat_structure(mat_path: str, max_lines: int = 200):
    """
    Quick utility to print the HDF5 structure of a tone .mat file.
    Use this in a notebook to see which keys to plug into
    `extract_success_punish`.

    Example (in a notebook):
        from core import inspect_tone_mat_structure
        inspect_tone_mat_structure("/path/to/some_tone_file.mat")
    """
    import h5py

    with h5py.File(mat_path, "r") as f:
        print(f"\n=== Top-level keys in {mat_path} ===")
        for k in f.keys():
            print("  ", k)

        print("\n=== Full tree (datasets show shape) ===")

        lines = []

        def _visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                lines.append(f"[DATA] {name} : shape={obj.shape}, dtype={obj.dtype}")
            else:
                lines.append(f"[GRP ] {name}")

        f.visititems(_visitor)

        for i, line in enumerate(lines):
            print(line)
            if i + 1 >= max_lines:
                print(f"... (truncated after {max_lines} lines) ...")
                break


# ======================================================================
# NEUROPIXELS SESSION SPLITTING
# ======================================================================

def train_test_split_neuropixel_only(
    root: os.PathLike | str | None = None,
    test_size: float = 0.2,
    random_seed: int = 42,
    required_dir: str = "firing rates",
    allowed_ext: tuple[str, ...] = (".mat",),
) -> Tuple[list[Path], list[Path]]:
    """
    Recursively collect only Neuropixels .mat files under 'firing rates'
    and split them into train/test sets.
    """
    root = _ensure_root(root)
    rng = np.random.default_rng(random_seed)
    all_paths: List[Path] = []

    for dirpath, _, filenames in os.walk(root):
        if required_dir.lower() not in dirpath.lower():
            continue
        for fname in filenames:
            if fname.endswith(allowed_ext):
                all_paths.append(Path(dirpath) / fname)

    if not all_paths:
        raise RuntimeError(
            f"NO Neuropixels files found under '{required_dir}' in: {root}"
        )

    all_paths = np.array(sorted(all_paths), dtype=object)
    rng.shuffle(all_paths)

    n_test = max(1, int(len(all_paths) * test_size))
    test_paths = all_paths[:n_test].tolist()
    train_paths = all_paths[n_test:].tolist()
    return train_paths, test_paths


def split_by_period_3way(
    paths: Sequence[os.PathLike | str],
) -> Tuple[list[Path], list[Path], list[Path]]:
    """
    Split a list of paths into free / tone / post based on folder names.
    """
    free_paths: List[Path] = []
    tone_paths: List[Path] = []
    post_paths: List[Path] = []

    for p in paths:
        p = Path(p)
        lower = str(p).lower()

        if "/free/" in lower or "freeperiod" in lower:
            free_paths.append(p)
        elif "/tone/" in lower or "toneperiod" in lower:
            tone_paths.append(p)
        elif "/post/" in lower or "postperiod" in lower:
            post_paths.append(p)
        else:
            raise RuntimeError(f"Could not infer period from path:\n{p}")

    return free_paths, tone_paths, post_paths


def load_sessions_raw(
    paths: Sequence[os.PathLike | str],
    loader_fn=load_mat_data,
) -> list[np.ndarray]:
    """
    Load each session as a 2D array [samples, neurons] (flatten trials if needed).
    """
    sessions: List[np.ndarray] = []
    for p in paths:
        X = loader_fn(p)
        if X.ndim == 3:
            n_trials, T, F = X.shape
            X = X.reshape(n_trials * T, F)
        elif X.ndim != 2:
            raise RuntimeError(f"Invalid shape from {p}: {X.shape}")
        sessions.append(X)
    return sessions


def compute_neuron_intersection(session_list: Sequence[np.ndarray]) -> int:
    """Minimum neuron count shared across sessions."""
    return min(X.shape[1] for X in session_list)


def apply_neuron_cut(
    session_list: Sequence[np.ndarray],
    n_neurons_target: int,
) -> list[np.ndarray]:
    """Trim all sessions to the same neuron dimension."""
    return [X[:, :n_neurons_target] for X in session_list]


def zscore_per_session(
    session_list: Sequence[np.ndarray],
    eps: float = 1e-8,
) -> list[np.ndarray]:
    """Z-score each session across time for each neuron."""
    out: List[np.ndarray] = []
    for X in session_list:
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + eps
        out.append((X - mu) / sd)
    return out


# ======================================================================
# CEBRA WRAPPER
# ======================================================================

def run_cebra_modern(
    neural_success: np.ndarray,
    neural_punish: np.ndarray,
    *,
    output_dimension: int = 3,
    model_architecture: str = "offset10-model",
    num_hidden_units: int = 32,
    temperature: float = 2.0,
    time_offsets: int = 10,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    max_iterations: int = 4000,
    distance: str = "cosine",
    device: str = "cuda_if_available",
    verbose: bool = True,
):
    """
    Train a CEBRA model on success + punish data and return embeddings.

    Accepts either:
    - 3D arrays: (n_trials, T, n_units)  -> returns (n_trials, T, output_dimension)
    - 2D arrays: (N, n_units)            -> returns (N, output_dimension)

    Parameters
    ----------
    neural_success, neural_punish : np.ndarray
        Neural data for success and punish conditions.
    output_dimension : int
        Dimensionality of CEBRA embedding space.

    Returns
    -------
    emb_success : np.ndarray
    emb_punish  : np.ndarray
    model       : trained CEBRA estimator
    """

    neural_success = np.asarray(neural_success)
    neural_punish = np.asarray(neural_punish)

    if neural_success.ndim != neural_punish.ndim:
        raise ValueError(
            f"neural_success.ndim={neural_success.ndim} "
            f"!= neural_punish.ndim={neural_punish.ndim}"
        )

    if neural_success.ndim not in (2, 3):
        raise ValueError(
            f"Expected 2D or 3D inputs, got {neural_success.ndim}D."
        )

    # ---------------------------------------------------------
    # CASE 1: 3D — (n_trials, T, n_units)
    # ---------------------------------------------------------
    if neural_success.ndim == 3:
        n_succ, T_s, n_units_s = neural_success.shape
        n_pun,  T_p, n_units_p = neural_punish.shape

        if n_units_s != n_units_p:
            raise ValueError(
                f"Neural feature dimension mismatch: "
                f"success has {n_units_s}, punish has {n_units_p}"
            )

        # Flatten over (trials, time) -> samples
        X_success = neural_success.reshape(-1, n_units_s)
        X_punish  = neural_punish.reshape(-1, n_units_p)

        X_all = np.concatenate([X_success, X_punish], axis=0)

        n_success_samples = X_success.shape[0]
        n_punish_samples  = X_punish.shape[0]

        session_ids = np.concatenate([
            np.zeros(n_success_samples, dtype=int),
            np.ones(n_punish_samples, dtype=int),
        ])

        model = CEBRA(
            model_architecture=model_architecture,
            output_dimension=output_dimension,
            num_hidden_units=num_hidden_units,
            temperature=temperature,
            time_offsets=time_offsets,
            conditional="session",
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            distance=distance,
            verbose=verbose,
            device=device,
        )

        model.fit(X_all, session_ids)
        emb_all = model.transform(X_all)

        emb_success = emb_all[:n_success_samples].reshape(
            n_succ, T_s, output_dimension
        )
        emb_punish = emb_all[n_success_samples:].reshape(
            n_pun, T_p, output_dimension
        )

        return emb_success, emb_punish, model

    # ---------------------------------------------------------
    # CASE 2: 2D — (N, n_units)
    # ---------------------------------------------------------
    else:  # ndim == 2
        n_succ, n_units_s = neural_success.shape
        n_pun,  n_units_p = neural_punish.shape

        if n_units_s != n_units_p:
            raise ValueError(
                f"Neural feature dimension mismatch: "
                f"success has {n_units_s}, punish has {n_units_p}"
            )

        X_success = neural_success
        X_punish  = neural_punish
        X_all = np.concatenate([X_success, X_punish], axis=0)

        n_success_samples = n_succ
        n_punish_samples  = n_pun

        session_ids = np.concatenate([
            np.zeros(n_success_samples, dtype=int),
            np.ones(n_punish_samples, dtype=int),
        ])

        model = CEBRA(
            model_architecture=model_architecture,
            output_dimension=output_dimension,
            num_hidden_units=num_hidden_units,
            temperature=temperature,
            time_offsets=time_offsets,
            conditional="session",
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            distance=distance,
            verbose=verbose,
            device=device,
        )

        model.fit(X_all, session_ids)
        emb_all = model.transform(X_all)

        emb_success = emb_all[:n_success_samples]          # (n_succ, out_dim)
        emb_punish  = emb_all[n_success_samples:]          # (n_pun, out_dim)

        return emb_success, emb_punish, model
    
#
# --- BACKWARD-COMPAT SHIM FOR OLD NOTEBOOKS ---
#

def run_cebra_legacy(
    dcz_data=None,
    saline_data=None,
    model_type="time",
    **kwargs,
):
    """
    Legacy adapter for older notebooks that called run_cebra(...)
    with model_type arguments (e.g., 'time', 'static').

    For now, this simply forwards to the modern run_cebra API.
    """
    # Old notebooks pass these as separate args — now we unify them
    return run_cebra(
        neural_success=dcz_data,
        neural_punish=saline_data,
        **kwargs,
    )


# Alias — so both names work
run_cebra_trials = run_cebra_legacy

# Allow old signature `run_cebra(model_type=..., dcz_data=..., saline_data=...)`
def run_cebra(*args, model_type=None, **kwargs):
    # If someone passes model_type, treat it as a legacy call
    if model_type is not None:
        return run_cebra_legacy(*args, model_type=model_type, **kwargs)

    # Otherwise fall through to the real implementation
    return run_cebra_modern(*args, **kwargs)



# -----------------------------------------------------------------------------
# CEBRA time model: reproduce old "messy notebook" pipeline
# -----------------------------------------------------------------------------

from collections import defaultdict
import numpy as np
from scipy.interpolate import interp1d
from cebra import CEBRA


def run_cebra_time_success_punish(
    neural_success,
    neural_punish,
    n_points=100,
    model_architecture="offset10-model",
    output_dimension=3,
    num_hidden_units=32,
    temperature=2.0,
    time_offsets=10,
    batch_size=256,
    learning_rate=3e-4,
    max_iterations=4000,
    distance="cosine",
    device="cuda_if_available",
    verbose=True,
):
    """
    Replicates the original messy-notebook behavior:

    1) Each trial may have:
           - different time length
           - different number of neurons (features)

    2) Trials are grouped by feature dimension (i.e., per recording session)

    3) For each group:
           - build X_all, session_ids, labels_time
           - train CEBRA-time (conditional='session')
           - transform embeddings

    4) Embeddings are concatenated across groups

    5) For each trial:
           - reconstruct latent trajectory
           - resample to n_points
           - split into success / punish
           - compute mean trajectories
    """

    # ------------------------------------------------------------
    # Build unified trial list with labels
    # ------------------------------------------------------------
    train_trials = []
    train_labels = []

    for tr in neural_success:
        train_trials.append(tr)
        train_labels.append(0)

    for tr in neural_punish:
        train_trials.append(tr)
        train_labels.append(1)

    # ------------------------------------------------------------
    # Group trials by number of neurons (feature dimension)
    # ------------------------------------------------------------
    groups = defaultdict(list)

    for tr, lbl in zip(train_trials, train_labels):
        key = tr.shape[1]   # n_units
        groups[key].append((tr, lbl))

    embeddings_all = []
    trial_ids_all = []
    labels_all = []

    trial_counter = 0

    # ------------------------------------------------------------
    # Train CEBRA separately per feature-dimension group
    # ------------------------------------------------------------
    for n_units, trials in groups.items():

        X_all_list = []
        session_list = []
        label_list = []

        for tr, lbl in trials:
            T = tr.shape[0]

            X_all_list.append(tr)
            session_list.append(np.full(T, trial_counter, dtype=int))
            label_list.append(np.full(T, lbl, dtype=int))

            trial_counter += 1

        X_all = np.concatenate(X_all_list, axis=0)
        session_ids = np.concatenate(session_list)
        labels_time = np.concatenate(label_list)

        assert (
            X_all.shape[0]
            == session_ids.shape[0]
            == labels_time.shape[0]
        )

        # ---- train CEBRA for this group ----
        cebra_time = CEBRA(
            model_architecture=model_architecture,
            output_dimension=output_dimension,
            num_hidden_units=num_hidden_units,
            temperature=temperature,
            time_offsets=time_offsets,
            conditional="session",
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            distance=distance,
            verbose=verbose,
            device=device,
        )

        # ----  sanitize NaNs the SAME way your old notebook did  ----
        if np.isnan(X_all).any():
            # Replace NaNs with column means (per neuron)
            col_means = np.nanmean(X_all, axis=0)
            inds = np.where(np.isnan(X_all))
            X_all[inds] = np.take(col_means, inds[1])

        cebra_time.fit(X_all, session_ids)
        emb_time = cebra_time.transform(X_all)

        embeddings_all.append(emb_time)
        trial_ids_all.append(session_ids)
        labels_all.append(labels_time)

    # ------------------------------------------------------------
    # Merge embeddings across groups
    # ------------------------------------------------------------
    emb_time = np.concatenate(embeddings_all, axis=0)
    trial_ids = np.concatenate(trial_ids_all, axis=0)
    labels_time = np.concatenate(labels_all, axis=0)

    # ------------------------------------------------------------
    # Resampling helper
    # ------------------------------------------------------------
    def resample_traj(traj, n_points):
        T = traj.shape[0]
        t_old = np.linspace(0, 1, T)
        t_new = np.linspace(0, 1, n_points)
        return interp1d(t_old, traj, axis=0, kind="linear")(t_new)

    # ------------------------------------------------------------
    # Reconstruct per-trial trajectories
    # ------------------------------------------------------------
    success_trajs = []
    punish_trajs = []

    for tid in np.unique(trial_ids):
        mask = (trial_ids == tid)
        traj = emb_time[mask]
        lab = labels_time[mask]

        outcome = 0 if (lab.mean() < 0.5) else 1
        traj_rs = resample_traj(traj, n_points)

        if outcome == 0:
            success_trajs.append(traj_rs)
        else:
            punish_trajs.append(traj_rs)

    success_trajs = np.stack(success_trajs)
    punish_trajs = np.stack(punish_trajs)

    mean_success = success_trajs.mean(axis=0)
    mean_punish = punish_trajs.mean(axis=0)

    return (
        mean_success,
        mean_punish,
        success_trajs,
        punish_trajs,
        emb_time,
        trial_ids,
        labels_time,
        cebra_time,   # last-trained model
    )


import numpy as np

def pad_trial_list_to_array(trial_list, pad_value=np.nan):
    """
    Convert a list of (T, N) trial matrices into a padded array
    shaped (n_trials, T_max, N_max). Shorter trials are padded
    with `pad_value`.
    """
    if not trial_list:
        raise ValueError("pad_trial_list_to_array received an empty list.")

    T_max = max(tr.shape[0] for tr in trial_list)
    N_max = max(tr.shape[1] for tr in trial_list)

    arr = np.full((len(trial_list), T_max, N_max), pad_value, dtype=float)

    for i, tr in enumerate(trial_list):
        T, N = tr.shape
        arr[i, :T, :N] = tr

    return arr

# -----------------------------------------------------------------------------
# Trial-wise loader for tone success/punish (for CEBRA-time)
# -----------------------------------------------------------------------------

def load_tone_success_punish_trials(root):
    """
    Returns:
        neural_success : (N_succ, T_max, N_max)
        neural_punish  : (N_pun,  T_max, N_max)
        tone_mat_paths : list of file paths used
    """
    import glob, os, numpy as np

    tone_mat_paths = sorted(
        glob.glob(os.path.join(root, "**", "tone", "*.mat"), recursive=True)
    )

    success_list = []
    punish_list  = []

    for mat_path in tone_mat_paths:
        succ, pun = extract_success_punish(mat_path)

        success_list.extend(succ)
        punish_list.extend(pun)

    if not success_list:
        raise RuntimeError("No SUCCESS trials found across tone files.")
    if not punish_list:
        raise RuntimeError("No PUNISH trials found across tone files.")

    neural_success = pad_trial_list_to_array(success_list)
    neural_punish  = pad_trial_list_to_array(punish_list)

    return neural_success, neural_punish, tone_mat_paths



# ======================================================================
# PLOTTING HELPERS
# ======================================================================

def plot_cebra_3d(
    emb_dcz: np.ndarray,
    emb_saline: np.ndarray,
    title: str = "CEBRA Embedding (3D)",
) -> None:
    """Interactive 3D scatter plot for two CEBRA embeddings (e.g. DCZ vs Saline)."""
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=emb_dcz[:, 0],
        y=emb_dcz[:, 1],
        z=emb_dcz[:, 2],
        mode="markers",
        marker=dict(size=3, color="red", opacity=0.7),
        name="DCZ",
    ))

    fig.add_trace(go.Scatter3d(
        x=emb_saline[:, 0],
        y=emb_saline[:, 1],
        z=emb_saline[:, 2],
        mode="markers",
        marker=dict(size=3, color="blue", opacity=0.7),
        name="Saline",
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Latent 1",
            yaxis_title="Latent 2",
            zaxis_title="Latent 3",
        ),
        width=900,
        height=700,
        template="plotly_dark",
    )

    fig.show()


def plot_cebra_embeddings_separate(
    emb_dcz: np.ndarray,
    emb_saline: np.ndarray,
    title_prefix: str = "CEBRA Embedding",
) -> None:
    """
    Create two separate 3D interactive plots (DCZ and SALINE) colored by time.
    """
    def rainbow_colors(n: int) -> list[str]:
        cmap = cm.get_cmap("rainbow")
        colors = (cmap(np.linspace(0, 1, n))[:, :3] * 255).astype(np.uint8)
        return [f"rgb({r},{g},{b})" for r, g, b in colors]

    colors_dcz = rainbow_colors(len(emb_dcz))
    fig_dcz = go.Figure(data=[
        go.Scatter3d(
            x=emb_dcz[:, 0],
            y=emb_dcz[:, 1],
            z=emb_dcz[:, 2],
            mode="markers",
            marker=dict(size=2, color=colors_dcz),
        )
    ])
    fig_dcz.update_layout(
        title=f"{title_prefix} — DCZ",
        scene=dict(
            xaxis_title="Latent 1",
            yaxis_title="Latent 2",
            zaxis_title="Latent 3",
        ),
        width=650,
        height=600,
    )
    fig_dcz.show()

    colors_sal = rainbow_colors(len(emb_saline))
    fig_sal = go.Figure(data=[
        go.Scatter3d(
            x=emb_saline[:, 0],
            y=emb_saline[:, 1],
            z=emb_saline[:, 2],
            mode="markers",
            marker=dict(size=2, color=colors_sal),
        )
    ])
    fig_sal.update_layout(
        title=f"{title_prefix} — Saline",
        scene=dict(
            xaxis_title="Latent 1",
            yaxis_title="Latent 2",
            zaxis_title="Latent 3",
        ),
        width=650,
        height=600,
    )
    fig_sal.show()
