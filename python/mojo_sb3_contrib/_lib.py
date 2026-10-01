"""Loads (and if necessary builds) the compiled Mojo library.

The kernels are all float64. Inputs are cast up, the arithmetic runs in
float64, and scalar results are cast back to the dtype they came in with, so a
float32 caller gets a float32 result back without changing the call site.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")

# `MOJO_SB3_CONTRIB_LIB` points at an already-built shared library. Two cases
# need it: an ordinary (non-editable) install, where the package is copied into
# site-packages and ROOT no longer names the checkout that holds src/ and
# build/; and any server that must never pay a `mojo build` at call time.
LIB = os.environ.get("MOJO_SB3_CONTRIB_LIB") or os.path.join(
    ROOT, "dist", "libmojo-sb3-contrib.so"
)

I = ctypes.c_int64
F = ctypes.c_double

# name -> (argtypes, restype)
_SIGNATURES = {
    "sb3c_quantile_huber_loss": (
        [I, I, I, I, I, I, I, I, ctypes.c_bool, ctypes.c_bool, I, I, I],
        F,
    ),
    "sb3c_quantile_huber_loss_gpu": (
        [I, I, I, I, I, I, I, I, I, ctypes.c_bool],
        F,
    ),
    "sb3c_conjugate_gradient_solver": ([I, I, I, I, I, I, F], I),
    "sb3c_apply_masking": ([I, I, I, I], None),
    "sb3c_entropy": ([I, I, I, I, ctypes.c_bool], F),
    "sb3c_batch_renorm_forward": (
        [I, I, I, I, I, I, I, I, I, ctypes.c_bool, I, F, F, ctypes.c_bool, F, F, I],
        I,
    ),
    "sb3c_qrdqn_target_quantiles": ([I, I, I, I, I, I, I, I, I], I),
    "sb3c_tqc_target_quantiles": ([I, I, I, I, I, I, I, I, I, I, I, F], I),
    "sb3c_crossq_target_q_values": ([I, I, I, I, I, I, I, I, F], I),
    "sb3c_crossq_critic_loss": ([I, I, I, I], F),
    "sb3c_ars_top_returns": ([I, I, I, I, I, I], I),
    "sb3c_ars_gather": ([I, I, I, I, I, I, I, I, I], I),
    "sb3c_ars_update_weights": ([I, I, I, I, I, I, F, F, I, I], F),
}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJO_SB3_CONTRIB_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi) and os.path.exists(os.path.join(ROOT, "pixi.toml")):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJO_SB3_CONTRIB_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    """Compile `src/capi.mojo` into `build/capi.so` if missing or stale."""
    if os.environ.get("MOJO_SB3_CONTRIB_LIB") and os.path.exists(LIB) and not force:
        # An explicitly supplied library is taken as given: the sources it was
        # built from need not be present at all, so there is nothing to compare
        # mtimes against.
        return LIB
    if not os.path.isdir(SRC):
        if os.path.exists(LIB):
            return LIB
        raise BuildError(
            f"no Mojo sources at {SRC} and no library at {LIB}; this looks like a "
            "copied install - point MOJO_SB3_CONTRIB_LIB at a built capi.so"
        )
    sources = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        newest = max(os.path.getmtime(s) for s in sources)
        if os.path.getmtime(LIB) >= newest:
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + [
        "build",
        "--emit",
        "shared-lib",
        "-I",
        SRC,
        os.path.join(SRC, "capi.mojo"),
        "-o",
        LIB,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib


def f64(a) -> np.ndarray:
    """A C-contiguous float64 view of `a`."""
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.ascontiguousarray(a, dtype=np.float64)


def scratch(n: int) -> np.ndarray:
    return np.zeros(max(n, 1), dtype=np.float64)


def main() -> int:
    """`python -m mojo_sb3_contrib._lib` rebuilds the library."""
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
