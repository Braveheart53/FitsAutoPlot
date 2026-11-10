# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------
_autoplot_gpu.py

Optional GPU arrays for the FitsAutoPlot suite, for NVIDIA and AMD alike.

Three GPU back ends are tried, then NumPy on the CPU:

* CuPy on CUDA     - NVIDIA GPUs (Windows, Linux).
* CuPy on ROCm     - AMD GPUs (Linux; the cupy-rocm builds or AMD's amd-cupy).
* PyOpenCL         - any OpenCL GPU: AMD, NVIDIA or Intel (Windows, Linux).
                     This is the AMD route on Windows, where ROCm CuPy does
                     not run.

Each operation also has a NumPy version, which gives the same result and is
used whenever the GPU would not help: no GPU, the GPU switched off, an
array below the size threshold, an array too large for the GPU memory, or
a failed start-up self-test.  At the suite's measured sizes (2026-11-09) the
CPU path is the one chosen: the largest FITS sort was 28 212 elements (0.04 s
for all 13 119 sorts of a 319-file batch), all 220 Touchstone transforms took
0.11 s, and an R&S average is a few sweeps of 1 001 to 100 001 points.  The
GPU pays off only above the thresholds below, which are estimates (no GPU
was available to measure them).

Operations (results identical to NumPy, or within 1e-9 dB / 1e-12 relative):

* argsort        - stable ascending sort order, NaN last (np.argsort(kind="stable")).
* power_mean_db  - average of dB values in the linear domain:
                   P_avg = 10 log10( (1/K) sum_k 10^(P_k/10) ), NaN left out.
* ifft_centered  - fftshift(ifft(ifftshift(x))) along one axis (time domain).

Select the back end with the environment variable AUTOPLOT_GPU:
auto (default: CuPy, then an OpenCL GPU), cuda, rocm, opencl (also accepts
an OpenCL CPU device, for testing with PoCL), or off.

Created on 2026-11-09

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216

# %%% Revisions
--------
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.0.1: First version - one GPU layer for NVIDIA and AMD.
Date: 2026-11-09
#              CuPy on CUDA (NVIDIA), CuPy on ROCm (AMD) and PyOpenCL (any
#              OpenCL GPU), with NumPy fallback, size thresholds and
#              start-up self-tests.
# %%%%% Function Descriptions
        GpuInfo (class): The GPU back end found by detect().
        _short_vendor: 'Advanced Micro Devices, Inc.' -> 'AMD', and so on.
        _detect_cupy: CuPy on CUDA (NVIDIA) or ROCm (AMD), or None with a note.
        _detect_opencl: The OpenCL GPU with the most memory (any OpenCL device when ``explicit``).
        detect: Find the GPU back end once (AUTOPLOT_GPU decides which); never raises.
        available: True when a GPU back end was found.
        describe: One line for logs and tooltips: the device in use, or why there is none.
        enable: Switch the GPU on or off for calls that pass use_gpu=None.
        enabled: True when the GPU is switched on and available.
        set_threshold: Smallest array (elements) for which ``operation`` uses the GPU.
        threshold: Current size threshold of ``operation`` (elements).
        usage: Calls per operation so far: {"argsort": {"gpu": n, "cpu": m}, ...}.
        usage_text: usage() as one line for the log, e.g.
        reset_usage: Set the usage counters to zero.
        _count: Add one call of ``operation`` on ``where`` ("gpu" or "cpu").
        _cupy: The CuPy module (only called once detect() found it).
        _cl_queue: (pyopencl, context, queue) of the detected OpenCL device, made once.
        _free_cupy: Return CuPy's cached device memory, so the next file does not run out.
        _supported: True when ``operation`` has a GPU version for this back end.
        _use_gpu: Decide whether one call runs on the GPU.
        _fail: Use NumPy for ``operation`` from now on (logged through describe()).
        sort_keys: Unsigned 64-bit keys whose ascending order is NumPy's sort order.
        _cl_argsort: Stable argsort on an OpenCL device (radix sort of the 64-bit keys).
        _cupy_argsort: Stable argsort with CuPy (CUDA or ROCm) on the 64-bit keys.
        argsort: Stable ascending sort order (int64), the same as np.argsort(kind="stable").
        _quiet: np.errstate(all="ignore") for NumPy; nothing for CuPy (it does not warn).
        power_mean_db_xp: power_mean_db for one array module ``xp`` (numpy or cupy); averages axis 0.
        _cl_power_mean: power_mean_db on an OpenCL device (double precision when it has fp64).
        power_mean_db: Mean of dB values taken in the linear domain, returned in dB.
        _numpy_ifft_centered: fftshift(ifft(ifftshift(x))) along ``axis`` with NumPy (the reference).
        _cl_ifft_centered: The same with pyvkfft on the OpenCL device (shifts on the host).
        ifft_centered: fftshift(ifft(ifftshift(x))) along ``axis``: t = 0 at the centre sample.
        _test_values: Random values with ties, NaN, +-inf and both zeros (the hard cases).
        _self_test: Run ``operation`` on the GPU with a small input and compare with NumPy.
# %%%%% Variable Descriptions
    Define all utilized variables
        GPU_ENV: environment variable that selects the back end (auto, cuda, rocm, opencl, off)
        BACKEND_CHOICES: accepted values of AUTOPLOT_GPU
        THRESHOLDS: smallest array (elements) per operation for which the GPU is used (estimates)
        MEMORY_FRACTION: largest share of the GPU memory one call may use
        CHUNK_ROWS: rows per block in power_mean_db (bounds the temporary memory)
        SIGN_BIT: 64-bit sign bit of the sort keys
        ALL_ONES: largest 64-bit key (NaN sorts last)
        _LOCK: one GPU call at a time (GUI and worker threads)
        _STATE: detected GpuInfo and the on/off switch
        _FAILED: operation -> why it fell back to NumPy for good
        _TESTED: operation -> self-test result
        _USAGE: GPU / CPU calls per operation
        _CL: OpenCL device, context, queue and compiled kernels
        _CL_RADIX_ARGS: argument list of the OpenCL radix sort
        _CL_POWER_MEAN: OpenCL C source of the linear-domain dB mean kernel
# %%%%% More Info
    Install one of (see REQUIREMENTS.md section 2.2):
        NVIDIA:              pip install cupy-cuda12x        (or cupy-cuda13x)
        AMD on Linux:        pip install cupy-rocm-7-0       (Python 3.10+)
        AMD / any, Windows:  pip install pyopencl            (+ the GPU driver's OpenCL)
        optional FFT on OpenCL: pyvkfft (not tested on hardware: the self-test
        decides whether it is used)
    Python 3.8 to 3.13; no Qt.  Thread-safe (one lock around GPU calls).
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
=============================================================================
"""
# %% Imports
from __future__ import annotations

__version__ = "0.0.1"          # Semantic Versioning: see the Revisions entries above

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# %% Settings
GPU_ENV = "AUTOPLOT_GPU"
BACKEND_CHOICES = ("auto", "cuda", "rocm", "opencl", "off")
# Estimated break-even sizes: below these the host <-> device copy and the
# kernel launch cost more than the GPU saves.  argsort: public CuPy/NumPy
# sort benchmarks on mid-range GPUs; the others: about 1 ms of copying.
THRESHOLDS: Dict[str, int] = {
    "argsort": 200_000,
    "power_mean_db": 1_000_000,
    "ifft_centered": 1_000_000,
}
MEMORY_FRACTION = 0.25
CHUNK_ROWS = 128
SIGN_BIT = np.uint64(1 << 63)
ALL_ONES = np.uint64(0xFFFFFFFFFFFFFFFF)

_LOCK = threading.RLock()                 # one GPU call at a time (GUI + worker threads)
_STATE: Dict[str, Any] = {"info": None, "enabled": False}
_FAILED: Dict[str, str] = {}              # operation -> why it fell back to NumPy for good
_TESTED: Dict[str, bool] = {}             # operation -> self-test result
_USAGE: Dict[str, Dict[str, int]] = {}
_CL: Dict[str, Any] = {}                  # OpenCL context, queue, compiled kernels


# %% Detection
@dataclass
class GpuInfo:
    """The GPU back end found by detect().

    backend is "cuda", "rocm", "opencl" or "" (none); notes say why other
    back ends were not used.
    """
    backend: str = ""
    library: str = ""
    device: str = ""
    vendor: str = ""
    memory_mb: int = 0
    fp64: bool = True
    is_gpu: bool = True
    notes: List[str] = field(default_factory=list)


def _short_vendor(text: str) -> str:
    """'Advanced Micro Devices, Inc.' -> 'AMD', and so on."""
    low = text.lower()
    for key, name in (("advanced micro", "AMD"), ("amd", "AMD"), ("nvidia", "NVIDIA"),
                      ("intel", "Intel"), ("apple", "Apple")):
        if key in low:
            return name
    return text.strip()


def _detect_cupy(choice: str, notes: List[str]) -> Optional[GpuInfo]:
    """CuPy on CUDA (NVIDIA) or ROCm (AMD), or None with a note."""
    try:
        import cupy                                   # optional
    except Exception as exc:
        notes.append("CuPy not installed (%s)" % type(exc).__name__)
        return None
    try:
        count = int(cupy.cuda.runtime.getDeviceCount())
    except Exception as exc:
        notes.append("CuPy %s found no GPU (%s)" % (cupy.__version__, exc))
        return None
    if count < 1:
        notes.append("CuPy %s found no GPU" % cupy.__version__)
        return None
    backend = "rocm" if bool(getattr(cupy.cuda.runtime, "is_hip", False)) else "cuda"
    if choice in ("cuda", "rocm") and choice != backend:
        notes.append("CuPy is built for %s, not %s" % (backend.upper(), choice.upper()))
        return None
    name, memory = "GPU 0", 0
    try:
        props = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.Device().id)
        name = props.get("name", b"GPU 0")
        if isinstance(name, bytes):
            name = name.decode("ascii", "replace")
        memory = int(props.get("totalGlobalMem", 0)) >> 20
    except Exception:
        pass
    return GpuInfo(backend=backend, library="CuPy %s" % cupy.__version__,
                   device=str(name).strip(), vendor="AMD" if backend == "rocm" else "NVIDIA",
                   memory_mb=memory, fp64=True, is_gpu=True, notes=notes)


def _detect_opencl(explicit: bool, notes: List[str]) -> Optional[GpuInfo]:
    """The OpenCL GPU with the most memory (any OpenCL device when ``explicit``)."""
    try:
        import pyopencl as cl                         # optional
    except Exception as exc:
        notes.append("PyOpenCL not installed (%s)" % type(exc).__name__)
        return None
    devices = []
    try:
        if os.environ.get("PYOPENCL_CTX"):            # the user's own choice
            devices = list(cl.create_some_context(interactive=False).devices)
        else:
            for platform in cl.get_platforms():
                devices.extend(platform.get_devices())
    except Exception as exc:
        notes.append("OpenCL found no platform (%s)" % exc)
        return None
    gpus = [d for d in devices if d.type & cl.device_type.GPU]
    pick = gpus or (devices if explicit else [])
    if not pick:
        notes.append("OpenCL found no GPU" + (" (a CPU device is used only with %s=opencl)"
                                              % GPU_ENV if devices else ""))
        return None
    device = max(pick, key=lambda d: (bool(d.type & cl.device_type.GPU), int(d.global_mem_size)))
    _CL.clear()
    _CL["device"] = device
    return GpuInfo(backend="opencl", library="PyOpenCL %s" % cl.VERSION_TEXT,
                   device=device.name.strip(), vendor=_short_vendor(device.vendor),
                   memory_mb=int(device.global_mem_size) >> 20,
                   fp64=("cl_khr_fp64" in device.extensions or "cl_amd_fp64" in device.extensions),
                   is_gpu=bool(device.type & cl.device_type.GPU), notes=notes)


def detect(refresh: bool = False) -> GpuInfo:
    """Find the GPU back end once (AUTOPLOT_GPU decides which); never raises.

    auto: CuPy (CUDA or ROCm), then an OpenCL GPU.  The libraries are
    imported here, not when this module is imported, so programs and their
    'spawn' workers start quickly.
    """
    with _LOCK:
        if _STATE["info"] is not None and not refresh:
            return _STATE["info"]
        notes: List[str] = []
        choice = (os.environ.get(GPU_ENV, "auto").strip().lower() or "auto")
        if choice not in BACKEND_CHOICES:
            notes.append("%s=%s is not one of %s; auto used" % (GPU_ENV, choice, "/".join(BACKEND_CHOICES)))
            choice = "auto"
        info = None
        if choice == "off":
            notes.append("%s=off" % GPU_ENV)
        else:
            if choice in ("auto", "cuda", "rocm"):
                info = _detect_cupy(choice, notes)
            if info is None and choice in ("auto", "opencl"):
                info = _detect_opencl(choice == "opencl", notes)
        if info is None:
            info = GpuInfo(notes=notes)
        _STATE["info"] = info
        _FAILED.clear()
        _TESTED.clear()
        if not info.backend:
            _STATE["enabled"] = False
        return info


def available() -> bool:
    """True when a GPU back end was found."""
    return bool(detect().backend)


def describe() -> str:
    """One line for logs and tooltips: the device in use, or why there is none."""
    info = detect()
    if not info.backend:
        return "No GPU: %s. NumPy on the CPU is used." % ("; ".join(info.notes) or "none found")
    names = {"cuda": "CUDA", "rocm": "ROCm", "opencl": "OpenCL"}
    text = "%s on %s: %s %s" % (info.library, names[info.backend], info.vendor, info.device)
    extras = []
    if info.memory_mb:
        extras.append("%.1f GB" % (info.memory_mb / 1024.0))
    if not info.fp64:
        extras.append("single precision only")
    if not info.is_gpu:
        extras.append("CPU device, for testing")
    if extras:
        text += " (%s)" % ", ".join(extras)
    if _FAILED:
        text += "; NumPy for %s" % ", ".join("%s (%s)" % kv for kv in sorted(_FAILED.items()))
    return text


def enable(flag: bool) -> None:
    """Switch the GPU on or off for calls that pass use_gpu=None."""
    _STATE["enabled"] = bool(flag) and available()


def enabled() -> bool:
    """True when the GPU is switched on and available."""
    return bool(_STATE["enabled"])


def set_threshold(operation: str, elements: int) -> None:
    """Smallest array (elements) for which ``operation`` uses the GPU."""
    if operation not in THRESHOLDS:
        raise KeyError("unknown operation %r (one of %s)" % (operation, ", ".join(THRESHOLDS)))
    THRESHOLDS[operation] = max(0, int(elements))


def threshold(operation: str) -> int:
    """Current size threshold of ``operation`` (elements)."""
    return int(THRESHOLDS[operation])


def usage() -> Dict[str, Dict[str, int]]:
    """Calls per operation so far: {"argsort": {"gpu": n, "cpu": m}, ...}."""
    return {op: dict(counts) for op, counts in _USAGE.items()}


def usage_text() -> str:
    """usage() as one line for the log, e.g. 'argsort: 0 GPU / 13119 CPU'."""
    parts = ["%s: %d GPU / %d CPU" % (op, c.get("gpu", 0), c.get("cpu", 0))
             for op, c in sorted(_USAGE.items())]
    return "; ".join(parts) if parts else "no GPU-capable operations yet"


def reset_usage() -> None:
    """Set the usage counters to zero."""
    _USAGE.clear()


def _count(operation: str, where: str) -> None:
    """Add one call of ``operation`` on ``where`` ("gpu" or "cpu")."""
    counts = _USAGE.setdefault(operation, {"gpu": 0, "cpu": 0})
    counts[where] = counts.get(where, 0) + 1


# %% Back-end plumbing
def _cupy():
    """The CuPy module (only called once detect() found it)."""
    import cupy
    return cupy


def _cl_queue():
    """(pyopencl, context, queue) of the detected OpenCL device, made once."""
    import pyopencl as cl
    if "queue" not in _CL:
        context = cl.Context([_CL["device"]])
        _CL["context"] = context
        _CL["queue"] = cl.CommandQueue(context)
    return cl, _CL["context"], _CL["queue"]


def _free_cupy() -> None:
    """Return CuPy's cached device memory, so the next file does not run out."""
    try:
        _cupy().get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def _supported(operation: str, info: GpuInfo) -> bool:
    """True when ``operation`` has a GPU version for this back end."""
    if info.backend in ("cuda", "rocm"):
        return True
    if info.backend == "opencl":
        if operation == "ifft_centered":
            if not info.fp64:
                return False
            try:
                import pyvkfft.fft  # noqa: F401  (optional, OpenCL FFT)
                return True
            except Exception:
                return False
        return True
    return False


def _use_gpu(operation: str, elements: int, nbytes: int,
             use_gpu: Optional[bool], min_elements: Optional[int]) -> bool:
    """Decide whether one call runs on the GPU.

    use_gpu None follows enable(); True asks for the GPU; False never.
    min_elements None uses THRESHOLDS; 0 ignores the size.
    """
    if use_gpu is False or (use_gpu is None and not enabled()):
        return False
    info = detect()
    if not info.backend or operation in _FAILED or not _supported(operation, info):
        return False
    limit = THRESHOLDS[operation] if min_elements is None else int(min_elements)
    if elements < max(1, limit):
        return False
    if info.memory_mb and nbytes > MEMORY_FRACTION * info.memory_mb * 1024 * 1024:
        return False
    if operation not in _TESTED:
        _TESTED[operation] = _self_test(operation)
    return _TESTED[operation]


def _fail(operation: str, reason: str) -> None:
    """Use NumPy for ``operation`` from now on (logged through describe())."""
    _FAILED[operation] = reason[:120]


# %% Sorting
def sort_keys(xp: Any, values: Any) -> Any:
    """Unsigned 64-bit keys whose ascending order is NumPy's sort order.

    Floats: the IEEE-754 bit pattern with the sign handled so that integer
    order is numeric order; -0.0 and +0.0 give the same key (NumPy treats
    them as equal) and every NaN gets the largest key (NumPy puts NaN last).
    Signed integers: the sign bit flipped.  Unsigned integers and booleans:
    unchanged.  ``xp`` is numpy or cupy, so the same code runs on both.
    """
    kind = values.dtype.kind
    if kind == "f":
        x = values.astype(xp.float64, copy=False)
        bits = xp.where(x == 0, xp.float64(0.0), x).view(xp.uint64)
        negative = (bits >> xp.uint64(63)) == xp.uint64(1)
        keys = xp.where(negative, ~bits, bits | SIGN_BIT)
        return xp.where(xp.isnan(x), ALL_ONES, keys)
    if kind == "i":
        return values.astype(xp.int64, copy=False).view(xp.uint64) ^ SIGN_BIT
    return values.astype(xp.uint64, copy=False)


_CL_RADIX_ARGS = "ulong *keys, uint *index"


def _cl_argsort(values: np.ndarray) -> np.ndarray:
    """Stable argsort on an OpenCL device (radix sort of the 64-bit keys)."""
    import pyopencl.array as cla
    cl, context, queue = _cl_queue()
    if "radix" not in _CL:
        from pyopencl.algorithm import RadixSort      # needs Mako
        _CL["radix"] = RadixSort(context, _CL_RADIX_ARGS, key_expr="keys[i]",
                                 sort_arg_names=["keys", "index"], key_dtype=np.uint64)
    keys = cla.to_device(queue, np.ascontiguousarray(sort_keys(np, values)))
    index = cla.arange(queue, values.size, dtype=np.uint32)
    (_keys, order), event = _CL["radix"](keys, index, key_bits=64)
    event.wait()
    return order.get().astype(np.int64)


def _cupy_argsort(values: np.ndarray) -> np.ndarray:
    """Stable argsort with CuPy (CUDA or ROCm) on the 64-bit keys."""
    cp = _cupy()
    try:
        order = cp.argsort(sort_keys(cp, cp.asarray(values)))
        return cp.asnumpy(order).astype(np.int64, copy=False)
    finally:
        _free_cupy()


def argsort(values: Any, use_gpu: Optional[bool] = None,
            min_elements: Optional[int] = None) -> np.ndarray:
    """Stable ascending sort order (int64), the same as np.argsort(kind="stable").

    Parameters
    ----------
    values : array_like
        1-D numbers (other shapes and dtypes always use NumPy).
    use_gpu : bool, optional
        None follows enable(); True asks for the GPU; False never.
    min_elements : int, optional
        GPU only from this size (default THRESHOLDS["argsort"]; 0 = any size).
    """
    a = np.asarray(values)
    if a.ndim == 1 and a.dtype.kind in "fiub" and a.size < 2 ** 32 and \
            _use_gpu("argsort", a.size, a.size * 24, use_gpu, min_elements):
        try:
            with _LOCK:
                order = (_cl_argsort(a) if detect().backend == "opencl" else _cupy_argsort(a))
            _count("argsort", "gpu")
            return order
        except Exception as exc:
            _fail("argsort", "%s: %s" % (type(exc).__name__, exc))
    _count("argsort", "cpu")
    return np.argsort(a, kind="stable")


# %% Linear-domain averages of dB values
def _quiet(xp: Any):
    """np.errstate(all="ignore") for NumPy; nothing for CuPy (it does not warn)."""
    if xp is np:
        return np.errstate(all="ignore")
    import contextlib
    return contextlib.nullcontext()


def power_mean_db_xp(xp: Any, data: Any, factor: float = 10.0,
                     chunk_rows: int = CHUNK_ROWS) -> Any:
    """power_mean_db for one array module ``xp`` (numpy or cupy); averages axis 0.

    peak  = max of the non-NaN values (the reference keeps 10^(...) <= 1,
            so nothing overflows)
    P_avg = peak + factor * log10( (1/K) sum_k 10^((P_k - peak) / factor) )
    K counts the non-NaN values.  -inf (zero power) counts and adds
    nothing; +inf gives +inf; a column without values gives NaN.  Rows are
    processed CHUNK_ROWS at a time, so the temporary arrays stay small.
    """
    rows = data.shape[0]
    peak = xp.full(data.shape[1:], -xp.inf)
    count = xp.zeros(data.shape[1:], dtype=xp.int64)
    for first in range(0, rows, chunk_rows):
        block = data[first:first + chunk_rows]
        valid = ~xp.isnan(block)
        count += valid.sum(axis=0)
        peak = xp.maximum(peak, xp.where(valid, block, -xp.inf).max(axis=0))
    reference = xp.where(xp.isfinite(peak), peak, 0.0)
    total = xp.zeros(data.shape[1:])
    with _quiet(xp):
        for first in range(0, rows, chunk_rows):
            block = data[first:first + chunk_rows]
            linear = xp.power(10.0, (block - reference) / factor)
            total += xp.where(xp.isnan(block), 0.0, linear).sum(axis=0)
        return reference + factor * xp.log10(total / count)


_CL_POWER_MEAN = r"""
#ifdef USE_DOUBLE
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double real;
#else
typedef float real;
#endif
/* One work item per output column j: linear-domain mean of the K values
   x[i*m + j], NaN left out, referenced to the column maximum. */
__kernel void power_mean(__global const real *x, const uint k, const uint m,
                         const real factor, __global real *out)
{
    const uint j = get_global_id(0);
    if (j >= m) return;
    real peak = -INFINITY;
    uint count = 0;
    for (uint i = 0; i < k; ++i) {
        const real v = x[(size_t)i * m + j];
        if (!isnan(v)) { ++count; peak = fmax(peak, v); }
    }
    if (count == 0) { out[j] = NAN; return; }
    const real ref = isfinite(peak) ? peak : (real)0;
    real total = 0;
    for (uint i = 0; i < k; ++i) {
        const real v = x[(size_t)i * m + j];
        if (!isnan(v)) total += exp10((v - ref) / factor);
    }
    out[j] = ref + factor * log10(total / (real)count);
}
"""


def _cl_power_mean(data: np.ndarray, factor: float) -> np.ndarray:
    """power_mean_db on an OpenCL device (double precision when it has fp64)."""
    import pyopencl.array as cla
    cl, context, queue = _cl_queue()
    fp64 = detect().fp64
    key = "power_mean_%d" % fp64
    if key not in _CL:
        options = ["-DUSE_DOUBLE"] if fp64 else []
        _CL[key] = cl.Program(context, _CL_POWER_MEAN).build(options=options).power_mean
    real = np.float64 if fp64 else np.float32
    rows = data.shape[0]
    flat = np.ascontiguousarray(data.reshape(rows, -1), dtype=real)
    columns = flat.shape[1]
    x = cla.to_device(queue, flat)
    out = cla.empty(queue, columns, real)
    _CL[key](queue, (columns,), None, x.data, np.uint32(rows), np.uint32(columns),
             real(factor), out.data)
    return out.get().astype(np.float64).reshape(data.shape[1:])


def power_mean_db(stack: Any, axis: int = 0, factor: float = 10.0,
                  use_gpu: Optional[bool] = None,
                  min_elements: Optional[int] = None) -> np.ndarray:
    """Mean of dB values taken in the linear domain, returned in dB.

        P_avg = 10 log10( (1/K) sum_k 10^(P_k / 10) )

    for power levels (dB, dBm, dBW, dBuV, dBc ...; ``factor`` 10).  Use
    ``factor`` 20 to average amplitudes (|S|, volts) instead of powers.
    NaN samples are left out; a column with no samples gives NaN.

    Parameters
    ----------
    stack : array_like
        dB values; ``axis`` is the one averaged (removed from the result).
    use_gpu, min_elements : see argsort().
    """
    data = np.asarray(stack, dtype=np.float64)
    if data.ndim == 0:
        raise ValueError("power_mean_db needs at least one axis to average")
    data = np.moveaxis(data, axis, 0)
    if data.shape[0] == 0:
        return np.full(data.shape[1:], np.nan)
    if data.size < 2 ** 32 and _use_gpu("power_mean_db", data.size, data.size * 16,
                                        use_gpu, min_elements):
        try:
            with _LOCK:
                if detect().backend == "opencl":
                    result = _cl_power_mean(data, factor)
                else:
                    cp = _cupy()
                    try:
                        result = cp.asnumpy(power_mean_db_xp(cp, cp.asarray(data), factor))
                    finally:
                        _free_cupy()
            _count("power_mean_db", "gpu")
            return result
        except Exception as exc:
            _fail("power_mean_db", "%s: %s" % (type(exc).__name__, exc))
    _count("power_mean_db", "cpu")
    return power_mean_db_xp(np, data, factor)


# %% Fourier transforms
def _numpy_ifft_centered(x: np.ndarray, axis: int) -> np.ndarray:
    """fftshift(ifft(ifftshift(x))) along ``axis`` with NumPy (the reference)."""
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis)


def _cl_ifft_centered(x: np.ndarray, axis: int) -> np.ndarray:
    """The same with pyvkfft on the OpenCL device (shifts on the host)."""
    import pyopencl.array as cla
    from pyvkfft.fft import ifftn
    _cl, _context, queue = _cl_queue()
    shifted = np.ascontiguousarray(np.fft.ifftshift(x, axes=axis), dtype=np.complex128)
    device = cla.to_device(queue, shifted)
    result = ifftn(device, axes=(axis % x.ndim,))     # norm=1: 1/N on the inverse, as NumPy
    return np.fft.fftshift(result.get(), axes=axis)


def ifft_centered(x: Any, axis: int = 0, use_gpu: Optional[bool] = None,
                  min_elements: Optional[int] = None) -> np.ndarray:
    """fftshift(ifft(ifftshift(x))) along ``axis``: t = 0 at the centre sample.

    The band-pass time-domain transform of scikit-rf.  CuPy (cuFFT on
    NVIDIA, hipFFT/rocFFT on AMD) or pyvkfft (OpenCL) for large arrays;
    NumPy otherwise.  use_gpu, min_elements: see argsort().
    """
    a = np.asarray(x)
    if _use_gpu("ifft_centered", a.size, a.size * 32, use_gpu, min_elements):
        try:
            with _LOCK:
                if detect().backend == "opencl":
                    result = _cl_ifft_centered(a, axis)
                else:
                    cp = _cupy()
                    try:
                        g = cp.asarray(a)
                        result = cp.asnumpy(cp.fft.fftshift(
                            cp.fft.ifft(cp.fft.ifftshift(g, axes=axis), axis=axis), axes=axis))
                    finally:
                        _free_cupy()
            _count("ifft_centered", "gpu")
            return result
        except Exception as exc:
            _fail("ifft_centered", "%s: %s" % (type(exc).__name__, exc))
    _count("ifft_centered", "cpu")
    return _numpy_ifft_centered(a, axis)


# %% Self-tests (run once per operation, on first GPU use)
def _test_values(n: int = 4099) -> np.ndarray:
    """Random values with ties, NaN, +-inf and both zeros (the hard cases)."""
    rng = np.random.default_rng(20261109)
    values = np.round(rng.normal(0.0, 30.0, n), 1)
    values[::17] = np.nan
    values[3::29] = -0.0
    values[5::31] = 0.0
    values[7::37] = np.inf
    values[11::41] = -np.inf
    return values


def _self_test(operation: str) -> bool:
    """Run ``operation`` on the GPU with a small input and compare with NumPy.

    A mismatch or an error switches the operation to NumPy for good (shown
    in describe()), so an untested driver or library can never change a
    result.
    """
    try:
        with _LOCK:
            backend = detect().backend
            if operation == "argsort":
                values = _test_values()
                got = _cl_argsort(values) if backend == "opencl" else _cupy_argsort(values)
                ok = np.array_equal(got, np.argsort(values, kind="stable"))
            elif operation == "power_mean_db":
                stack = _test_values(64 * 37).reshape(64, 37)
                stack[:, 4] = np.nan                       # an empty column
                stack[:, 6] = -np.inf                      # zero power
                if backend == "opencl":
                    got = _cl_power_mean(stack, 10.0)
                else:
                    cp = _cupy()
                    got = cp.asnumpy(power_mean_db_xp(cp, cp.asarray(stack), 10.0))
                ref = power_mean_db_xp(np, stack, 10.0)
                tol = 1e-9 if detect().fp64 else 1e-3
                finite = np.isfinite(ref)
                ok = (np.array_equal(np.isnan(got), np.isnan(ref))
                      and np.array_equal(got[np.isinf(ref)], ref[np.isinf(ref)])
                      and bool(np.all(np.abs(got[finite] - ref[finite]) <= tol)))
            else:
                rng = np.random.default_rng(1)
                x = rng.normal(size=(257, 2, 2)) + 1j * rng.normal(size=(257, 2, 2))
                if backend == "opencl":
                    got = _cl_ifft_centered(x, 0)
                else:
                    cp = _cupy()
                    g = cp.asarray(x)
                    got = cp.asnumpy(cp.fft.fftshift(cp.fft.ifft(cp.fft.ifftshift(g, axes=0), axis=0), axes=0))
                ref = _numpy_ifft_centered(x, 0)
                ok = bool(np.allclose(got, ref, rtol=1e-12, atol=1e-14))
        if backend in ("cuda", "rocm"):
            _free_cupy()
    except Exception as exc:
        _fail(operation, "self-test %s: %s" % (type(exc).__name__, exc))
        return False
    if not ok:
        _fail(operation, "self-test result differs from NumPy")
    return ok
