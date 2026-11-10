# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------
tests/test_parallel_gpu.py

Tests of the CPU process pool (_autoplot_parallel) and the GPU layer
(_autoplot_gpu): NVIDIA and AMD back ends must give exactly the NumPy
results, and dB values must always be averaged in the linear domain.

* The NumPy versions are checked against hand-computed values and against
  vszh5_writer.linear_power_mean_dbm.
* The CuPy code (CUDA and ROCm) is run with a stand-in "cupy" module built
  on NumPy, because no GPU is available here: it checks the selection, the
  self-test, the fallback and the code path, not GPU arithmetic.
* The OpenCL code runs for real when PyOpenCL finds a device (PoCL on the
  CPU was used on 2026-11-09); otherwise those tests are skipped.

Run:  python -m unittest discover -s tests

Created on 2026-11-09

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216

# %%% Revisions
--------
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.0.1: First tests of the process pool and the GPU layer.
Date: 2026-11-09
#              Adaptive pool, GPU selection, sort keys, linear-domain dB
#              means, CuPy stand-in, OpenCL (PoCL).
# %%%%% Function Descriptions
        fake_cupy: A stand-in 'cupy' module built on NumPy (no GPU needed).
        opencl_device_found: True when PyOpenCL is installed and lists a device.
        GpuState (class): Restore _autoplot_gpu's selection, thresholds and counters after a test.
        ParallelTests (class): _autoplot_parallel: pool decision, order, callbacks, spawn safety.
        SortKeyTests (class): sort_keys() orders exactly like np.argsort(kind="stable").
        PowerMeanTests (class): power_mean_db(): the mean of dB values is taken in the linear domain.
        CupyBackendTests (class): The CuPy (CUDA / ROCm) code path with the NumPy stand-in.
        OpenCLBackendTests (class): The OpenCL code path on a real OpenCL device (PoCL on the CPU is enough).
# %%%%% Variable Descriptions
    Define all utilized variables
        ROOT: repository folder
# %%%%% More Info
    See README.md and REQUIREMENTS.md.
=============================================================================
"""
# %% Imports
import os
import sys
import tempfile
import types
import unittest
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import _autoplot_gpu as gpu                               # noqa: E402
import _autoplot_parallel as par                          # noqa: E402


# %% Helpers
def fake_cupy(is_hip: bool = False, wrong_argsort: bool = False) -> types.ModuleType:
    """A stand-in 'cupy' module built on NumPy (no GPU needed).

    ``is_hip`` makes it look like a ROCm (AMD) build; ``wrong_argsort``
    makes its argsort return a wrong order, to test the self-test.
    """
    fake = types.ModuleType("cupy")
    fake.__version__ = "0.0-test"

    def __getattr__(name):
        """Everything not defined below comes from NumPy."""
        return getattr(np, name)

    fake.__getattr__ = __getattr__
    runtime = types.SimpleNamespace(
        getDeviceCount=lambda: 1, is_hip=is_hip,
        getDeviceProperties=lambda index: {"name": b"Test GPU", "totalGlobalMem": 8 << 30})
    fake.cuda = types.SimpleNamespace(runtime=runtime,
                                      Device=lambda *args: types.SimpleNamespace(id=0))
    fake.asarray = np.asarray
    fake.asnumpy = np.asarray
    fake.fft = np.fft
    fake.get_default_memory_pool = lambda: types.SimpleNamespace(free_all_blocks=lambda: None)
    if wrong_argsort:
        fake.argsort = lambda a: np.argsort(a, kind="stable")[::-1].copy()
    else:
        fake.argsort = lambda a: np.argsort(a, kind="stable")
    return fake


def opencl_device_found() -> bool:
    """True when PyOpenCL is installed and lists a device."""
    try:
        import pyopencl as cl
        return any(p.get_devices() for p in cl.get_platforms())
    except Exception:
        return False


class GpuState(unittest.TestCase):
    """Restore _autoplot_gpu's selection, thresholds and counters after a test."""

    def setUp(self):
        """Remember the environment, thresholds and any real cupy module."""
        self._env = os.environ.get(gpu.GPU_ENV)
        self._thresholds = dict(gpu.THRESHOLDS)
        self._cupy = sys.modules.get("cupy")

    def tearDown(self):
        """Put everything back and detect again."""
        if self._env is None:
            os.environ.pop(gpu.GPU_ENV, None)
        else:
            os.environ[gpu.GPU_ENV] = self._env
        if self._cupy is None:
            sys.modules.pop("cupy", None)
        else:
            sys.modules["cupy"] = self._cupy
        gpu.THRESHOLDS.clear()
        gpu.THRESHOLDS.update(self._thresholds)
        gpu.enable(False)
        gpu.detect(refresh=True)
        gpu.reset_usage()

    def select(self, choice, module=None):
        """Set AUTOPLOT_GPU (and the cupy module) and detect again."""
        os.environ[gpu.GPU_ENV] = choice
        if module is not None:
            sys.modules["cupy"] = module
        info = gpu.detect(refresh=True)
        gpu.reset_usage()
        return info


# %% CPU process pool
class ParallelTests(unittest.TestCase):
    """_autoplot_parallel: pool decision, order, callbacks, spawn safety."""

    def test_pool_decision(self):
        """t_seq - (t_start + 1.1 t_seq / W) >= 0.5 s decides."""
        self.assertTrue(par.should_use_pool(0.1, 100, 4, start_s=0.3))      # 10 s vs 3.05 s
        self.assertFalse(par.should_use_pool(0.001, 100, 4, start_s=0.3))   # 0.1 s
        self.assertFalse(par.should_use_pool(1.0, 100, 1, start_s=0.3))     # one worker
        self.assertFalse(par.should_use_pool(1.0, 1, 4, start_s=0.3))       # one item
        self.assertFalse(par.should_use_pool(0.05, 40, 2, start_s=1.0))     # 2 s vs 2.1 s

    def test_in_process_order_and_callbacks(self):
        """force=False: results in input order; progress and on_result for every item."""
        progress, seen, log = [], [], []
        out = par.adaptive_map(abs, [3, -1, -4, 1, -5], 4, progress=lambda d, t: progress.append((d, t)),
                               log=log.append, force=False, on_result=lambda i, r: seen.append(i))
        self.assertEqual(out, [3, 1, 4, 1, 5])
        self.assertEqual(progress[-1], (5, 5))
        self.assertEqual(sorted(seen), [0, 1, 2, 3, 4])
        self.assertIn("one process", log[0])

    def test_process_pool_keeps_order(self):
        """force=True: a 'spawn' pool runs the items after the probe; order kept."""
        paths = ["/a/%02d.txt" % i for i in range(12)]
        log = []
        out = par.adaptive_map(os.path.basename, paths, 2, log=log.append, force=True)
        self.assertEqual(out, [os.path.basename(p) for p in paths])
        if par.main_module_is_spawn_safe():
            self.assertIn("process pool of 2", log[-1])

    def test_spawn_safety_of_main_script(self):
        """A main script without the __main__ guard never gets a pool."""
        real_main = sys.modules["__main__"]
        with tempfile.TemporaryDirectory() as folder:
            try:
                for text, expected in (("print('top-level work')\n", False),
                                       ("def main():\n    pass\n\nif __name__ == \"__main__\":\n    main()\n", True)):
                    script = os.path.join(folder, "main_script.py")
                    with open(script, "w", encoding="utf-8") as fh:
                        fh.write(text)
                    fake_main = types.ModuleType("__main__")
                    fake_main.__file__ = script
                    fake_main.__spec__ = None
                    sys.modules["__main__"] = fake_main
                    self.assertEqual(par.main_module_is_spawn_safe(), expected)
                fake_main = types.ModuleType("__main__")
                fake_main.__file__ = script
                fake_main.__spec__ = types.SimpleNamespace(name="unittest.__main__")
                sys.modules["__main__"] = fake_main
                self.assertTrue(par.main_module_is_spawn_safe())   # python -m package
            finally:
                sys.modules["__main__"] = real_main


# %% Sorting
class SortKeyTests(unittest.TestCase):
    """sort_keys() orders exactly like np.argsort(kind="stable")."""

    def check(self, values):
        """Stable sort of the keys == stable sort of the values."""
        keys = gpu.sort_keys(np, values)
        self.assertEqual(keys.dtype, np.uint64)
        np.testing.assert_array_equal(np.argsort(keys, kind="stable"),
                                      np.argsort(values, kind="stable"))

    def test_floats_with_nan_inf_zeros_and_ties(self):
        """NaN last, -0.0 == +0.0, +-inf, many ties; float64 and float32."""
        values = gpu._test_values(20011)
        self.check(values)
        self.check(values.astype(np.float32))
        self.check(np.array([np.nan, -0.0, 0.0, -np.inf, np.inf, 5e-324, -5e-324, np.nan, 1.0, 1.0]))

    def test_integers_and_booleans(self):
        """Signed (extremes too), unsigned and boolean arrays."""
        rng = np.random.default_rng(5)
        signed = rng.integers(-2 ** 62, 2 ** 62, 5000)
        signed[:3] = [np.iinfo(np.int64).min, np.iinfo(np.int64).max, 0]
        self.check(signed)
        self.check(rng.integers(-100, 100, 5000).astype(np.int16))
        self.check(rng.integers(0, 2 ** 63, 5000, dtype=np.uint64))
        self.check(rng.integers(0, 2, 5000).astype(bool))

    def test_argsort_without_gpu_is_numpy(self):
        """No GPU (AUTOPLOT_GPU=off): argsort() is np.argsort(kind='stable')."""
        values = gpu._test_values(1000)
        np.testing.assert_array_equal(gpu.argsort(values, use_gpu=False),
                                      np.argsort(values, kind="stable"))


# %% Linear-domain averages of dB values
class PowerMeanTests(unittest.TestCase):
    """power_mean_db(): the mean of dB values is taken in the linear domain."""

    def test_hand_values(self):
        """0 dB and -10 dB average to 10 log10(0.55) = -2.596 dB, not -5 dB."""
        got = gpu.power_mean_db([[0.0], [-10.0]], use_gpu=False)
        self.assertAlmostEqual(float(got[0]), 10 * np.log10(0.55), places=12)
        self.assertNotAlmostEqual(float(got[0]), -5.0, places=3)
        # equal levels stay put; 3 dB steps
        self.assertAlmostEqual(float(gpu.power_mean_db([[-30.0], [-30.0]])[0]), -30.0, places=12)
        self.assertAlmostEqual(float(gpu.power_mean_db([[3.0], [0.0]])[0]),
                               10 * np.log10((10 ** 0.3 + 1) / 2), places=12)

    def test_amplitude_factor_20(self):
        """factor=20 averages amplitudes (|S|, volts): 0 dB and -6.0206 dB -> 0.75."""
        got = gpu.power_mean_db([[0.0], [20 * np.log10(0.5)]], factor=20.0)
        self.assertAlmostEqual(float(got[0]), 20 * np.log10(0.75), places=12)

    def test_nan_inf_and_empty_columns(self):
        """NaN left out; -inf = zero power; +inf -> +inf; no values -> NaN."""
        stack = np.array([[0.0, np.nan, -np.inf, np.inf, np.nan],
                          [np.nan, np.nan, -np.inf, 0.0, -10.0],
                          [-10.0, np.nan, 0.0, 0.0, np.nan]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")                 # no RuntimeWarning escapes
            got = gpu.power_mean_db(stack)
        self.assertAlmostEqual(got[0], 10 * np.log10(0.55), places=12)   # [0, -10]
        self.assertTrue(np.isnan(got[1]))
        self.assertAlmostEqual(got[2], 10 * np.log10(1 / 3.0), places=12)  # [-inf, -inf, 0]
        self.assertEqual(got[3], np.inf)
        self.assertAlmostEqual(got[4], -10.0, places=12)
        self.assertEqual(float(gpu.power_mean_db([[-np.inf], [-np.inf]])[0]), -np.inf)

    def test_axis_and_large_levels(self):
        """Any axis; +-400 dB levels do not overflow (max-referenced)."""
        rng = np.random.default_rng(3)
        stack = rng.normal(-60, 10, (7, 5, 9))
        np.testing.assert_allclose(gpu.power_mean_db(stack, axis=1),
                                   10 * np.log10(np.mean(10 ** (stack / 10), axis=1)), rtol=0, atol=1e-10)
        self.assertAlmostEqual(float(gpu.power_mean_db([[400.0], [400.0]])[0]), 400.0, places=10)
        self.assertAlmostEqual(float(gpu.power_mean_db([[-400.0], [-400.0]])[0]), -400.0, places=10)

    def test_same_as_writer_for_finite_data(self):
        """Bit-identical to vszh5_writer.linear_power_mean_dbm (used before 1.1.3 by R&S)."""
        try:
            from vszh5_writer import linear_power_mean_dbm
        except Exception as exc:                            # h5py missing in this environment
            self.skipTest("vszh5_writer not importable: %s" % exc)
        rng = np.random.default_rng(9)
        for rows in (1, 3, 128, 129, 300):
            stack = rng.normal(-70, 15, (rows, 801))
            np.testing.assert_array_equal(gpu.power_mean_db(stack, use_gpu=False),
                                          linear_power_mean_dbm(stack, axis=0))


# %% GPU back ends
class CupyBackendTests(GpuState):
    """The CuPy (CUDA / ROCm) code path with the NumPy stand-in."""

    def test_cuda_and_rocm_selection(self):
        """is_hip picks ROCm (AMD); AUTOPLOT_GPU=rocm refuses a CUDA build and so on."""
        info = self.select("auto", fake_cupy(is_hip=False))
        self.assertEqual((info.backend, info.vendor), ("cuda", "NVIDIA"))
        self.assertIn("Test GPU", gpu.describe())
        info = self.select("auto", fake_cupy(is_hip=True))
        self.assertEqual((info.backend, info.vendor), ("rocm", "AMD"))
        info = self.select("rocm", fake_cupy(is_hip=False))
        self.assertNotIn(info.backend, ("cuda", "rocm"))
        self.assertTrue(any("not ROCM" in n for n in info.notes), info.notes)
        info = self.select("off", fake_cupy())
        self.assertEqual(info.backend, "")
        self.assertIn("AUTOPLOT_GPU=off", gpu.describe())

    def test_operations_match_numpy(self):
        """argsort, power_mean_db and ifft_centered through the CuPy path equal NumPy."""
        self.select("cuda", fake_cupy())
        gpu.enable(True)
        values = gpu._test_values(5000)
        np.testing.assert_array_equal(gpu.argsort(values, min_elements=0),
                                      np.argsort(values, kind="stable"))
        stack = gpu._test_values(40 * 33).reshape(40, 33)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # NumPy stand-in warns like NumPy
            got = gpu.power_mean_db(stack, min_elements=0)
        np.testing.assert_array_equal(got, gpu.power_mean_db(stack, use_gpu=False))
        x = np.random.default_rng(2).normal(size=(101, 2, 2)) + 0j
        np.testing.assert_array_equal(gpu.ifft_centered(x, min_elements=0),
                                      np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=0), axis=0), axes=0))
        used = gpu.usage()
        self.assertEqual([used[op]["gpu"] for op in ("argsort", "power_mean_db", "ifft_centered")], [1, 1, 1])

    def test_threshold_and_switch(self):
        """Small arrays and a switched-off GPU stay on the CPU."""
        self.select("cuda", fake_cupy())
        values = gpu._test_values(1000)
        gpu.argsort(values)                                 # not enabled
        gpu.enable(True)
        gpu.argsort(values)                                 # below 200 000
        gpu.set_threshold("argsort", 500)
        gpu.argsort(values)                                 # above 500: GPU
        self.assertEqual(gpu.usage()["argsort"], {"gpu": 1, "cpu": 2})
        with self.assertRaises(KeyError):
            gpu.set_threshold("no_such_operation", 1)

    def test_failed_self_test_falls_back(self):
        """A back end that gives a wrong order is never used; results stay right."""
        self.select("cuda", fake_cupy(wrong_argsort=True))
        values = gpu._test_values(3000)
        np.testing.assert_array_equal(gpu.argsort(values, use_gpu=True, min_elements=0),
                                      np.argsort(values, kind="stable"))
        self.assertIn("argsort", gpu._FAILED)
        self.assertIn("NumPy for argsort", gpu.describe())
        self.assertEqual(gpu.usage()["argsort"], {"gpu": 0, "cpu": 1})

    def test_common_wrappers(self):
        """_autoplot_common.gpu_argsort keeps its 0.0.10 API (force=True/False/None)."""
        try:
            import _autoplot_common as common
        except Exception as exc:                            # Qt / astropy missing here
            self.skipTest("_autoplot_common not importable: %s" % exc)
        self.select("cuda", fake_cupy())
        values = gpu._test_values(2000)
        ref = np.argsort(values, kind="stable")
        np.testing.assert_array_equal(common.gpu_argsort(values, force=True), ref)
        np.testing.assert_array_equal(common.gpu_argsort(values, force=False), ref)
        np.testing.assert_array_equal(common.gpu_argsort(values), ref)
        self.assertEqual(gpu.usage()["argsort"], {"gpu": 1, "cpu": 2})
        self.assertTrue(common.is_gpu_available())
        self.assertIn("200000", common.gpu_tooltip().replace(" ", ""))


@unittest.skipUnless(opencl_device_found(), "PyOpenCL with an OpenCL device is not installed")
class OpenCLBackendTests(GpuState):
    """The OpenCL code path on a real OpenCL device (PoCL on the CPU is enough)."""

    def setUp(self):
        """Select OpenCL (a CPU device is accepted with AUTOPLOT_GPU=opencl)."""
        super().setUp()
        sys.modules["cupy"] = None                          # make sure CuPy is not picked
        info = self.select("opencl")
        if info.backend != "opencl":
            self.skipTest("no OpenCL device: %s" % info.notes)
        gpu.enable(True)

    def test_argsort_identical(self):
        """Radix sort of the 64-bit keys gives NumPy's stable order exactly."""
        for values in (gpu._test_values(70001), gpu._test_values(999).astype(np.float32),
                       np.random.default_rng(4).integers(-2 ** 62, 2 ** 62, 30000)):
            np.testing.assert_array_equal(gpu.argsort(values, min_elements=0),
                                          np.argsort(values, kind="stable"))
        self.assertEqual(gpu.usage()["argsort"]["gpu"], 3)

    def test_power_mean_matches(self):
        """The OpenCL kernel equals the NumPy mean (1e-9 dB in double precision)."""
        stack = np.round(np.random.default_rng(8).normal(-60.0, 20.0, (300, 801)), 2)
        stack[::7, :] = np.nan                              # NaN cells are left out
        stack[:, 7] = np.nan                                # a column without values
        stack[::5, 9] = -np.inf                             # zero power
        stack[3, 11] = np.inf
        got = gpu.power_mean_db(stack, min_elements=0)
        ref = gpu.power_mean_db(stack, use_gpu=False)
        self.assertEqual(gpu.usage()["power_mean_db"]["gpu"], 1)
        np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))
        finite = np.isfinite(ref)
        tol = 1e-9 if gpu.detect().fp64 else 1e-3
        self.assertLessEqual(float(np.max(np.abs(got[finite] - ref[finite]))), tol)
        np.testing.assert_array_equal(got[np.isinf(ref)], ref[np.isinf(ref)])


if __name__ == "__main__":
    unittest.main()
