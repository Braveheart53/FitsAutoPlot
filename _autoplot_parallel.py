# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------
_autoplot_parallel.py

Shared CPU parallelism for the FitsAutoPlot suite: a process pool that is
used only when it pays off.

Reading and processing files in a 'spawn' process pool scales with the CPU
cores, but starting the pool costs time: about 0.2-0.3 s on Linux and up to
about 1 s on Windows, where every worker starts a new Python and imports
numpy, astropy or scikit-rf again.  For small batches the pool is slower
than a plain loop.  Measured on 2026-11-09 (2 cores): 220 Touchstone files
took 0.7 s in a loop and 2.1 s in a pool; 25 ATR files 0.01 s and 0.19 s.

adaptive_map() runs the first items in this process, times them, estimates
the rest, and starts a pool only when the estimate says it will finish
sooner - so a large batch gets every core and a small one never waits for
a pool.  The decision is logged through ``log`` when one is given.

Created on 2026-11-09

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216

# %%% Revisions
--------
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.0.1: First version - adaptive process pool.
Date: 2026-11-09
#              Used by ATR, R&S, Touchstone, FITS and Franks AutoPlot.
# %%%%% Function Descriptions
        default_workers: Number of worker processes to offer (logical CPUs, at least 1).
        pool_start_seconds: Estimated cost (s) of starting a spawn process pool on this platform.
        should_use_pool: True when a pool of ``workers`` should finish ``remaining`` items sooner.
        module_import_seconds: Pool start-up estimate for a program whose import began at ``t0``.
        main_module_is_spawn_safe: True when 'spawn' workers can import the main module without side effects.
        adaptive_map: ``[func(item) for item in items]``, in a 'spawn' pool only when it pays off.
# %%%%% Variable Descriptions
    Define all utilized variables
        POOL_START_S_WINDOWS: pool start-up estimate on Windows (s)
        POOL_START_S_OTHER: pool start-up estimate on Linux / macOS (s)
        MIN_SAVING_S: smallest saving that justifies a pool (s)
        PROBE_ITEMS: items timed in this process before deciding
# %%%%% More Info
    Only for the standalone programs.  The Veusz plugins must not start
    process pools: with 'spawn' every worker would re-run Veusz's own main
    module.  The functions given to adaptive_map() must be module-level
    (picklable) and so must their arguments and results.
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
=============================================================================
"""
# %% Imports
from __future__ import annotations

__version__ = "0.0.1"          # Semantic Versioning: see the Revisions entries above

import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, List, Optional, Sequence

# %% Settings
POOL_START_S_WINDOWS = 1.0
POOL_START_S_OTHER = 0.3
MIN_SAVING_S = 0.5
PROBE_ITEMS = 2


# %% Decisions
def default_workers() -> int:
    """Number of worker processes to offer (logical CPUs, at least 1)."""
    return max(1, int(os.cpu_count() or 1))


def pool_start_seconds() -> float:
    """Estimated cost (s) of starting a spawn process pool on this platform."""
    return POOL_START_S_WINDOWS if sys.platform.startswith("win") else POOL_START_S_OTHER


def should_use_pool(seconds_per_item: float, remaining: int, workers: int,
                    start_s: Optional[float] = None) -> bool:
    """True when a pool of ``workers`` should finish ``remaining`` items sooner.

    Sequential estimate:  t_seq = s * N
    Pool estimate:        t_pool = t_start + 1.1 * s * N / W
    (10 % for pickling the arguments and results).  The pool is used when
    t_seq - t_pool >= MIN_SAVING_S.  ``start_s`` is t_start; by default
    pool_start_seconds().  Each 'spawn' worker imports the program's main
    module again, so callers add the time their own import took
    (module_import_seconds).
    """
    workers = min(int(workers), int(remaining))
    if workers < 2 or remaining < 2:
        return False
    t_start = pool_start_seconds() if start_s is None else float(start_s)
    t_seq = seconds_per_item * remaining
    t_pool = t_start + 1.1 * t_seq / workers
    return t_seq - t_pool >= MIN_SAVING_S


def module_import_seconds(t0: float) -> float:
    """Pool start-up estimate for a program whose import began at ``t0``.

    ``t0`` is time.perf_counter() taken at the top of the program; the
    result is pool_start_seconds() plus the program's own import time,
    which every 'spawn' worker pays again.
    """
    return pool_start_seconds() + max(0.0, time.perf_counter() - float(t0))


def main_module_is_spawn_safe() -> bool:
    """True when 'spawn' workers can import the main module without side effects.

    Every 'spawn' worker imports the parent's main script again (as
    __mp_main__).  A script with top-level work and no
    ``if __name__ == "__main__":`` guard would run that work in every worker
    (a test or helper script that opens a window, say).  The suite's
    programs all have the guard; for any other main script without one the
    pool is not used.  Interactive sessions, frozen programs and
    ``python -m package`` mains (such as ``python -m unittest``, which spawn
    does not re-run) are safe.
    """
    main = sys.modules.get("__main__")
    spec_name = getattr(getattr(main, "__spec__", None), "name", None) or ""
    if spec_name == "__main__" or spec_name.endswith(".__main__"):
        return True               # 'python -m package': spawn does not re-run it
    path = getattr(main, "__file__", None)
    if not path or not os.path.isfile(path):
        return True
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return False
    return "__name__ == \"__main__\"" in text or "__name__ == '__main__'" in text


# %% Mapping
def adaptive_map(func: Callable[[Any], Any], items: Sequence[Any], workers: int,
                 progress: Optional[Callable[[int, int], None]] = None,
                 log: Optional[Callable[[str], None]] = None,
                 probe: int = PROBE_ITEMS, force: Optional[bool] = None,
                 on_result: Optional[Callable[[int, Any], None]] = None,
                 start_s: Optional[float] = None) -> List[Any]:
    """``[func(item) for item in items]``, in a 'spawn' pool only when it pays off.

    Parameters
    ----------
    func : callable
        Module-level function (picklable); it should catch its own errors
        and return them, as the suite's reader workers do.
    items : sequence
        Arguments, one call each.
    workers : int
        Largest number of worker processes (1 = never a pool).
    progress : callable, optional
        ``progress(done, total)`` after every item.
    log : callable, optional
        Receives one line saying how the batch was run.
    probe : int
        Items timed in this process before deciding.
    force : bool, optional
        True: pool for everything after the probe (if workers > 1);
        False: never a pool; None (default): decide from the timing.
    on_result : callable, optional
        ``on_result(index, result)`` as each item finishes (any order),
        before ``progress``.
    start_s : float, optional
        Pool start-up estimate (s); see should_use_pool().

    Returns
    -------
    list
        Results in the order of ``items``.
    """
    items = list(items)
    total = len(items)
    results: List[Any] = [None] * total
    done = 0
    start = time.perf_counter()
    n_probe = min(total, max(1, int(probe)))
    for index in range(n_probe):
        results[index] = func(items[index])
        done += 1
        if on_result:
            on_result(index, results[index])
        if progress:
            progress(done, total)
    remaining = total - n_probe
    if remaining == 0:
        return results
    per_item = (time.perf_counter() - start) / n_probe
    if force is None:
        use_pool = should_use_pool(per_item, remaining, workers, start_s)
    else:
        use_pool = bool(force) and workers > 1 and remaining > 1
    if use_pool and not main_module_is_spawn_safe():
        use_pool = False
        if log:
            log("main script has no __main__ guard: no process pool")
    if log:
        log("%d items, %.3f s each: %s" % (
            total, per_item,
            "process pool of %d" % min(workers, remaining) if use_pool else "one process"))
    if not use_pool:
        for index in range(n_probe, total):
            results[index] = func(items[index])
            done += 1
            if on_result:
                on_result(index, results[index])
            if progress:
                progress(done, total)
        return results
    context = multiprocessing.get_context("spawn")      # fork can deadlock with Qt threads
    with ProcessPoolExecutor(max_workers=min(int(workers), remaining), mp_context=context) as pool:
        futures = {pool.submit(func, items[index]): index for index in range(n_probe, total)}
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
            done += 1
            if on_result:
                on_result(index, results[index])
            if progress:
                progress(done, total)
    return results
