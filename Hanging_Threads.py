# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------
Hanging_Threads.py

Debug helper: print the stack of any thread that stays at the same place
for ``seconds_frozen`` seconds (a hang or deadlock).

    from Hanging_Threads import start_monitoring
    monitor = start_monitoring(seconds_frozen=20)
    ...
    monitor.stop()

Note: a GUI main thread waiting in its event loop (app.exec()) does not
move either, so it is reported while the GUI is idle.  Pass
``ignore_idle_main=True`` (the default) to skip a main thread whose top
frame is the Qt event loop.

Created on 2025 (date not recorded in the original template)

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216
Licence: MIT (original code (C) 2014 Nicco Kunzmann, see below)

# %%% References
[1] N. Kunzmann, "hanging_threads.py," GitHub Gist 6038331, 2014.
    https://gist.github.com/niccokunzmann/6038331
[2] Python Software Foundation, "sys._current_frames," Python 3 docs.
    https://docs.python.org/3/library/sys.html#sys._current_frames

# %%% Revisions
--------
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.0.3: Stoppable monitor, no auto-start, one report per hang.
# Date: 2026-11-05
#              * Importing the module no longer starts a monitor (the R&S
#                plotter then started a second one).
#              * start_monitoring() returns a HangMonitor with stop(); the
#                caller's monitor.stop() used to raise AttributeError on the
#                plain Thread it got.
#              * The report named the wrong thread (a variable left over
#                from the previous loop); a frozen stack is now reported
#                once per hang, not again every seconds_frozen.
#              * Idle Qt main threads (event loop) are skipped by default;
#                checks run 10 times per second (was 100) to keep the cost
#                down, configurable.
#              * History is newest-first, as in the rest of the repository.
# %%%% 0.0.2: Template revision (no description).
# %%%% 0.0.1: Copy of the MIT-licensed gist with the author's header.
=============================================================================
"""

## The MIT License (MIT)
## ---------------------
##
## Copyright (C) 2014 Nicco Kunzmann
##
## https://gist.github.com/niccokunzmann/6038331
##
## Permission is hereby granted, free of charge, to any person obtaining
## a copy of this software and associated documentation files (the "Software"),
## to deal in the Software without restriction, including without limitation
## the rights to use, copy, modify, merge, publish, distribute, sublicense,
## and/or sell copies of the Software, and to permit persons to whom the
## Software is furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included
## in all copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
## OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
## FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
## IN THE SOFTWARE.

# %% Import Modules
import linecache
import sys
import threading
import time

__version__ = "0.0.3"

SECONDS_FROZEN = 20       # seconds without movement before a report
TESTS_PER_SECOND = 10     # checks per second
_EVENT_LOOP_NAMES = ("exec", "exec_", "processEvents")


# %% Stack text
def frame2string(frame):
    """One stack line, as the traceback module writes it."""
    lineno = frame.f_lineno
    code = frame.f_code
    text = '  File "{}", line {}, in {}'.format(code.co_filename, lineno, code.co_name)
    line = linecache.getline(code.co_filename, lineno, frame.f_globals).lstrip()
    return text + "\n\t" + line


def thread2list(frame):
    """Whole stack of a frame, outermost first."""
    lines = []
    while frame:
        lines.insert(0, frame2string(frame))
        frame = frame.f_back
    return lines


def print_frame_list(frame_list, thread_id, stream=None):
    """Write one report to ``stream`` (stderr)."""
    stream = stream or sys.stderr
    stream.write("-" * 20 + "Thread {}".format(thread_id).center(20) + "-" * 20 + "\n" +
                 "".join(frame_list))
    stream.flush()


# %% Monitor
class HangMonitor:
    """Background thread that reports threads frozen for seconds_frozen."""

    def __init__(self, seconds_frozen=SECONDS_FROZEN, tests_per_second=TESTS_PER_SECOND,
                 ignore_idle_main=True, stream=None):
        self.seconds_frozen = float(seconds_frozen)
        self.interval = 1.0 / max(0.1, float(tests_per_second))
        self.ignore_idle_main = ignore_idle_main
        self.stream = stream
        self.reports = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="HangMonitor", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self, timeout=2.0):
        """Stop the monitor thread (returns when it has ended or timeout)."""
        self._stop.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout)

    @property
    def running(self):
        return self._thread.is_alive()

    def _idle_event_loop(self, thread_id, frame):
        if not self.ignore_idle_main or thread_id != threading.main_thread().ident:
            return False
        return frame is not None and frame.f_code.co_name in _EVENT_LOOP_NAMES

    def _run(self):
        own = threading.get_ident()
        seen = {}                                    # thread id -> (stack, since, reported)
        while not self._stop.wait(self.interval):
            now = time.time()
            current = {}
            for thread_id, frame in sys._current_frames().items():
                if thread_id == own or self._idle_event_loop(thread_id, frame):
                    continue
                stack = thread2list(frame)
                old = seen.get(thread_id)
                if old is None or old[0] != stack:
                    current[thread_id] = (stack, now, False)
                    continue
                reported = old[2]
                if not reported and now - old[1] >= self.seconds_frozen:
                    print_frame_list(stack, thread_id, self.stream)
                    self.reports += 1
                    reported = True
                current[thread_id] = (stack, old[1], reported)
            seen = current


def start_monitoring(seconds_frozen=SECONDS_FROZEN, tests_per_second=TESTS_PER_SECOND,
                     ignore_idle_main=True, stream=None):
    """Start and return a HangMonitor (call .stop() when done)."""
    return HangMonitor(seconds_frozen, tests_per_second, ignore_idle_main, stream).start()


if __name__ == "__main__":
    monitor = start_monitoring(seconds_frozen=1)
    time.sleep(3)  # TEST: reported once after about 1 s
    monitor.stop()
