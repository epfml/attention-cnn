import time
import json
from contextlib import contextmanager
from io import StringIO

import numpy as np
import torch

NS = 1.0 / 1_000_000_000  # 1[ns] in [s]


class Timer:
    """
    Timer for PyTorch code
    Comes in the form of a contextmanager:

    Example:
    >>> timer = Timer()
    ... for i in range(10):
    ...     with timer("expensive operation"):
    ...         x = torch.randn(100)
    ... print(timer.summary())
    """

    def __init__(self, verbosity_level=1, log_fn=None, skip_first=True):
        self.verbosity_level = verbosity_level
        self.log_fn = log_fn if log_fn is not None else self._default_log_fn
        self.skip_first = skip_first

        self.reset()

    def reset(self):
        """Reset the timer"""
        self.totals = {}  # Total time per label
        self.first_time = {}  # First occurrence of a label (start time)
        self.last_time = {}  # Last occurence of a label (end time)
        self.call_counts = {}  # Number of times a label occurred

    @contextmanager
    def __call__(self, label, epoch=-1.0, verbosity=1):
        # Don't measure this if the verbosity level is too high
        if verbosity > self.verbosity_level:
            yield
            return

        # Measure the time
        self._cuda_sync()
        start = time.time_ns() * NS
        yield
        self._cuda_sync()
        end = time.time_ns() * NS

        # Update first and last occurrence of this label
        if not label in self.first_time:
            self.first_time[label] = start
        self.last_time[label] = end

        # Update the totals and call counts
        if not label in self.totals and self.skip_first:
            self.totals[label] = 0.0
            del self.first_time[label]
            self.call_counts[label] = 0
        elif not label in self.totals and not self.skip_first:
            self.totals[label] = end - start
            self.call_counts[label] = 1
        else:
            self.totals[label] += end - start
            self.call_counts[label] += 1

        if self.call_counts[label] > 0:
            # We will reduce the probability of logging a timing linearly with the number of times
            # we have seen it.
            # It will always be recorded in the totals, though
            if np.random.rand() < 1 / self.call_counts[label]:
                self.log_fn(
                    "timer", {"epoch": float(epoch), "value": end - start}, {"event": label}
                )

    def summary(self):
        """
        Return a summary in string-form of all the timings recorded so far
        """
        with StringIO() as buffer:
            print("--- Timer summary -----------------------------------------------", file=buffer)
            print("  Event                          |  Count | Average time |  Frac.", file=buffer)
            for event_label in sorted(self.totals):
                total = self.totals[event_label]
                count = self.call_counts[event_label]
                if count == 0:
                    continue
                avg_duration = total / count
                total_runtime = self.last_time[event_label] - self.first_time[event_label]
                runtime_percentage = 100 * total / total_runtime
                print(
                    f"- {event_label:30s} | {count:6d} | {avg_duration:11.5f}s | {runtime_percentage:5.1f}%",
                    file=buffer,
                )
            print("-----------------------------------------------------------------", file=buffer)
            return buffer.getvalue()

    def save_summary(self, json_file_path):
        data = {}
        for event_label in sorted(self.totals):
            total = self.totals[event_label]
            count = self.call_counts[event_label]
            if count == 0:
                continue
            avg_duration = total / count
            data[event_label] = {
                "label": event_label,
                "average_duration": avg_duration,
                "n_events": count,
                "total_time": total,
            }

        with open(json_file_path, "w") as fp:
            json.dump(data, fp)

    def _cuda_sync(self):
        """Finish all asynchronous GPU computations to get correct timings"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _default_log_fn(self, _, values, tags):
        label = tags["label"]
        epoch = values["epoch"]
        duration = values["value"]
        print(f"Timer: {label:30s} @ {epoch:4.1f} - {duration:8.5f}s")
