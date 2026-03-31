"""
RunLogger — lightweight run-time logger for AEzip compress / decompress.

Records per-step wall-clock time and output file sizes, then writes a
human-readable summary to stdout and a structured JSON log to disk.

Usage
-----
    logger = RunLogger("compressed.log.json", run_type="compress",
                       args=vars(args))

    with logger.step("Featurize"):
        feat = MDFeaturizer(...)

    logger.log_file(args.output_file, label="compressed")
    logger.save()
"""

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime


class RunLogger:
    def __init__(self, log_path: str, run_type: str, args: dict = None):
        """
        Parameters
        ----------
        log_path  : path where the JSON log will be written
        run_type  : "compress" or "decompress"
        args      : dict of CLI arguments (vars(parsed_args))
        """
        self.log_path   = log_path
        self.run_type   = run_type
        self.args       = {k: str(v) for k, v in (args or {}).items()}
        self.steps: list[dict] = []
        self.files: list[dict] = []
        self._t0        = time.perf_counter()
        self.timestamp  = datetime.now().isoformat(timespec="seconds")

    # ------------------------------------------------------------------
    @contextmanager
    def step(self, name: str):
        """Context manager that times a named step and prints progress."""
        print(f"\n[{name}]")
        t = time.perf_counter()
        yield
        dt = time.perf_counter() - t
        self.steps.append({"name": name, "duration_s": round(dt, 3)})
        print(f"  → done in {_fmt_duration(dt)}")

    # ------------------------------------------------------------------
    def log_file(self, path: str, label: str = None):
        """Record the size of an output file (silently skips if missing)."""
        if path and os.path.exists(path):
            size_bytes = os.path.getsize(path)
            self.files.append({
                "label":    label or os.path.basename(path),
                "path":     str(path),
                "size_mb":  round(size_bytes / 1_048_576, 3),
                "size_kb":  round(size_bytes / 1_024, 1),
            })

    # ------------------------------------------------------------------
    def save(self):
        """Print a summary table and write the JSON log file."""
        total = time.perf_counter() - self._t0

        # ---- stdout summary ----
        w = max((len(s["name"]) for s in self.steps), default=20) + 2
        print("\n" + "─" * (w + 16))
        print(f"  {'Step':<{w}} {'Duration':>10}")
        print("─" * (w + 16))
        for s in self.steps:
            print(f"  {s['name']:<{w}} {_fmt_duration(s['duration_s']):>10}")
        print("─" * (w + 16))
        print(f"  {'TOTAL':<{w}} {_fmt_duration(total):>10}")

        if self.files:
            print()
            fw = max((len(f["label"]) for f in self.files), default=20) + 2
            print(f"  {'File':<{fw}} {'Size':>10}")
            print("─" * (fw + 16))
            for f in self.files:
                size_str = f"{f['size_mb']} MB" if f["size_mb"] >= 1 else f"{f['size_kb']} KB"
                print(f"  {f['label']:<{fw}} {size_str:>10}   {f['path']}")
        print("─" * (w + 16))

        # ---- JSON log ----
        data = {
            "run_type":         self.run_type,
            "timestamp":        self.timestamp,
            "args":             self.args,
            "steps":            self.steps,
            "output_files":     self.files,
            "total_duration_s": round(total, 3),
        }
        with open(self.log_path, "w") as fh:
            json.dump(data, fh, indent=2)
        print(f"\n  Log written to: {self.log_path}\n")


# ---------------------------------------------------------------------------
def _fmt_duration(seconds: float) -> str:
    """Return a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"
