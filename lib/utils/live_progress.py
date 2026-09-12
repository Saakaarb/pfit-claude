"""Append evaluated best points for the read-only browser view (no extra solves)."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np


def emit_progress(optimizer, stage, iteration, loss, position, kind="iteration"):
    """Positions enter scaled and are saved in physical units with parameter names.

    Called outside JIT, only at existing evaluation boundaries. Monitoring must
    never interrupt optimization, including when the progress file is unwritable.
    """
    try:
        reader = optimizer.input_reader
        physical = optimizer.unscale_design_point(np.asarray(position))
        def finite(value):
            value = float(value)
            return value if math.isfinite(value) else None
        event = {
            "time": datetime.now(timezone.utc).isoformat(),
            "stage": stage, "iteration": int(iteration), "kind": kind,
            "loss": finite(loss) if loss is not None else None,
            "parameters": {name: finite(value) for name, value in
                           zip(reader.trainable_parameter_names, physical, strict=True)},
        }
        with (Path(reader.output_dir) / "live_progress.jsonl").open("a") as handle:
            handle.write(json.dumps(event, allow_nan=False) + "\n")
    except Exception as exc:
        if not getattr(optimizer, "_progress_warning", False):
            print(f"[live view] progress unavailable: {exc}; optimization continues")
            optimizer._progress_warning = True
