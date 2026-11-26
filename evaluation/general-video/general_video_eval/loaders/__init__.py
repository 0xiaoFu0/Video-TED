from __future__ import annotations

import os


def load_dataset(benchmark, data_dir, annotation_path=None):
    if benchmark == "mvbench":
        from .mvbench import load_mvbench
        return load_mvbench(data_dir, annotation_path=annotation_path)
    if benchmark == "videomme":
        from .videomme import load_videomme
        return load_videomme(data_dir, annotation_path=annotation_path)
    if benchmark == "tempcompass":
        from .tempcompass import load_tempcompass
        return load_tempcompass(data_dir, annotation_path=annotation_path)
    if benchmark == "seedbench":
        from .seedbench import load_seedbench
        return load_seedbench(data_dir, annotation_path=annotation_path)
    raise ValueError(f"Unsupported benchmark: {benchmark}")
