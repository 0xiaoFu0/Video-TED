"""Check the upload package launchers without any GPU or server access."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mode,model,name,run", [
    ("base", "/path/to/workspace/models/Qwen/Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-7B-Instruct", "qwen25_base_8gpu_v6"),
    ("trained", "/gemini-3/space/zwj/AR-RLSD/eval_models/ar_v3_025233_step292", "AR-RLSD-step292", "ar_rlsd_step292_8gpu_v6"),
])
@pytest.mark.parametrize("launcher", ["run_v5.sh", "run_v6.sh"])
def test_short_launch_commands_use_exact_models_and_separate_v6_results(tmp_path, mode, model, name, run, launcher):
    capture = tmp_path / "capture.py"
    capture.write_text("#!" + sys.executable + "\nimport sys, json, os\n"
                       "print(json.dumps({'argv': sys.argv[1:], 'mp': os.environ['VLLM_WORKER_MULTIPROC_METHOD']}))\n")
    capture.chmod(0o755)
    env = dict(os.environ, EVAL_PYTHON=str(capture), VLLM_WORKER_MULTIPROC_METHOD="fork")
    env.pop("EVAL_V6_RESULTS_ROOT", None)
    output = subprocess.check_output(["bash", str(ROOT / launcher), mode, "--resume"], env=env, text=True)
    value = json.loads(output)
    args = value["argv"]
    assert args[0].endswith("parallel_eval.py")
    for flag, expected in [("--model", model), ("--model-name", name), ("--run-name", run),
                           ("--model-role", "base" if mode == "base" else "ar-trained"),
                           ("--gpus", "0,1,2,3,4,5,6,7"),
                           ("--output-root", "/gemini-3/space/zwj/VideoSTAR-Eval-results-v6/runs"),
                           ("--table-dir", "/gemini-3/space/zwj/VideoSTAR-Eval-results-v6")]:
        assert args[args.index(flag) + 1] == expected
    assert "--allow-train-overlap" in args
    assert args[-1] == "--resume" and value["mp"] == "spawn"


def test_invalid_mode_does_not_launch_python():
    result = subprocess.run(["bash", str(ROOT / "run_v6.sh"), "unknown"], capture_output=True, text=True)
    assert result.returncode == 2 and "Usage:" in result.stderr
