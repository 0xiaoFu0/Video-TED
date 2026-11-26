"""Exercise cancellation, orphan cleanup, and experiment exclusion without CUDA."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest

from general_video_eval.run import experiment_lock, stop_workers


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _environment():
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    return environment


def _running(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # An orphan may briefly remain a zombie until init/launchd reaps it. Such a
    # process cannot retain CUDA resources or execute more work.
    status = Path(f"/proc/{pid}/stat")
    if sys.platform.startswith("linux"):
        try:
            # The command name in parentheses may itself contain spaces.
            state = status.read_text().rsplit(")", 1)[-1].strip().split()[0]
        except FileNotFoundError:
            return False
        return state != "Z"
    return True


def _wait_stopped(pid, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _running(pid):
            return True
        time.sleep(0.05)
    return not _running(pid)


def _kill_group(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class _ProcessAdapter:
    """Expose the multiprocessing lifecycle interface for a real OS subprocess."""
    def __init__(self, process):
        self.process = process
        self.pid = process.pid

    def is_alive(self):
        return self.process.poll() is None

    def join(self, timeout=None):
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass

    def terminate(self):
        self.process.terminate()

    def kill(self):
        self.process.kill()


@unittest.skipUnless(os.name == "posix" and hasattr(os, "killpg"), "POSIX process groups required")
class ProcessCleanupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def test_sigterm_reports_interruption_and_stops_sleeping_worker(self):
        # A real spawned worker sleeps in place of model loading/inference.
        # This exercises the parent's installed SIGTERM handler and finalizer,
        # not a direct call to the handler or a mocked stop_workers function.
        dataset = self.root / "data"
        (dataset / "data").mkdir(parents=True)
        (dataset / "data" / "clip.mp4").touch()
        (dataset / "test.json").write_text(json.dumps([{
            "question_id": "one", "videoID": "clip", "question": "Which action?",
            "options": ["A. running", "B. sitting"], "answer": "A",
            "duration": "short", "task_type": "Action Recognition",
        }]), encoding="utf-8")
        helper = self.root / "signal_helper.py"
        helper.write_text(textwrap.dedent('''
            import os
            from pathlib import Path
            import sys
            import time
            from general_video_eval import run

            def sleeping_worker(worker_id, devices, config, shard, run_dir, events):
                os.setsid()
                (Path(run_dir) / "worker.pid").write_text(str(os.getpid()))
                events.put({"event": "started", "worker": worker_id})
                time.sleep(60)

            if __name__ == "__main__":
                root = Path(sys.argv[1])
                run.worker_main = sleeping_worker
                raise SystemExit(run.main([
                    "--backend", "mock", "--benchmarks", "videomme",
                    "--videomme-data-dir", str(root / "data"),
                    "--output-dir", str(root / "outputs"), "--exp-name", "cancel",
                    "--allow-nonstandard-counts", "--num-gpus", "1",
                    "--worker-timeout", "15",
                ]))
        '''), encoding="utf-8")
        process = subprocess.Popen([sys.executable, str(helper), str(self.root)],
                                   cwd=PROJECT_ROOT, env=_environment(), start_new_session=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        worker_pid = None
        try:
            run_dir = self.root / "outputs" / "cancel"
            marker = run_dir / "worker.pid"
            deadline = time.monotonic() + 10
            while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            if not marker.exists():
                process.kill()
                output, _ = process.communicate(timeout=3)
                self.fail(f"Sleeping worker did not start: {output}")
            worker_pid = int(marker.read_text())
            self.assertTrue(_running(worker_pid))
            process.send_signal(signal.SIGTERM)
            output, _ = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 1, output)
            self.assertTrue(_wait_stopped(worker_pid), f"Worker {worker_pid} survived SIGTERM cleanup")
            errors = json.loads((run_dir / "errors.json").read_text())
            self.assertTrue(any("interrupted" in error["error"].lower() for error in errors))
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["n_done"], 0)
            self.assertEqual(summary["n_missing"], 1)
        finally:
            if worker_pid is not None:
                _kill_group(worker_pid)
            _kill_group(process.pid)
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=3)

    def test_descendant_is_killed_after_its_group_leader_exits(self):
        leader_code = textwrap.dedent('''
            import subprocess
            import sys
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
            print(child.pid, flush=True)
        ''')
        leader = subprocess.Popen([sys.executable, "-c", leader_code], start_new_session=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        child_pid = None
        try:
            output, errors = leader.communicate(timeout=5)
            self.assertEqual(leader.returncode, 0, errors)
            child_pid = int(output.strip())
            self.assertTrue(_running(child_pid))
            self.assertEqual(os.getpgid(child_pid), leader.pid)
            self.assertFalse(_ProcessAdapter(leader).is_alive())
            stop_workers([_ProcessAdapter(leader)])
            self.assertTrue(_wait_stopped(child_pid), "Descendant survived after its leader exited")
        finally:
            _kill_group(leader.pid)
            if leader.poll() is None:
                leader.kill()
            leader.communicate(timeout=3)

    def test_duplicate_experiment_is_rejected_then_lock_is_released(self):
        run_dir = self.root / "run"
        contender = textwrap.dedent('''
            from pathlib import Path
            import sys
            from general_video_eval.run import experiment_lock
            try:
                with experiment_lock(Path(sys.argv[1])):
                    print("acquired")
            except RuntimeError as exc:
                print(str(exc))
                raise SystemExit(23)
        ''')
        command = [sys.executable, "-c", contender, str(run_dir)]
        with experiment_lock(run_dir):
            blocked = subprocess.run(command, cwd=PROJECT_ROOT, env=_environment(),
                                     capture_output=True, text=True, timeout=5)
            self.assertEqual(blocked.returncode, 23, blocked.stdout + blocked.stderr)
            self.assertIn("already running", blocked.stdout)
        accepted = subprocess.run(command, cwd=PROJECT_ROOT, env=_environment(),
                                  capture_output=True, text=True, timeout=5)
        self.assertEqual(accepted.returncode, 0, accepted.stdout + accepted.stderr)
        self.assertIn("acquired", accepted.stdout)


if __name__ == "__main__":
    unittest.main()
