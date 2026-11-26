"""Head-local SSH preparation shared by foreground and detached launchers.

No private-key contents are stored in the run specification or node logs.
The parent transfers this session's metadata to the detached coordinator;
only that coordinator cleans up the temporary resources after a successful
Popen. Constructing a session never creates resources or changes an agent.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid


def ssh_args(spec, host):
    """Offer the configured key first while retaining normal identity fallback."""
    session = spec.get("_ssh", {})
    if session.get("cleaned"):
        session = {}
    args = [
        "ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
        "-o", "StrictHostKeyChecking=accept-new", "-o", "LogLevel=ERROR",
    ]
    key = session.get("effective_key", spec.get("ssh_key", ""))
    if key:
        args += ["-i", str(key)]
    if session.get("control_path"):
        args += ["-o", "ControlMaster=auto", "-o", f"ControlPath={session['control_path']}",
                 "-o", "ControlPersist=600"]
    return args + [f"{spec['ssh_user']}@{host}"]


class SSHSession:
    def __init__(self, spec):
        self.spec = spec
        self.record = spec.get("_ssh")

    def environment(self):
        """Preserve the caller's agent unless this launch started its own one."""
        env = os.environ.copy()
        if self.record and not self.record.get("cleaned"):
            env.update(self.record.get("agent_env", {}))
        return env

    def _owns_directory(self):
        if not self.record or self.record.get("cleaned"):
            return False
        directory = Path(self.record.get("temp_dir", ""))
        marker = directory / ".owner"
        try:
            return (directory.name.startswith("ar-mm-ssh-") and not directory.is_symlink()
                    and directory.is_dir() and not marker.is_symlink()
                    and marker.read_text() == self.record.get("owner_token"))
        except OSError:
            return False

    def _prepare_resources(self):
        if self.record:
            if not self._owns_directory():
                raise RuntimeError("SSH session resources are missing or no longer owned by this launch")
            return
        # Keep control socket names short even when the caller's TMPDIR is long.
        directory = Path(tempfile.mkdtemp(prefix="ar-mm-ssh-", dir="/tmp"))
        directory.chmod(0o700)
        self.record = {
            "temp_dir": str(directory), "owner_token": uuid.uuid4().hex,
            "effective_key": "", "control_path": str(directory / "cm-%C"),
        }
        self.spec["_ssh"] = self.record
        (directory / ".owner").write_text(self.record["owner_token"])
        key_name = self.spec.get("ssh_key", "")
        if not key_name:
            return
        key = Path(key_name).expanduser()
        if not key.is_file() or not os.access(key, os.R_OK):
            raise RuntimeError(f"SSH private key is not readable: {key}")
        mode = stat.S_IMODE(key.stat().st_mode)
        if mode not in (0o600, 0o400):
            private_copy = directory / "identity"
            # The destination is private from its first write; never chmod the
            # user's original key or put its bytes in shared launch storage.
            descriptor = os.open(private_copy, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as target, key.open("rb") as source:
                shutil.copyfileobj(source, target)
            self.record["effective_key"] = str(private_copy)
            print(f"SSH key permissions are {mode:o}; using a launch-local 0600 copy.", flush=True)
        else:
            self.record["effective_key"] = str(key)

    @staticmethod
    def _can_prompt():
        if not sys.stdin.isatty() or not sys.stderr.isatty():
            return False
        try:
            descriptor = os.open("/dev/tty", os.O_RDWR)
            os.close(descriptor)
            return True
        except OSError:
            return False

    def _unlock_private_agent(self):
        """Parse ssh-agent's two assignments without evaluating shell output."""
        socket = str(Path(self.record["temp_dir"]) / "agent.sock")
        result = subprocess.run(["ssh-agent", "-a", socket, "-s"], capture_output=True,
                                text=True, check=False, timeout=15, env=self.environment())
        pid_match = re.search(r"(?:^|\n)SSH_AGENT_PID=([1-9][0-9]*);", result.stdout)
        socket_match = re.search(r"(?:^|\n)SSH_AUTH_SOCK=([^;\r\n]+);", result.stdout)
        if result.returncode or not pid_match or not socket_match or socket_match.group(1) != socket:
            raise RuntimeError("Could not start a private ssh-agent for this launch: "
                               + (result.stderr.strip() or "unexpected ssh-agent output"))
        self.record["agent_pid"] = int(pid_match.group(1))
        self.record["agent_env"] = {"SSH_AUTH_SOCK": socket, "SSH_AGENT_PID": pid_match.group(1)}
        key = self.record.get("effective_key")
        command = ["ssh-add", key] if key else ["ssh-add"]
        print("SSH authentication needs a key. Unlock it once before background submission.", flush=True)
        result = subprocess.run(command, check=False, timeout=300, env=self.environment())
        if result.returncode:
            raise RuntimeError(f"ssh-add failed (exit {result.returncode}); the training pipeline was not submitted")

    def _probe(self, host, *, retry=False):
        log = Path(self.spec["state_dir"]) / "nodes" / f"ssh-{host}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(ssh_args(self.spec, host) + ["true"],
                                    stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                    check=False, timeout=45, env=self.environment())
            output, error, code = result.stdout, result.stderr, result.returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            output = getattr(exc, "stdout", "") or ""
            error = getattr(exc, "stderr", "") or str(exc)
            if isinstance(output, bytes):
                output = output.decode(errors="replace")
            if isinstance(error, bytes):
                error = error.decode(errors="replace")
            code = "timeout" if isinstance(exc, subprocess.TimeoutExpired) else "not started"
        with log.open("a" if retry else "w") as stream:
            stream.write(f"SSH {self.spec['ssh_user']}@{host}: exit={code}\n")
            stream.write(f"stdout:\n{output}\nstderr:\n{error}\n")
        return code, error, log

    def prepare(self, interactive=False):
        """Check all worker connections before the caller detaches the pipeline."""
        try:
            self._prepare_resources()
            errors = []
            agent_attempted = bool(self.record.get("agent_pid"))
            for host in self.spec["worker_hosts"]:
                code, error, log = self._probe(host)
                auth_error = any(word in error.lower() for word in ("permission denied", "passphrase"))
                if code and auth_error and not agent_attempted and interactive and self._can_prompt():
                    agent_attempted = True
                    try:
                        self._unlock_private_agent()
                    except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
                        with log.open("a") as stream:
                            stream.write(f"Key unlock failed: {exc}\n")
                        raise RuntimeError(f"SSH {host}: {exc}. Details: {log}") from exc
                    code, error, log = self._probe(host, retry=True)
                if code:
                    message = (f"SSH {self.spec['ssh_user']}@{host} failed (exit={code}): "
                               f"{error.strip() or 'no stderr returned'}\nDetails: {log}")
                    if auth_error and not (interactive and self._can_prompt()):
                        key = self.spec.get("ssh_key", "")
                        unlock = "ssh-add" + (" " + shlex.quote(key) if key else "")
                        message += (f"\nNo interactive terminal is available. Run {unlock} in the head-node "
                                    "terminal with a working ssh-agent, verify SSH access, then retry.")
                    if "IDENTIFICATION HAS CHANGED" in error:
                        message += "\nThe recorded host key changed; verify the host identity before updating known_hosts."
                    errors.append(message)
                else:
                    print(f"SSH {self.spec['ssh_user']}@{host}: ok ({log})", flush=True)
            if errors:
                raise RuntimeError("SSH worker checks failed:\n" + "\n".join(errors))
            self.record["prepared"] = True
            return self.record
        except BaseException:
            self.cleanup()
            raise

    def cleanup(self):
        """Best-effort release of this launch's controls, agent and private files."""
        problems = []
        if not self._owns_directory():
            return problems
        for host in self.spec.get("worker_hosts", []):
            try:
                # -O applies only to the ControlPath in our private directory.
                args = ssh_args(self.spec, host)
                subprocess.run(args[:-1] + ["-O", "exit", args[-1]], stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               check=False, timeout=10, env=self.environment())
            except (OSError, subprocess.SubprocessError) as exc:
                problems.append(f"SSH control cleanup for {host}: {exc}")
        agent = self.record.get("agent_env", {})
        expected_socket = str(Path(self.record["temp_dir"]) / "agent.sock")
        if (self.record.get("agent_pid") and agent.get("SSH_AGENT_PID") == str(self.record["agent_pid"])
                and agent.get("SSH_AUTH_SOCK") == expected_socket):
            try:
                subprocess.run(["ssh-agent", "-k"], stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               check=False, timeout=10, env=self.environment())
            except (OSError, subprocess.SubprocessError) as exc:
                problems.append(f"Private ssh-agent cleanup: {exc}")
        try:
            shutil.rmtree(self.record["temp_dir"])
        except OSError as exc:
            problems.append(f"SSH temporary directory cleanup: {exc}")
        self.record["cleaned"] = True
        for problem in problems:
            print("WARNING: " + problem, file=sys.stderr)
        return problems
