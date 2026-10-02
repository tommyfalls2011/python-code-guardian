from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExecutionResult:
    passed: bool
    returncode: int
    command: list[str]
    stdout: str
    stderr: str


class PythonTestRunner:
    """Run a project's pytest test suite safely as a subprocess."""

    def __init__(self, timeout: float = 120.0) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be > 0")
        self.timeout = timeout

    def run(self, root: Path) -> ExecutionResult:
        root = root.resolve()

        if not root.exists():
            raise FileNotFoundError(root)

        if not root.is_dir():
            raise ValueError(
                f"Test root must be a directory: {root}"
            )

        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
        ]

        try:
            completed = subprocess.run(
                command,
                cwd=root,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""

            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")

            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")

            return ExecutionResult(
                passed=False,
                returncode=-1,
                command=command,
                stdout=stdout,
                stderr=(stderr + "\nTest suite timed out.").strip(),
            )

        return ExecutionResult(
            passed=completed.returncode == 0,
            returncode=completed.returncode,
            command=command,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
