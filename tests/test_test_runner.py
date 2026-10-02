from pathlib import Path

from codeguardian.test_runner import PythonTestRunner


def test_runner_passes_project(tmp_path: Path):
    (tmp_path / "test_example.py").write_text(
        "def test_answer():\n"
        "    assert 2 + 2 == 4\n",
        encoding="utf-8",
    )

    result = PythonTestRunner().run(tmp_path)

    assert result.passed is True
    assert result.returncode == 0
    assert result.command[-1] == "-q"
    assert "passed" in result.stdout


def test_runner_reports_failed_project(tmp_path: Path):
    (tmp_path / "test_example.py").write_text(
        "def test_answer():\n"
        "    assert 2 + 2 == 5\n",
        encoding="utf-8",
    )

    result = PythonTestRunner().run(tmp_path)

    assert result.passed is False
    assert result.returncode != 0


def test_runner_rejects_file_root(tmp_path: Path):
    path = tmp_path / "not_a_directory.py"
    path.write_text("x = 1\n", encoding="utf-8")

    try:
        PythonTestRunner().run(path)
    except ValueError as exc:
        assert "directory" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_runner_rejects_nonexistent_root(tmp_path: Path):
    path = tmp_path / "missing"

    try:
        PythonTestRunner().run(path)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError")


def test_runner_rejects_invalid_timeout():
    try:
        PythonTestRunner(timeout=0)
    except ValueError as exc:
        assert "timeout" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
