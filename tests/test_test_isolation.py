import os
import subprocess  # nosec B404
import sys
from pathlib import Path


def test_all_tests_collect_without_network_or_seaborn_downloads(tmp_path):
    script = """
import socket
import seaborn
import pytest

def forbidden(*args, **kwargs):
    raise AssertionError('Test collection must not access the network or seaborn datasets')

socket.socket.connect = forbidden
seaborn.load_dataset = forbidden
raise SystemExit(pytest.main(['tests', '--collect-only', '-q', '-o', 'log_cli=false']))
"""
    # No shell or user-supplied command text: only this interpreter and the
    # constant script above. A fresh process is needed to test import-time I/O.
    result = subprocess.run(  # nosec B603
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        shell=False,
        env={**os.environ, "SEABORN_DATA": str(tmp_path)},
        timeout=60,
    )
    if result.returncode:
        raise AssertionError(result.stdout + result.stderr)
