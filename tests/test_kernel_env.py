"""``FASTGOERTZEL_KERNEL`` is read once at import, so each case runs in a child process.

A wrong value must fail the import loudly instead of silently running the default kernel, and
``avx2`` must never be forced onto a machine that cannot execute it.
"""

import os
import subprocess
import sys

import pytest

PROBE = "import fastgoertzel._core as c; print(c._kernel)"


def run(value: str | None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("FASTGOERTZEL_KERNEL", None)
    if value is not None:
        env["FASTGOERTZEL_KERNEL"] = value
    return subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, env=env)


@pytest.mark.parametrize("value", [None, "", "auto", "AUTO"])
def test_auto_matches_the_default_choice(value):
    r = run(value)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == run(None).stdout.strip()


@pytest.mark.parametrize("value", ["baseline", "Baseline", "BASELINE"])
def test_baseline_is_honoured_case_insensitively(value):
    r = run(value)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "baseline"


@pytest.mark.parametrize("value", ["bogus", "avx512", "sse2", "1"])
def test_unknown_value_fails_the_import(value):
    r = run(value)
    assert r.returncode != 0
    assert "FASTGOERTZEL_KERNEL must be" in r.stderr and value in r.stderr


def test_forced_avx2_is_checked_against_the_cpu():
    # Judge by what a child process selects on its own, not by this process: under qemu-user
    # (the release workflow's pre-AVX check) the parent is emulated as a Nehalem while the
    # children it spawns run natively on the host CPU.
    auto = run(None).stdout.strip()
    r = run("avx2")
    if auto == "avx2":  # the child's machine can run it, so forcing it is fine
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "avx2"
    else:  # no AVX2 there (or a non-x86 build): must refuse, not crash later
        assert r.returncode != 0
        assert "FASTGOERTZEL_KERNEL=avx2 requested" in r.stderr
