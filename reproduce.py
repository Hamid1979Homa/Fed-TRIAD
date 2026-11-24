"""Run and verify the Fed-TRIAD manuscript experiments in a new directory."""

import argparse
import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from run_files import directory_lock, sha256, write_json


def environment(mode):
    env = os.environ.copy()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    env["PYTHONHASHSEED"] = "0"
    env["MPLBACKEND"] = "Agg"
    if mode == "reference":
        env["OPENBLAS_CORETYPE"] = "SkylakeX"
        env["NPY_DISABLE_CPU_FEATURES"] = "AVX512_SPR,AVX512_ICL"
    return env


def lock_versions():
    packages = {}
    for line in (ROOT / "requirements-lock.txt").read_text(encoding="utf-8").splitlines():
        if "==" in line and line[:1].isalnum():
            name, _, version = line.partition("==")
            packages[name] = version.split()[0]
    return packages


def preflight(mode, env):
    if mode == "reference" and (platform.system() != "Linux" or platform.machine() != "x86_64"
                                or platform.python_version() != "3.13.5"):
        raise ValueError("The exact-reference profile uses Linux x86-64 and Python 3.13.5. "
                         "For another environment use --mode native.")
    for name, version in lock_versions().items():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise ValueError(f"Missing {name}=={version}. Install requirements-lock.txt.") from exc
        if installed != version:
            raise ValueError(f"{name}=={installed} is installed, expected {version}. Install requirements-lock.txt.")
    if mode == "reference":
        # Detect the CPU capability before importing a BLAS library with a forced kernel.
        check_env = env.copy()
        check_env.pop("OPENBLAS_CORETYPE", None)
        probe = ("from numpy._core import _multiarray_umath as m; "
                 "raise SystemExit(0 if m.__cpu_features__.get('X86_V4') else 1)")
        if subprocess.run([sys.executable, "-c", probe], env=check_env, check=False).returncode:
            raise ValueError("Reference mode requires x86-64-v4 CPU instructions. Use --mode native on this host.")


def run(script, *args, env, check=True):
    command = [sys.executable, "-u", str(ROOT / script), *(str(arg) for arg in args)]
    print(f"Running {script}", flush=True)
    return subprocess.run(command, cwd=ROOT, env=env, check=check).returncode


def reproduce(args, env):
    out = args.out.resolve()
    archive = ROOT / "reference"
    result_dir = out / "results"
    outputs = out / "outputs"
    with directory_lock(out):
        manifest = out / "reproduction.json"
        identity = {
            "mode": args.mode,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "references_sha256": {name: sha256(archive / name) for name in (
                "numbers_reference.json", "paper_tables.json", "drift_m12_first_study_seeds1-5.npy",
            )},
        }
        if args.resume:
            if manifest.is_file() and json.loads(manifest.read_text(encoding="utf-8")) != identity:
                raise ValueError("Resume requires the same output directory, reference profile and local environment.")
            if not manifest.exists() and not (result_dir / "study.json").is_file():
                raise ValueError("Resume requires an existing reproduction or sweep manifest.")
        else:
            if any(child.name != ".writer.lock" for child in out.iterdir()):
                raise ValueError(f"Choose an empty output directory. {out} already contains files.")
            write_json(manifest, identity)
        run("src/theory_checks.py", env=env)
        run("src/test_sim.py", archive / "drift_m12_first_study_seeds1-5.npy", env=env)
        run("src/run_sweep.py", "--out", result_dir, "--jobs", args.jobs,
            *(("--resume",) if args.resume else ()), env=env)
        if not manifest.exists():
            write_json(manifest, identity)
        run("src/analyze.py", result_dir, outputs, env=env)
        strict = args.mode == "reference"
        tables = run("src/check_against_reference.py", outputs / "numbers.json", archive / "numbers_reference.json",
                     "--criterion", "exact" if strict else "tables",
                     "--report", out / "numbers_verification.json", env=env, check=False)
        raw = run("src/verify_raw.py", result_dir, ROOT / "results",
                  "--report", out / "raw_verification.json",
                  *(("--require-exact",) if strict else ()), env=env, check=False)
    return int(bool(tables or raw))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("reproduced"))
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--mode", choices=("reference", "native"), default="reference")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    env = environment(args.mode)
    try:
        preflight(args.mode, env)
        return reproduce(args, env)
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
