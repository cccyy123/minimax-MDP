import subprocess
import sys
import time
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def run_script(script_name: str) -> float:
    print(f"\n=== Running {script_name} ===")
    start = time.perf_counter()
    result = subprocess.run([sys.executable, script_name], cwd=BASE_DIR)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        raise SystemExit(f"{script_name} failed with exit code {result.returncode}")
    print(f"=== Finished {script_name} in {elapsed:.3f}s ===")
    return elapsed


if __name__ == "__main__":
    total_start = time.perf_counter()
    timings = {}

    for script in (
        "g.py",
        "rr.py",
        "export_avg_time_by_ins_acc.py",
        "plot_perf_time_scatter.py",
    ):
        timings[script] = run_script(script)

    total_elapsed = time.perf_counter() - total_start

    print("\nRun time summary:")
    for script, elapsed in timings.items():
        print(f"- {script}: {elapsed:.3f}s")
    print(f"- total: {total_elapsed:.3f}s")
    print("\nAll scripts completed successfully.")
