import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path

LOG_FILE = Path(__file__).parent / "run_log.txt"
STATE_FILE = Path(__file__).parent / "pipeline_state.txt"
SEPARATOR = "=" * 80

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def read_pipeline(order_file: Path) -> list[str]:
    scripts = []
    for line in order_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        scripts.append(line)
    return scripts


def load_completed_steps() -> set[str]:
    if not STATE_FILE.exists():
        return set()
    return set(
        line.strip() for line in STATE_FILE.read_text(encoding="utf-8").splitlines() if line.strip()
    )


def mark_step_completed(script_path: str) -> None:
    with open(STATE_FILE, "a", encoding="utf-8") as f:
        f.write(script_path + "\n")


def run_one(script_path: str, log) -> int:
    start_time = datetime.datetime.now()
    header = (
        f"\n{SEPARATOR}\n"
        f"SCRIPT : {script_path}\n"
        f"START  : {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{SEPARATOR}\n"
    )
    print(header, end="")
    log.write(header)
    log.flush()

    child_env = os.environ.copy()
    child_env["PYTHONUTF8"] = "1"
    child_env["PYTHONIOENCODING"] = "utf-8"

    process = subprocess.Popen(
        [sys.executable, "-u", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=child_env,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        log.write(line)
        log.flush()
    process.wait()

    end_time = datetime.datetime.now()
    footer = (
        f"\n{'-' * 80}\n"
        f"END       : {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"DURATION  : {end_time - start_time}\n"
        f"EXIT CODE : {process.returncode}\n"
        f"{SEPARATOR}\n\n"
    )
    print(footer, end="")
    log.write(footer)
    return process.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("order_file", help="Text file containing the order of scripts to run")
    parser.add_argument("--continue-on-error", action="store_true",
                         help="Keep running remaining steps even if one fails (default: stop on error)")
    parser.add_argument("--resume", action="store_true",
                         help="Skip steps already recorded as completed in pipeline_state.txt")
    parser.add_argument("--reset", action="store_true",
                         help="Clear pipeline_state.txt before running (start fully from scratch)")
    args = parser.parse_args()

    if args.reset and STATE_FILE.exists():
        STATE_FILE.unlink()
        print("pipeline_state.txt cleared — running everything from scratch.")

    scripts = read_pipeline(Path(args.order_file))
    if not scripts:
        print("No scripts found in the order file.")
        sys.exit(1)

    completed = load_completed_steps() if args.resume else set()
    if completed:
        print(f"{len(completed)} steps already completed, skipping them.")

    print(f"{len(scripts)} steps found to run.\n")

    with open(LOG_FILE, "a", encoding="utf-8") as log:
        run_header = (
            f"\n{'#' * 80}\n"
            f"PIPELINE RUN START: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'#' * 80}\n"
        )
        print(run_header, end="")
        log.write(run_header)

        for i, script in enumerate(scripts, 1):
            if script in completed:
                print(f"[{i}/{len(scripts)}] Skipped (already completed): {script}")
                continue

            print(f"[{i}/{len(scripts)}] Running: {script}")
            exit_code = run_one(script, log)

            if exit_code == 0:
                mark_step_completed(script)
            elif not args.continue_on_error:
                msg = f"\n❌ Step '{script}' stopped with exit code {exit_code}. Pipeline aborted.\n"
                print(msg)
                log.write(msg)
                sys.exit(exit_code)

        run_footer = (
            f"\n{'#' * 80}\n"
            f"PIPELINE RUN COMPLETE: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'#' * 80}\n"
        )
        print(run_footer, end="")
        log.write(run_footer)


if __name__ == "__main__":
    main()