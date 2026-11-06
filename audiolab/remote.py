"""Same interface as audiolab.train, but the run executes on a Modal GPU (see modal_app.py).

    python -m audiolab.remote --config configs/baseline.json --run-dir runs/baseline

Runs locally without torch: it only needs `modal`. Files streamed back from the GPU are written
into the local run directory, so status.json / train.log / metrics.json appear as with a local run.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def write(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides as key=value (value parsed as JSON)")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    for kv in args.set:
        key, value = kv.split("=", 1)
        cfg[key] = json.loads(value)
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    write(run_dir / "config.json", json.dumps(cfg, indent=2).encode())
    write(run_dir / "status.json", json.dumps({"state": "starting", "backend": "modal"}).encode())

    sys.path.insert(0, str(ROOT))
    import modal
    from modal_app import app, train_remote

    import random
    import time

    try:
        for attempt in range(6):
            try:
                with modal.enable_output(), app.run():
                    for files in train_remote.remote_gen(cfg, run_dir.name):
                        for name, data in files.items():
                            write(run_dir / name, data)
                break
            except modal.exception.ResourceExhaustedError as exc:
                # parallel launches can hit Modal's app-create rate limit; nothing ran yet, so retry
                if attempt == 5:
                    raise
                wait = 15 * (attempt + 1) + random.uniform(0, 10)
                print(f"rate limited ({exc}); retrying in {wait:.0f} s", flush=True)
                time.sleep(wait)
    except BaseException as exc:
        status_path = run_dir / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if status.get("state") != "done":
            status.update(state="failed", error=f"remote: {type(exc).__name__}: {exc}")
            write(status_path, json.dumps(status, indent=2).encode())
        raise
    status = json.loads((run_dir / "status.json").read_text())
    if status.get("state") != "done":
        sys.exit(f"remote run ended in state {status.get('state')}: {status.get('error')}")


if __name__ == "__main__":
    main()
