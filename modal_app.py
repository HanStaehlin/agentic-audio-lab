"""Run training on a Modal GPU.

    modal run modal_app.py::prepare_data          # once: download + resample VoiceBank-DEMAND into the volume
    python -m audiolab.remote --config configs/baseline.json --run-dir runs/baseline

The agent uses audiolab.remote for every launch (run_agent --backend modal): the local process
streams status, logs, metrics and weights back into runs/<run_id>/, so the agent's tools work
exactly as with local runs. A copy of every run also stays in the volume under /vol/runs.
"""
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import modal

app = modal.App("agentic-audio-lab")
data_app = modal.App("agentic-audio-lab-data")  # CPU only, separate so it runs without GPU access
vol = modal.Volume.from_name("agentic-audio-lab", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential")
    .pip_install("torch==2.5.1", "torchaudio==2.5.1", "soundfile", "numpy<2.3", "cython", "pesq", "pystoi")
    .add_local_python_source("audiolab")
    .add_local_file("scripts/prepare_data.py", "/root/prepare_data.py")
)

VOL_DATA = Path("/vol/data/vb16k")
VOL_TAR = Path("/vol/data/vb16k.tar")  # one archive: the volume is slow for many small files
# VoiceBank-DEMAND, CC BY 4.0, https://datashare.ed.ac.uk/handle/10283/2791 (DSpace REST content links)
BITSTREAMS = {
    "clean_trainset_28spk_wav": "245452b6-6235-44b6-a6f9-e7eb19797769",
    "noisy_trainset_28spk_wav": "ecb5a102-bb00-46d3-8af5-40c79823b837",
    "clean_testset_wav": "dec213d3-bf57-4777-9663-c24bdce92d5e",
    "noisy_testset_wav": "13c1bfbf-14a6-41db-9b41-8f7310f01ad5",
}


@data_app.function(image=image, volumes={"/vol": vol}, cpu=4, timeout=3 * 3600)
def prepare_data():
    if len(list((VOL_DATA / "test" / "clean").glob("*.wav"))) == 824:
        print("data already prepared")
        return
    raw = Path("/tmp/raw")
    raw.mkdir(parents=True, exist_ok=True)
    for name, uuid in BITSTREAMS.items():
        url = f"https://datashare.ed.ac.uk/server/api/core/bitstreams/{uuid}/content"
        print(f"downloading {name}", flush=True)
        urllib.request.urlretrieve(url, f"/tmp/{name}.zip")
        with zipfile.ZipFile(f"/tmp/{name}.zip") as z:
            z.extractall(raw)
        Path(f"/tmp/{name}.zip").unlink()
    subprocess.run([sys.executable, "/root/prepare_data.py", "--raw-dir", str(raw), "--out-dir", "/tmp/vb16k"], check=True)
    shutil.rmtree(VOL_DATA, ignore_errors=True)
    shutil.copytree("/tmp/vb16k", VOL_DATA)
    _write_tar(Path("/tmp/vb16k"))
    vol.commit()
    print("done:", {s: len(list((VOL_DATA / s / "clean").glob("*.wav"))) for s in ("train", "test")})


def _write_tar(src):
    import tarfile

    with tarfile.open("/tmp/vb16k.tar", "w") as tar:
        tar.add(src, arcname="vb16k")
    shutil.copy("/tmp/vb16k.tar", VOL_TAR)


@data_app.function(image=image, volumes={"/vol": vol}, cpu=4, timeout=2 * 3600)
def pack_data():
    """Pack the prepared wav files into one tar (reads in parallel, since each volume read is slow)."""
    from concurrent.futures import ThreadPoolExecutor

    files = [p for p in VOL_DATA.rglob("*.wav")]
    print(f"packing {len(files)} files", flush=True)

    def copy(p):
        dst = Path("/tmp/vb16k") / p.relative_to(VOL_DATA)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(p.read_bytes())

    with ThreadPoolExecutor(64) as pool:
        for i, _ in enumerate(pool.map(copy, files)):
            if i % 2000 == 0:
                print(f"{i}/{len(files)}", flush=True)
    _write_tar(Path("/tmp/vb16k"))
    vol.commit()
    print("packed", VOL_TAR.stat().st_size // 2**20, "MiB")


def _snapshot(run_dir, names):
    out = {}
    for name in names:
        p = run_dir / name
        if p.exists():
            out[name] = p.read_bytes()
    return out


@app.function(image=image, gpu="T4", cpu=8, volumes={"/vol": vol}, timeout=4 * 3600)
def train_remote(cfg: dict, run_id: str):
    """Run audiolab.train on the GPU and stream the run files back every 15 s."""
    import json

    import tarfile

    with tarfile.open(VOL_TAR) as tar:  # one sequential read; local disk is fast for random access
        tar.extractall("/tmp")
    shutil.move("/tmp/vb16k", "/tmp/data")
    cfg = {**cfg, "data_dir": "/tmp/data"}
    run_dir = Path("/tmp/run")
    run_dir.mkdir()
    (run_dir / "cfg.json").write_text(json.dumps(cfg))
    with open(run_dir / "stdout.log", "w") as out:
        proc = subprocess.Popen([sys.executable, "-m", "audiolab.train", "--config", str(run_dir / "cfg.json"),
                                 "--run-dir", str(run_dir)], stdout=out, stderr=subprocess.STDOUT, cwd="/root")
        while proc.poll() is None:
            time.sleep(15)
            files = _snapshot(run_dir, ["train.log", "status.json"])
            # a final state is only sent at the very end, after metrics.json; otherwise the agent
            # can see "done" in the gap before the metrics arrive
            if b'"state": "done"' in files.get("status.json", b"") or b'"state": "failed"' in files.get("status.json", b""):
                files.pop("status.json")
            yield files
    shutil.copytree(run_dir, f"/vol/runs/{run_id}", dirs_exist_ok=True)
    vol.commit()
    # status.json last, so the agent never sees "done" before metrics.json exists
    yield _snapshot(run_dir, ["train.log", "metrics.json", "model.pt", "stdout.log"])
    yield _snapshot(run_dir, ["status.json"])
