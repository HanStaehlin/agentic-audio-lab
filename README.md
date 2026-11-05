# Agentic Audio Lab

An LLM agent that runs speech-enhancement experiments on its own: it plans a run, launches it on a GPU, monitors it, diagnoses failures from the logs, compares results against a baseline and writes a report. A human approves every launch.

The audio side is deliberately small: a causal GRU that predicts a spectral mask, so it can run as a stream (one STFT frame in, one frame out).

## How it fits together

```
agent/run_agent.py   tool-use loop (Claude, Anthropic SDK)
agent/tools.py       tools + safeguards: list_runs, launch_run, get_run_status,
                     wait_for_run, compare_runs, write_report
audiolab/train.py    one run: train, evaluate, write status.json / metrics.json
audiolab/model.py    MaskGRU (causal, streaming-capable)
audiolab/metrics.py  PESQ (wideband), STOI, SI-SDR, streaming latency
configs/baseline.json
notebooks/colab.ipynb
```

Each run lives in `runs/<run_id>/` with its config, overrides, the agent's reason for the run, logs, `status.json` and `metrics.json`. Everything the agent decides is therefore reproducible from files.

## Safeguards

- **Human approval:** every launch shows the overrides and the hypothesis and waits for `y`.
- **Whitelisted changes:** only listed config keys within fixed ranges (`ALLOWED` in `agent/tools.py`).
- **Budget:** a hard cap on launches (`--max-runs`), and one active run at a time.
- **Failures surface:** a NaN loss or a crashed process shows up as `failed` / `crashed` with the error and log tail, so the agent has to diagnose it before going on.

## Data

[VoiceBank-DEMAND](https://datashare.ed.ac.uk/handle/10283/2791): 11,572 training and 824 test utterances, resampled to 16 kHz by `scripts/prepare_data.py`.

## Metrics

| Metric | Meaning |
|---|---|
| PESQ (wb) | perceived speech quality, 1 to 4.5 |
| STOI | intelligibility, 0 to 1 |
| SI-SDR | scale-invariant signal-to-distortion ratio, dB |
| ms per hop, RTF | CPU compute per STFT hop, batch 1; RTF below 1 is faster than real time |
| algorithmic latency | one window (n_fft / sample rate) |

Metrics are reported for the enhanced output and for the noisy input, so every result has its own reference point.

## Run it

On Colab: open `notebooks/colab.ipynb` with a GPU runtime and run the cells.

Locally:

```bash
pip install -r requirements.txt
python scripts/prepare_data.py --raw-dir data/raw --out-dir data/vb16k
python -m audiolab.train --config configs/baseline.json --run-dir runs/baseline
export ANTHROPIC_API_KEY=...
python -m agent.run_agent --goal "Improve PESQ over the baseline, rtf below 0.1" --baseline baseline --max-runs 6
```

## Results

_To be added: baseline vs. best agent run, and one example report._
