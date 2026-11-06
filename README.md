# Agentic Audio Lab

**How much speech quality does each millisecond of latency cost?** This repo answers that question for a small, real-time speech enhancer, using an LLM agent that plans, runs, diagnoses and reports the experiments on cloud GPUs.

## The question

Speech enhancement in earbuds and hearing devices has two hard limits:

- **Algorithmic latency.** In transparency or conversation modes, the processed sound has to reach the ear within roughly 10 ms, or the wearer hears an echo of their own voice. Many enhancement models use a 32 ms STFT window, so they cannot run there.
- **Compute.** The model runs on a small processor. Here the budget is a real-time factor below 0.1 on one CPU thread, so the model takes at most 10% of the time it processes.

Shrinking the window lowers latency but also frequency resolution, and smaller hops raise quality but also compute. Which configuration is best at 8, 16 and 32 ms is an empirical question. Answering it means running a grid of window, hop and model sizes, repeating the promising ones over seeds, and deciding what is a real difference and what is noise. That is the repetitive work the agent does.

## Answer

![Quality vs. latency: Pareto front](results/latency/pareto.png)

*Each point is a model configuration that runs below 10% of one CPU core. The orange staircase is the Pareto front: no configuration has both lower latency and higher quality than the points on it. Plot: `python scripts/plot_pareto.py`, from [runs.csv](results/latency/runs.csv).*

Best configuration per latency budget, with rtf < 0.1, 3 seeds each, on the full VoiceBank-DEMAND test set (824 files):

| Latency budget | Best configuration | PESQ (mean ± std) | STOI | SI-SDR | rtf |
|---|---|---|---|---|---|
| 8 ms | hop 2 ms, 1 GRU layer × 128 | 2.450 ± 0.047 | 0.932 | 17.5 dB | 0.070–0.079 |
| 16 ms | hop 4 ms, 2 layers × 256 | **2.503 ± 0.016** | 0.936 | 18.6 dB | 0.089–0.097 |
| 32 ms | hop 8 ms, 2 layers × 256 (default) | 2.470 ± 0.003 | 0.934 | 18.8 dB | 0.04–0.05 |
| *noisy input* | | *1.97* | *0.921* | *8.4 dB* | |

- **16 ms costs nothing compared with 32 ms.** The 16 ms model is better on every seed: its worst seed (2.485) beats the best 32 ms seed (2.472).
- **Going to 8 ms costs about 0.05 PESQ.** That is suggestive only, because the 8 ms seeds vary a lot (std 0.047).
- **The hop size matters more than the window.** At 16 ms a 4 ms hop beats an 8 ms hop by 0.08 PESQ, and at 8 ms a 2 ms hop beats a 4 ms hop by 0.12. GRU width barely mattered.
- **At 8 ms, compute is the limit, not frequency resolution.** Each GRU layer costs about 0.1 ms per frame whatever its width. At a 2 ms hop that leaves room for only one layer. Two layers reach 2.50 PESQ, as good as 16 ms, but at rtf 0.13. Cheaper per-frame inference (a fused or exported GRU step) is therefore the most promising next step.

Details, failures and next steps are in the agent's own report: [results/latency/report.md](results/latency/report.md). All 21 runs, including each run's hypothesis, are listed in [results/latency/runs.csv](results/latency/runs.csv), and the agent's tool calls in [results/latency/agent_transcript.txt](results/latency/agent_transcript.txt).

## How the agent works

```
you: goal, budget ──► agent (Claude, tool-use loop) ──► tools ──► training runs on Modal GPUs (up to 4 in parallel)
                        ▲                                  │
                        └──── status, logs, metrics ◄──────┘          ──► report + trade-off plot
```

| Tool | What it does |
|---|---|
| `list_runs` | runs of the study with state, PESQ, latency, rtf; budget and free parallel slots |
| `launch_run` | starts a run from the base config plus overrides, with the hypothesis it tests |
| `wait_for_run`, `get_run_status` | block until a run ends; read its log tail to diagnose failures |
| `compare_runs` | metrics against a reference, and configurations pooled over seeds (mean, std) |
| `plot_tradeoff` | the plot above |
| `write_report` | the final Markdown report |

**Safeguards:**
- **Human approval:** every launch shows the overrides and the hypothesis and waits for `y`. The latency study ran with `--auto-approve` within a fixed budget.
- **Whitelist:** only listed config keys can change, within fixed ranges. Invalid STFT setups (hop above half the window) are rejected.
- **Budget:** a hard cap on launches and on parallel runs.
- **Failures surface:** NaN losses, crashed processes and remote errors show up as `failed` or `crashed`, with the error and the log tail.

**What went wrong during the study, and who fixed it:**
- **Rate limit:** 4 of 20 launches hit Modal's app-creation rate limit when 4 runs started at once.
- **Preemption:** one run was preempted by the platform.
- **The agent's response:** it diagnosed every case as an infrastructure failure rather than a bad config, relaunched the runs it needed, and chose not to relaunch one that no longer mattered for the answer.
- **My fix:** during the study I added a retry with backoff for the rate limit.
- **An earlier bug:** in an earlier round, a race condition let the agent see a finished run before its metrics had arrived. The agent noticed the missing metrics and refused to draw conclusions from that run. I then fixed the race.

**Why seeds matter:** a first hyperparameter round with one seed per run produced PESQ swings of about ±0.05 that did not follow any pattern. The agent flagged this in its report. That is why the study requires at least 2 seeds before a difference counts as real.

## Setup

- **Model:** a causal GRU that predicts a spectral mask frame by frame (`audiolab/model.py`), so it can run as a stream: one STFT frame in, one frame out. It is trained with an SI-SDR loss for 10k steps.
- **Data:** [VoiceBank-DEMAND](https://datashare.ed.ac.uk/handle/10283/2791) (CC BY 4.0), resampled to 16 kHz. It has 11,572 training and 824 test utterances.
- **Metrics:**

  | Metric | Meaning |
  |---|---|
  | PESQ (wb) | perceived speech quality, 1 to 4.5 |
  | STOI | intelligibility, 0 to 1 |
  | SI-SDR | scale-invariant signal-to-distortion ratio, dB |
  | algorithmic latency | one STFT window: n_fft / 16 kHz |
  | rtf | compute time per hop on one CPU thread, batch 1, divided by the hop duration |

```
agent/run_agent.py    tool-use loop (Anthropic SDK)
agent/tools.py        tools and safeguards
audiolab/train.py     one run: train, evaluate, write status.json / metrics.json
audiolab/remote.py    the same run on a Modal GPU, streaming the files back
audiolab/model.py     MaskGRU
audiolab/metrics.py   PESQ, STOI, SI-SDR, streaming latency
modal_app.py          Modal image, data volume, GPU function
configs/              baseline and latency-study configs
results/latency/      Pareto plot, the agent's report and plot, run table, agent transcript
scripts/plot_pareto.py  Pareto plot from runs.csv
```

## Run it

On a [Modal](https://modal.com) GPU. The agent and the API key stay on your machine; only training runs remotely.

```bash
pip install -r requirements.txt && modal token new
modal run modal_app.py::data_app.prepare_data        # once: download and resample into a Modal volume
python -m audiolab.remote --config configs/latency_study.json --run-dir runs/latency/ref-w512-h128
python -m agent.run_agent --backend modal --study latency --base-config configs/latency_study.json \
    --baseline ref-w512-h128 --max-runs 20 --max-parallel 4 \
    --goal "For latency budgets of 8, 16 and 32 ms, find the best PESQ with rtf below 0.1 ..."
```

Locally on one GPU or CPU, use `--backend local` with `audiolab.train` in place of `audiolab.remote`. Set the API key with `ANTHROPIC_API_KEY` or in `~/.config/anthropic/api_key`. The notebook `notebooks/colab.ipynb` runs the first baseline round on Colab.

## Limitations

- **One dataset and one training recipe.** All runs use 10k steps; the larger-context models may be undertrained.
- **rtf is measured on one thread of a cloud CPU.** That is not the target device, and identical configs varied by about ±10%. The 16 ms winner sits at rtf 0.089–0.097, close to the limit.
- **PESQ, STOI and SI-SDR are objective proxies.** There was no listening test.
