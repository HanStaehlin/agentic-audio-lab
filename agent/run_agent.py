"""Experiment agent: plans, launches, monitors, diagnoses and reports speech-enhancement runs.

    python -m agent.run_agent --goal "Beat the baseline PESQ with at most 6 runs" --baseline <run_id> --max-runs 6

Needs ANTHROPIC_API_KEY. Every launch asks for approval unless --auto-approve is set.
"""
import argparse
import json
import os
from pathlib import Path

import anthropic

from agent.tools import SCHEMAS, Session

MODEL = "claude-opus-5-5"

SYSTEM = """You run speech-enhancement experiments for an audio ML researcher.
The model is a small causal GRU that predicts a spectral mask; it is trained with an SI-SDR loss on
VoiceBank-DEMAND (16 kHz) and evaluated on the test set with PESQ (wideband), STOI and SI-SDR,
plus streaming latency per hop on CPU. It must stay real-time capable (rtf well below 1).

How to work:
- Start with list_runs. Form a hypothesis before each launch and state it in `reason`.
- Change one or two things per run so results stay interpretable. Only one run can be active.
- Use wait_for_run instead of polling. If a run fails or crashes, read its logs with get_run_status,
  diagnose the cause, and say what you conclude before launching anything else.
- Respect the run budget. If the human declines a launch, do not retry the same run.
- Judge runs with compare_runs against the baseline. Treat PESQ differences below ~0.03 as noise
  on a single seed, and weigh quality against latency.
- Finish by calling write_report, then give a short summary."""


def execute(session, name, args):
    fn = getattr(session, name, None)
    if fn is None or name.startswith("_"):
        raise ValueError(f"unknown tool {name}")
    return fn(**args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", required=True)
    ap.add_argument("--baseline", required=True, help="run_id of the finished baseline run")
    ap.add_argument("--max-runs", type=int, default=6)
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--auto-approve", action="store_true")
    ap.add_argument("--backend", choices=["local", "modal"], default="local", help="where runs execute")
    args = ap.parse_args()

    key_file = Path.home() / ".config" / "anthropic" / "api_key"  # outside the repo, never committed
    if not os.environ.get("ANTHROPIC_API_KEY") and key_file.exists():
        os.environ["ANTHROPIC_API_KEY"] = key_file.read_text().strip()
    client = anthropic.Anthropic()
    session = Session(args.max_runs, args.auto_approve, args.backend)
    messages = [{"role": "user", "content":
                 f"Goal: {args.goal}\nBaseline run: {args.baseline}\nRun budget: {args.max_runs} launches."}]

    for _ in range(args.max_turns):
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM,
            tools=SCHEMAS,
            messages=messages,
            output_config={"effort": "high"},
            # On a safety decline, let the API re-run the request on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        # Append the full content (thinking blocks included) so the history stays intact.
        messages.append({"role": "assistant", "content": response.content})
        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"\n[agent] {block.text}")

        if response.stop_reason == "refusal":
            print("\n[stopped] the model declined to continue.")
            return
        if response.stop_reason == "max_tokens":
            messages.append({"role": "user", "content": "You hit the output limit. Continue, more briefly."})
            continue
        if response.stop_reason == "pause_turn":
            continue
        if response.stop_reason != "tool_use":
            return  # end_turn: the agent is done

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"\n[tool] {block.name} {json.dumps(block.input)[:300]}")
            try:
                out = execute(session, block.name, block.input)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(out)})
            except Exception as exc:
                print(f"[tool error] {exc}")
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": f"{type(exc).__name__}: {exc}", "is_error": True})
        messages.append({"role": "user", "content": results})

    print(f"\n[stopped] reached --max-turns {args.max_turns}.")


if __name__ == "__main__":
    main()
