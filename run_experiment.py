"""
Essay Grading Study — runner.

Three conditions are tested:
  A. prompt_only       — grader is told to use ATLAS criteria, no rubric in context
  B. prompt_plus_simple — short rubric (one or two sentences per dimension)
  C. prompt_plus_complex — full ATLAS-style rubric with descriptors per score point

Each condition is run N_TRIALS times against the SAME essay. Temperature is held
above 0 so we measure model variance rather than just deterministic output. The
grader is asked to return JSON so scores parse cleanly.

Usage:
    python3 run_experiment.py
    python3 run_experiment.py --trials 30 --model gpt-4o
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def load_text(name: str) -> str:
    return (ARTIFACTS / name).read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = (
    "You are an experienced middle-school English Language Arts teacher scoring "
    "a student essay for the Arkansas Teaching and Learning Assessment System "
    "(ATLAS). Score the essay on three domains: Purpose/Focus/Organization (0-4), "
    "Evidence/Elaboration (0-4), and Conventions of Standard English (0-2). "
    "Return ONLY a single JSON object with this exact shape and nothing else:\n"
    '{"purpose_focus_organization": <int 0-4>, '
    '"evidence_elaboration": <int 0-4>, '
    '"conventions": <int 0-2>, '
    '"justification": "<one or two sentences>"}'
)


def build_user_message(condition: str, prompt: str, essay: str,
                       rubric_simple: str, rubric_complex: str,
                       rubric_complex_strict: str) -> str:
    if condition == "prompt_only":
        return (
            "Score this Grade 8 argumentative essay using the ATLAS rubric. "
            "Use the standard ATLAS three-domain structure (Purpose/Focus/"
            "Organization 0-4, Evidence/Elaboration 0-4, Conventions 0-2).\n\n"
            f"=== WRITING PROMPT ===\n{prompt}\n\n"
            f"=== ESSAY ===\n{essay}\n"
        )
    if condition == "prompt_plus_simple":
        return (
            "Score this Grade 8 argumentative essay using the rubric below.\n\n"
            f"=== RUBRIC ===\n{rubric_simple}\n\n"
            f"=== WRITING PROMPT ===\n{prompt}\n\n"
            f"=== ESSAY ===\n{essay}\n"
        )
    if condition == "prompt_plus_complex":
        return (
            "Score this Grade 8 argumentative essay using the detailed rubric below. "
            "Apply each score-point descriptor carefully.\n\n"
            f"=== RUBRIC ===\n{rubric_complex}\n\n"
            f"=== WRITING PROMPT ===\n{prompt}\n\n"
            f"=== ESSAY ===\n{essay}\n"
        )
    if condition == "prompt_plus_complex_strict":
        return (
            "Score this Grade 8 argumentative essay using the detailed rubric below. "
            "Apply each score-point descriptor carefully.\n\n"
            f"=== RUBRIC ===\n{rubric_complex_strict}\n\n"
            f"=== WRITING PROMPT ===\n{prompt}\n\n"
            f"=== ESSAY ===\n{essay}\n"
        )
    raise ValueError(f"unknown condition: {condition}")


# ---------------------------------------------------------------------------
# Trial execution
# ---------------------------------------------------------------------------

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Trial:
    condition: str
    trial_index: int
    purpose_focus_organization: int | None
    evidence_elaboration: int | None
    conventions: int | None
    total: int | None
    justification: str
    raw_response: str
    parse_error: str | None
    latency_seconds: float


def parse_response(raw: str) -> tuple[dict | None, str | None]:
    match = JSON_BLOCK_RE.search(raw)
    if not match:
        return None, "no JSON object found in response"
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return None, f"json decode failed: {exc}"
    required = {"purpose_focus_organization", "evidence_elaboration", "conventions"}
    if not required.issubset(data):
        return None, f"missing keys; got {sorted(data)}"
    return data, None


def run_trial(client: OpenAI, model: str, temperature: float | None,
              condition: str, trial_index: int, user_message: str) -> Trial:
    started = time.time()
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_message},
        ],
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    completion = client.chat.completions.create(**kwargs)
    latency = time.time() - started
    raw = completion.choices[0].message.content or ""
    data, err = parse_response(raw)
    if data is None:
        return Trial(
            condition=condition,
            trial_index=trial_index,
            purpose_focus_organization=None,
            evidence_elaboration=None,
            conventions=None,
            total=None,
            justification="",
            raw_response=raw,
            parse_error=err,
            latency_seconds=latency,
        )
    pfo = int(data["purpose_focus_organization"])
    ee = int(data["evidence_elaboration"])
    conv = int(data["conventions"])
    return Trial(
        condition=condition,
        trial_index=trial_index,
        purpose_focus_organization=pfo,
        evidence_elaboration=ee,
        conventions=conv,
        total=pfo + ee + conv,
        justification=str(data.get("justification", "")),
        raw_response=raw,
        parse_error=None,
        latency_seconds=latency,
    )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def summarize(trials: list[Trial]) -> dict:
    by_condition: dict[str, dict] = {}
    for cond in {t.condition for t in trials}:
        rows = [t for t in trials if t.condition == cond and t.total is not None]
        totals = [t.total for t in rows]
        pfos = [t.purpose_focus_organization for t in rows]
        ees = [t.evidence_elaboration for t in rows]
        convs = [t.conventions for t in rows]
        by_condition[cond] = {
            "n_valid": len(rows),
            "n_failed_parse": sum(1 for t in trials if t.condition == cond and t.total is None),
            "total": _stats(totals),
            "purpose_focus_organization": _stats(pfos),
            "evidence_elaboration": _stats(ees),
            "conventions": _stats(convs),
        }
    return by_condition


def _stats(xs: list[int]) -> dict:
    if not xs:
        return {"mean": None, "stdev": None, "min": None, "max": None, "range": None}
    mean = statistics.mean(xs)
    stdev = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return {
        "mean": round(mean, 3),
        "stdev": round(stdev, 3),
        "min": min(xs),
        "max": max(xs),
        "range": max(xs) - min(xs),
        "values": xs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20,
                        help="trials per condition (default 20)")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="OpenAI chat model (default gpt-4o)")
    parser.add_argument("--temperature", type=str, default="default",
                        help="sampling temperature; pass a float (e.g. 0.7) "
                             "or the string 'default' to use the model's API default")
    parser.add_argument("--essay", type=str, default="essay_strong.txt",
                        help="essay file in artifacts/ (default essay_strong.txt)")
    parser.add_argument("--conditions", type=str,
                        default="prompt_only,prompt_plus_simple,prompt_plus_complex",
                        help="comma-separated list of conditions to run; "
                             "valid: prompt_only, prompt_plus_simple, "
                             "prompt_plus_complex, prompt_plus_complex_strict")
    parser.add_argument("--out", type=str, default="results/run.json")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set (.env not loaded?)", file=sys.stderr)
        return 2

    client = OpenAI()

    if args.temperature == "default":
        temperature: float | None = None
        temperature_label: str | float = "default"
    else:
        temperature = float(args.temperature)
        temperature_label = temperature

    prompt = load_text("prompt.txt")
    essay = load_text(args.essay)
    rubric_simple = load_text("rubric_simple.txt")
    rubric_complex = load_text("rubric_complex.txt")
    rubric_complex_strict = load_text("rubric_complex_strict.txt")

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    valid = {"prompt_only", "prompt_plus_simple", "prompt_plus_complex",
             "prompt_plus_complex_strict"}
    bad = set(conditions) - valid
    if bad:
        print(f"ERROR: unknown conditions {sorted(bad)}", file=sys.stderr)
        return 2
    trials: list[Trial] = []

    total_calls = args.trials * len(conditions)
    call_index = 0
    print(f"Running {args.trials} trials × {len(conditions)} conditions = "
          f"{total_calls} calls (model={args.model}, temperature={temperature_label})")

    for condition in conditions:
        user_message = build_user_message(
            condition, prompt, essay, rubric_simple, rubric_complex,
            rubric_complex_strict,
        )
        for i in range(args.trials):
            call_index += 1
            try:
                trial = run_trial(
                    client, args.model, temperature,
                    condition, i, user_message,
                )
            except Exception as exc:  # network / API issues
                trial = Trial(
                    condition=condition, trial_index=i,
                    purpose_focus_organization=None,
                    evidence_elaboration=None,
                    conventions=None, total=None,
                    justification="", raw_response="",
                    parse_error=f"api error: {exc}",
                    latency_seconds=0.0,
                )
            trials.append(trial)
            score = trial.total if trial.total is not None else "FAIL"
            print(f"  [{call_index:>3}/{total_calls}] {condition:>22s} #{i:>2} → {score}")

    summary = summarize(trials)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for cond in conditions:
        s = summary.get(cond)
        if not s:
            continue
        t = s["total"]
        print(f"{cond:>30s}  n={s['n_valid']}  "
              f"mean={t['mean']}  stdev={t['stdev']}  "
              f"range={t['min']}-{t['max']} (Δ{t['range']})")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config": {
            "model": args.model,
            "temperature": temperature_label,
            "trials_per_condition": args.trials,
            "essay_file": args.essay,
        },
        "summary": summary,
        "trials": [asdict(t) for t in trials],
    }, indent=2), encoding="utf-8")

    print(f"\nFull results written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
