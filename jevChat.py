#!/usr/bin/env python3
"""Experiment: construct a short answer with Jev one character at a time.

Jev is stateless, so every request includes both the original question and all
characters selected so far.  This intentionally uses Jev as a generator-like
loop, even though its Choice primitive is normally better suited to selecting
among complete candidates.

Usage:
    python codex-jev-chat-completion.py "Capital of India"

The program prints an answer only after Jev chooses ``None`` *confidently*. If
20 character choices are made without that terminator, or a low-confidence
``None`` is received, it exits with a non-zero status rather than presenting a
truncated or dubious string as a complete answer.
"""

from __future__ import annotations

import argparse
import os
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parent / ".env")

API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
MAX_STEPS = 20
TERMINATOR = "None"
DEFAULT_MIN_TERMINATOR_CONFIDENCE = 0.8
CHARACTER_CHOICES = tuple(string.ascii_uppercase) + (" ", TERMINATOR)

ChoiceRequester = Callable[[dict[str, Any]], dict[str, Any]]


class JevResponseError(RuntimeError):
    """Raised when the API response cannot be used as a character decision."""


@dataclass(frozen=True)
class CharacterDecision:
    """One classified character choice returned by Jev."""

    choice: str
    confidence: float | None
    probabilities: dict[str, float]


@dataclass(frozen=True)
class CompletionResult:
    """Result of the bounded character-completion experiment."""

    answer: str
    completed: bool
    stop_reason: str
    decisions: tuple[CharacterDecision, ...]


def build_payload(question: str, current_answer: str, *, model: str = MODEL) -> dict[str, Any]:
    """Build the stateless Jev Choice request for one next-character decision."""
    criteria = {
        character: f"Append {character!r} as the next character of the answer."
        for character in string.ascii_uppercase
    }
    criteria[" "] = "Append one space as the next character of the answer."
    criteria[TERMINATOR] = "The answer is complete; append no more characters."

    # Preserve the deliberately literal wording used by this experiment. It is
    # useful when the answer is already present in the question (for example,
    # "Spell ... PONDICHERRY"), even though it cannot make Jev a text generator.
    prompt = (
        f"{question} ; fill up the next character of the current answer. "
        f'If the answer is completed, then return "{TERMINATOR}".\n'
        f"currentAnswer: {current_answer!r}"
    )
    return {
        "model": model,
        # Keep the complete state on every request: Jev does not retain prior calls.
        "state": {"question": question, "currentAnswer": current_answer},
        "questions": {
            "next_character": {
                "type": "choice",
                "instructions": prompt,
                "criteria": criteria,
            }
        },
    }


def _headers() -> dict[str, str]:
    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise RuntimeError("API_KEY is not set. Add it to .env or the environment.")
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def request_choice(payload: dict[str, Any]) -> dict[str, Any]:
    """Send one request to Jev and return its Choice answer object."""
    response = requests.post(API_URL, headers=_headers(), json=payload, timeout=60)
    response.raise_for_status()
    try:
        return response.json()["answers"]["next_character"]
    except (KeyError, TypeError, ValueError) as error:
        raise JevResponseError("Jev response did not contain answers.next_character") from error


def _parse_decision(answer: dict[str, Any]) -> CharacterDecision:
    try:
        choice = answer["choice"]
    except (KeyError, TypeError) as error:
        raise JevResponseError("Jev Choice response did not contain a choice") from error

    if choice not in CHARACTER_CHOICES:
        raise JevResponseError(f"Jev returned an unsupported choice: {choice!r}")

    confidence = answer.get("confidence")
    if confidence is not None and not isinstance(confidence, (int, float)):
        raise JevResponseError("Jev Choice confidence was not numeric")

    probabilities = answer.get("probabilities", {})
    if not isinstance(probabilities, dict):
        raise JevResponseError("Jev Choice probabilities were not an object")

    return CharacterDecision(
        choice=choice,
        confidence=float(confidence) if confidence is not None else None,
        probabilities=probabilities,
    )


def complete_answer(
    question: str,
    *,
    max_steps: int = MAX_STEPS,
    model: str = MODEL,
    min_terminator_confidence: float = DEFAULT_MIN_TERMINATOR_CONFIDENCE,
    requester: ChoiceRequester = request_choice,
) -> CompletionResult:
    """Repeatedly ask Jev for the next character until ``None`` or the cap.

    ``requester`` is injectable so the control flow can be tested without making
    API calls. ``max_steps`` is deliberately capped at 20 for this experiment.
    A low-confidence ``None`` is rejected because an early terminator otherwise
    converts a bad partial sequence into a false successful answer.
    """
    if not question.strip():
        raise ValueError("question must not be empty")
    if not 1 <= max_steps <= MAX_STEPS:
        raise ValueError(f"max_steps must be between 1 and {MAX_STEPS}")
    if not 0 <= min_terminator_confidence <= 1:
        raise ValueError("min_terminator_confidence must be between 0 and 1")

    current_answer = ""
    decisions: list[CharacterDecision] = []
    for _ in range(max_steps):
        payload = build_payload(question, current_answer, model=model)
        decision = _parse_decision(requester(payload))
        decisions.append(decision)

        if decision.choice == TERMINATOR:
            if (
                decision.confidence is not None
                and decision.confidence < min_terminator_confidence
            ):
                return CompletionResult(
                    answer=current_answer,
                    completed=False,
                    stop_reason="uncertain_terminator",
                    decisions=tuple(decisions),
                )
            return CompletionResult(
                answer=current_answer,
                completed=True,
                stop_reason="terminator",
                decisions=tuple(decisions),
            )

        current_answer += decision.choice

    return CompletionResult(
        answer=current_answer,
        completed=False,
        stop_reason="max_steps",
        decisions=tuple(decisions),
    )


def _print_trace(result: CompletionResult) -> None:
    """Print per-call observations without altering the final stdout answer."""
    prefix = ""
    for index, decision in enumerate(result.decisions, start=1):
        print(
            f"{index:02d} prefix={prefix!r} choice={decision.choice!r} "
            f"confidence={decision.confidence!r}",
            file=sys.stderr,
        )
        if decision.choice != TERMINATOR:
            prefix += decision.choice


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question for Jev to answer")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help=f"Maximum character requests, from 1 to {MAX_STEPS} (default: {MAX_STEPS})",
    )
    parser.add_argument(
        "--min-terminator-confidence",
        type=float,
        default=DEFAULT_MIN_TERMINATOR_CONFIDENCE,
        help=(
            "Reject a `None` result below this confidence to avoid accepting an "
            f"early stop (default: {DEFAULT_MIN_TERMINATOR_CONFIDENCE})"
        ),
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print each prefix, selected character, and confidence to stderr",
    )
    args = parser.parse_args(argv)

    result = complete_answer(
        args.question,
        max_steps=args.max_steps,
        min_terminator_confidence=args.min_terminator_confidence,
    )
    if args.trace:
        _print_trace(result)
    if result.completed:
        print(result.answer)
        return 0

    if result.stop_reason == "uncertain_terminator":
        confidence = result.decisions[-1].confidence
        print(
            f"Rejected {TERMINATOR!r} at confidence {confidence!r}; "
            f"partial answer was {result.answer!r}.",
            file=sys.stderr,
        )
    else:
        print(
            f"No {TERMINATOR!r} received within {args.max_steps} character requests; "
            f"partial answer was {result.answer!r}.",
            file=sys.stderr,
        )
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))