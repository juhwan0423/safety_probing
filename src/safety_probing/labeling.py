from __future__ import annotations

from typing import Any

from .config import LabelingConfig


OBSERVED_TIP = "observed_tip"
CONFIRMED_RECOVERED = "confirmed_recovered"
RIGHT_CENSORED = "right_censored"
SAFETY_OUTCOMES = (OBSERVED_TIP, CONFIRMED_RECOVERED, RIGHT_CENSORED)


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _event_candidates(
    source_run: dict[str, Any], dense_summary: dict[str, Any]
) -> list[tuple[float | None, str, str]]:
    candidates: list[tuple[float | None, str, str]] = []
    if bool(source_run.get("tip_over")):
        candidates.append(
            (
                _number(source_run.get("failure_time")),
                str(source_run.get("tip_over_reason") or "source_tip_over"),
                "source_metadata",
            )
        )

    roll_time = _number(dense_summary.get("posthoc_roll_tip_time_s"))
    if roll_time is not None:
        candidates.append((roll_time, "posthoc_roll_threshold", "dense_log"))

    support_time = _number(dense_summary.get("posthoc_support_loss_time_s"))
    if support_time is not None:
        candidates.append(
            (support_time, "posthoc_sustained_support_loss", "dense_log")
        )
    return candidates


def _earliest_event(
    candidates: list[tuple[float | None, str, str]],
) -> tuple[float | None, str, str]:
    timed = [candidate for candidate in candidates if candidate[0] is not None]
    if timed:
        return min(timed, key=lambda candidate: float(candidate[0]))
    return candidates[0]


def derive_safety_label(
    source_run: dict[str, Any],
    dense_summary: dict[str, Any],
    config: LabelingConfig,
) -> dict[str, Any]:
    """Create a censor-aware label without modifying simulator metadata."""

    candidates = _event_candidates(source_run, dense_summary)
    source_tip_over = bool(source_run.get("tip_over"))
    terminal_abs_roll = _number(dense_summary.get("terminal_abs_roll_deg"))
    terminal_abs_roll_rate = _number(
        dense_summary.get("terminal_abs_roll_rate_deg_s")
    )
    observation_end = _number(dense_summary.get("last_sim_time_s"))

    if candidates:
        event_time, event_reason, event_source = _earliest_event(candidates)
        outcome = OBSERVED_TIP
        event_observed = True
        binary_tip_label: int | None = 1
        censor_time = None
    else:
        terminal_unresolved = bool(
            terminal_abs_roll is not None
            and terminal_abs_roll_rate is not None
            and terminal_abs_roll >= config.right_censor_min_abs_roll_deg
            and terminal_abs_roll_rate >= config.right_censor_min_abs_roll_rate_deg_s
        )
        if terminal_unresolved:
            outcome = RIGHT_CENSORED
            binary_tip_label = None
            censor_time = observation_end
        else:
            outcome = CONFIRMED_RECOVERED
            binary_tip_label = 0
            censor_time = None
        event_time = None
        event_reason = "terminal_dynamics_unresolved" if terminal_unresolved else "recovered"
        event_source = "dense_log"
        event_observed = False

    future_eligible = bool(
        outcome != OBSERVED_TIP
        or event_time is None
        or event_time > config.observation_end_s
    )
    future_tip_label = binary_tip_label if future_eligible else None
    followup_time = event_time if event_time is not None else observation_end

    return {
        "label_schema_version": config.schema_version,
        "safety_outcome": outcome,
        "event_observed": event_observed,
        "binary_tip_label": binary_tip_label,
        "event_time_s": event_time,
        "event_reason": event_reason,
        "event_source": event_source,
        "censor_time_s": censor_time,
        "followup_time_s": followup_time,
        "source_tip_over": source_tip_over,
        "source_failure_time_s": _number(source_run.get("failure_time")),
        "source_tip_over_reason": source_run.get("tip_over_reason"),
        "upgraded_from_source": bool(event_observed and not source_tip_over),
        "future_prediction_eligible": future_eligible and outcome != RIGHT_CENSORED,
        "future_tip_label": (
            future_tip_label if outcome != RIGHT_CENSORED else None
        ),
        "observation_end_s": config.observation_end_s,
        "terminal_abs_roll_deg": terminal_abs_roll,
        "terminal_abs_roll_rate_deg_s": terminal_abs_roll_rate,
        "longest_support_loss_streak_steps": dense_summary.get(
            "longest_support_loss_streak_steps"
        ),
        "terminal_support_loss_streak_steps": dense_summary.get(
            "terminal_support_loss_streak_steps"
        ),
    }


def safety_transition(outcome_a: str, outcome_b: str) -> str:
    if outcome_a == RIGHT_CENSORED or outcome_b == RIGHT_CENSORED:
        return f"{outcome_a}_to_{outcome_b}"
    if outcome_a == OBSERVED_TIP and outcome_b == CONFIRMED_RECOVERED:
        return "A_tip_to_B_recovered"
    if outcome_a == CONFIRMED_RECOVERED and outcome_b == OBSERVED_TIP:
        return "A_recovered_to_B_tip"
    if outcome_a == OBSERVED_TIP:
        return "both_tip"
    return "both_recovered"


def modernized_quality(
    *,
    outcome_a: str,
    outcome_b: str,
    same_action: bool,
    max_roll_deg_delta: float | None,
    max_pitch_deg_delta: float | None,
) -> dict[str, Any]:
    if not same_action:
        return {
            "tier": "reject",
            "accepted": False,
            "reason": "Action trajectory changes between runs.",
        }
    if RIGHT_CENSORED in {outcome_a, outcome_b}:
        return {
            "tier": "censored",
            "accepted": False,
            "reason": "At least one endpoint has unresolved terminal dynamics.",
        }
    if outcome_a != outcome_b:
        return {
            "tier": "gold",
            "accepted": True,
            "reason": "Same-action pair with a censor-aware binary safety transition.",
        }

    deltas = [
        abs(value)
        for value in (max_roll_deg_delta, max_pitch_deg_delta)
        if value is not None
    ]
    margin = max(deltas, default=0.0)
    if margin >= 5.0:
        return {
            "tier": "silver",
            "accepted": True,
            "reason": "Same-action pair with a large stability consequence change.",
        }
    if margin >= 2.0:
        return {
            "tier": "bronze",
            "accepted": True,
            "reason": "Same-action pair with a measurable stability consequence change.",
        }
    return {
        "tier": "reject",
        "accepted": False,
        "reason": "Consequence delta is too small for a useful causal contrast.",
    }
