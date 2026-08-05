# Measured-age temporal alignment runtime

## Status

This module is implemented, unit-tested, and evaluated in two deliberately
separate official-checkpoint studies. The historical 40-rollout/20-pair pilot
is exploratory because it did not pair OpenPI policy-sampling RNG. The held-out
successor in `evidence/pi05_libero_measured_age_confirmatory_001` contains 120
matched pairs and explicitly pairs both response jitter and pi0.5 flow-sampling
noise. It improved success from 88/120 to 116/120 with exact McNemar
`p=1.941574737e-6`. The legacy `fixed_steps` path remains the default, so all
deterministic-delay evidence keeps its original meaning.

The implementation consists of:

- `integrations/openpi/deadline_alignment.py`: pure timing, suffix, deadline,
  and keyed-jitter decisions;
- `integrations/openpi/libero_runtime.py`: measured observation age, simulated
  controller catch-up, action selection, and bounded hold-refresh;
- `integrations/openpi/serve_policy_attested.py`: explicit keyed pi0.5
  flow-sampling noise and server-side request/hash attestation;
- `integrations/openpi/validate_measured_age_artifact.py`: independent timing,
  pairing, noise, source, matrix, and video recomputation;
- `integrations/openpi/measured_age_confirmatory_analysis.py`: prespecified
  primary and task-cluster robustness analyses;
- `tests/test_deadline_alignment.py`: timing boundary and jitter determinism;
- measured-wall cases in `tests/test_libero_runtime.py`: end-to-end runtime
  behavior with a fake monotonic clock.

## Problem

The frozen study knows the injected delay is four 50 ms control steps. Its
dispatcher therefore skips `actions[0:4]`. That is useful causal evidence but
not a deployment mechanism: a live runtime receives timestamps and a response,
not the experimenter's `latency_steps` label.

The measured path records time from observation capture until a response is
available:

```text
observation captured
  -> request construction
  -> websocket / model inference
  -> optional keyed response-delivery jitter
  -> response available
```

It then makes two distinct discrete calculations at a 50 ms controller period:

```text
completed controller ticks = floor(observation_age / 50 ms)
conservative action offset = ceil(observation_age / 50 ms)
```

The completed-tick count advances LIBERO under the last command to model what an
independent controller would have executed while inference was unavailable.
The conservative offset never selects an action slot whose nominal start time
has already passed. `floor` alignment is implemented as a pre-registerable
ablation; it must not be selected after seeing task outcomes.

## Decision rule

For a chunk with `H` actions, requested execution horizon `R`, measured age
`a`, controller period `dt`, and conservative offset `k = ceil(a / dt)`:

```text
execute actions[k:k+R] only when
  a <= deadline
  and k + R <= H
```

Otherwise the runtime preserves the gripper command, zeros Cartesian motion,
executes one explicit fallback-hold simulator tick, and requests a fresh chunk.
Consecutive refresh is bounded and resets after an accepted chunk; total
refreshes remain an episode metric. Once `max_age_refreshes` is exhausted, the
episode fails closed with a distinct deadline, horizon, or combined termination
reason. It never shortens the requested suffix silently and never falls back to
executing index zero.

## Paired jitter

`keyed_discrete_jitter_ms` derives each value from SHA-256 over:

```text
seed, task suite, task id, episode index, replan horizon, query index
```

The runtime mode is deliberately absent. Baseline and candidate therefore
receive the same query-index schedule even if they are executed in a different
order. The function does not consume NumPy's global RNG, so environment
sampling and condition order cannot shift the jitter stream.

The function only generates a value. A caller must record the pairing key and
candidate set, pass the value through `response_jitter_ms`, and preserve every
realized value in the query artifact.

## Minimal Python use

```python
from integrations.openpi.libero_runtime import (
    LATENCY_ALIGNED,
    MEASURED_WALL_LATENCY,
    RuntimeConfig,
    run_episode,
)

config = RuntimeConfig(
    mode=LATENCY_ALIGNED,
    replan_steps=5,
    latency_steps=0,
    max_task_steps=220,
    latency_source=MEASURED_WALL_LATENCY,
    control_period_ms=50.0,
    age_rounding="ceil",
    deadline_ms=250.0,
    max_age_refreshes=2,
)

result = run_episode(
    environment,
    policy_client,
    initial_state,
    task_description,
    config,
    response_jitter_ms=paired_jitter_for_query,
)
```

`latency_steps` must be zero in this path. Combining hidden fixed steps with a
measured estimate is rejected as an ambiguous protocol.

## Runtime diagnostics

Set breakpoints in this order:

1. `run_episode`, immediately before `policy.infer`, to inspect the observation
   timestamp and request.
2. `plan_alignment`, to inspect `observation_age_ms`, `stale_steps`, deadline,
   and suffix length.
3. The measured rejection branch in `run_episode`, to verify that an overrun
   changes `last_action` to hold and increments the refresh count.
4. The action slice, to verify the first executed indexed action equals the
   recorded `action_offset_steps`.

Every `QueryRecord` exposes:

- `latency_source`;
- `observation_age_ms` and `response_jitter_ms`;
- `measured_stale_steps` and `action_offset_steps`;
- `available_suffix_steps`;
- `deadline_exceeded` and `horizon_overrun`;
- `age_refresh_index`, `fallback_hold_steps`, and the explicit decision string.

Run the focused deterministic suite from the repository root:

```powershell
& '..\.venv\Scripts\python.exe' -m pytest -q `
  tests\test_deadline_alignment.py tests\test_libero_runtime.py
```

The indexed-action test is the fastest debugger: at 120 ms age, 50 ms period,
and ceil rounding, the runtime advances two completed controller ticks but
starts execution at action index three.

## Warm-up protocol

The preserved external evaluation has 8,431 query records. Normal client P95
latency is approximately 82-83 ms, while the maximum in each suite is about
18.5 seconds on the first recorded query, consistent with compilation or
warm-up overhead. A scored measured-latency study must
therefore perform and record an attested warm-up before assigning the first
paired condition. Otherwise condition order, not the runtime method, determines
which mode receives the compilation outlier.

Warm-up must use the same checkpoint and tensor shapes, validate the returned
action contract, and remain outside task success and policy-query outcomes. Its
latency and response validation still belong in provenance.

The registered pilot confirmed the order effect directly. Its three unscored
warm-up ages were 18,424.229 ms, 84.145 ms, and 79.465 ms. The subsequent 810
scored queries had per-mode inference-latency P95 values of 82.920 ms and
82.390 ms. Keeping the first compilation query out of the randomized modes was
therefore necessary, not cosmetic.

## Held-out confirmatory evidence

The successor protocol was frozen at
`12070625cd6f46186282317262065d015c8fbe27` before scored held-out episodes were
inspected. It uses all ten Spatial tasks, episode indices 5-16, 120 matched
pairs, counterbalanced condition order, and a mode-independent jitter/noise key
for every common query identity. The explicit float32 noise shape is `10 x 32`,
matching pi0.5-LIBERO's action horizon and action dimension.

The primary paired difference is +23.33 points (bootstrap 95%
[+15.00,+31.67]), with 32 aligned-only successes, 4 baseline-only successes,
84 ties, and exact two-sided McNemar `p=1.941574737e-6`. Prespecified
task-sensitivity checks also remain positive: whole-task bootstrap 95%
[+10.83,+38.33], exhaustive `2^10` task sign-flip `p=0.015625`, and
leave-one-task-out effects from +17.59 to +25.93 points. Condition-first
effects are +13.33 and +33.33 points, so order-magnitude heterogeneity must be
reported even though both strata are positive.

Run `scripts/measured_age_confirmatory_acceptance.cmd` to revalidate the
artifact, recompute the statistics, verify all 240 manifest-bound video files,
and open the paired dashboard.

## Limitations

- `policy.infer` remains a blocking call. LIBERO catch-up occurs after response
  arrival; this models a separately ticking controller but is not an OS-level
  real-time loop.
- A real deadline watchdog requires inference in a worker thread or process,
  controller ticks on an independent clock, cancellation or response discard,
  and explicit websocket reconnection after timeout.
- Time-only suffix selection assumes temporal consistency inside the returned
  chunk. RTC-style flow inpainting requires access inside policy sampling and
  cannot be implemented faithfully from completed actions alone.
- The official measured-age confirmation is a powered 120-pair matrix with
  paired policy noise, but it still covers one pi0.5 checkpoint family and one
  simulator. It does not establish cross-model generality.
- This layer checks temporal availability, not joint, acceleration, collision,
  or dynamics feasibility in the formal LIBERO path.

See `docs/research/VLA_TOP_VENUE_GAP_ANALYSIS_2026.md` for the direct comparison
with RTC, OpenVLA-OFT, DPPO, HIL-SERL, and recent asynchronous VLA preprints.
