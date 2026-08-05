# RTC-guided pi0.5 integration

## Status

ArmBench now contains a policy-internal RTC-style VJP guidance path for the
official `pi05_libero` checkpoint. The integration changes the pinned OpenPI
flow sampler without fine-tuning pi0.5. A fixed-observation G0 gate passed. The
first three-method v2 execution was later rejected by a pairing audit; a
corrected-v3 held-out matrix has now completed 300/300 closed-loop rollouts.

These stages establish sampler integration, correction direction, latency,
memory, auditable remediation, and a same-checkpoint simulation comparison.
Corrected-v3 does not establish task-success improvement. The project also does
not establish full equivalence to the RTC paper, independently ticking
inference and control, cross-model generality, or real-robot performance.

## Scheduler contract

For action horizon `H=10`, fixed execute horizon `E=5`, and inference delay
`d=4`, suffix selection and RTC overlap do not advance the same number of
simulation steps:

```text
measured-age suffix selection:
  catch up for d steps, then execute new[d:d+E]  -> d+E total steps

RTC overlap scheduling:
  execute old[0:d], then execute new[d:E]        -> E total steps
```

After each overlap query, ArmBench shifts `new[E:H]` to the front of a
zero-padded `H`-step reference. Query zero only samples an unexecuted reference;
query one uses the same initial observation. This reference-only bootstrap is
shared by all compared methods. Treating the earlier suffix-selection result as
RTC would conflate two distinct scheduler contracts.

## Reverse-time OpenPI mapping

The public RTC reference integrates from noise to action in the opposite time
direction from the pinned OpenPI sampler. With OpenPI time `t=1` at noise and
`t=0` at action, the implemented denoised-action estimate and guidance update
are:

```text
D_t(x) = x - t * v_openpi(x, t)
error = weights * (reference - D_t(x))
correction = J_D(x)^T * error
v_guided = v_openpi - gain(t) * correction

gain(t) = min((t^2 + (1-t)^2) / (t(1-t)), 5)
```

The correction uses a minus sign because OpenPI's Euler step has `dt < 0`.
Endpoint gain is explicitly capped, avoiding division by zero at `t=1`.
`jax.vjp` computes the required transpose-Jacobian product over the batched
`1 x 10 x 32` model action. A JVP would compute a different operation.

Raw LIBERO references follow the policy's normal input transforms:
`10 x 7` raw actions are normalized and padded to `10 x 32`; dimensions 7-31
remain zero. The server derives guidance weights from the attested `H/E/d`
contract and rejects hard projection combined with soft guidance.

The three evaluator methods remain distinct:

- `overlap_unconditioned`: same scheduler and reference bootstrap, no sampler
  conditioning;
- `projected_overlap`: hard projection of committed slots before and after
  every Euler step;
- `rtc_guided_overlap`: soft RTC-style denoised-action VJP guidance.

Hard projection is an ablation with exact conditioned-slot residual. It is not
the RTC guidance method.

## Frozen identities

- RTC public reference: `9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b`
- OpenPI upstream base: `15a9616a00943ada6c20a0f158e3adb39df2ccac`
- OpenPI RTC extension: `54592c7148ba69bf52757385502782f80f2285e0`
- ArmBench G0 evidence commit: `7ed75a9`
- ArmBench three-method evaluator commit: `2aef062`
- Checkpoint content SHA-256:
  `9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5`

## Completed G0 gate

The preserved
[`pi05_rtc_guidance_g0_001`](../evidence/pi05_rtc_guidance_g0_001/README.md)
artifact uses one fixed observation, one explicit sampling-noise tensor, and
20 warm queries per measured path.

| Gate | Measured result |
| --- | ---: |
| Zero-weight legacy parity | bitwise exact |
| Baseline weighted model RMSE | 0.0814669 |
| Guided weighted model RMSE | 0.0270451 |
| Guided / baseline residual | 0.3320 |
| Baseline warm wall P95 | 79.5584 ms |
| Guided warm wall P95 | 108.0617 ms |
| Warm wall P95 ratio | 1.3583x |
| Peak JAX bytes in use | 6,906,793,216 (6.43 GiB) |

Baseline and guided outputs were exactly repeatable under explicit noise. G0
proves that the extension runs on the claimed checkpoint and moves the weighted
model residual in the intended direction within the frozen latency and memory
gates. It is not a closed-loop task result.

## Rejected v2 execution and corrected-v3 primary

The original v2 evaluator reused one LIBERO environment across method
conditions. A later invariant audit found 6/20 query-0 action mismatches in the
development artifact and 15/50 in each held-out seed attempt. Tasks 3, 8, and 9
retained different policy images across resets even though robot state, prompt,
sampling key, and sampling noise matched. The v2 artifacts remain immutable,
but their success and seam summaries are excluded from all method-effect
claims. See the
[pairing audit](research/RTC_OVERLAP_PAIRING_AUDIT_20260805.md).

Corrected-v3 constructs and closes a fresh environment for every rollout and
requires matching query-0 policy-input, response-action, sampling-key, and
sampling-noise hashes before writing a root manifest. The held-out matrix is
10 tasks x 5 initial states x 2 sampling seeds x 3 methods: 300 rollouts and
100 matched triplets.

| Method | Success | Motion seam mean | Gripper seam mean |
| --- | ---: | ---: | ---: |
| Unconditioned overlap | 96/100 | 0.106729 | 0.053754 |
| Hard projected overlap | 97/100 | 0.083089 | 0.055490 |
| RTC-guided overlap | 97/100 | 0.087204 | 0.043190 |

Each conditioned contrast has a `+1` percentage-point success difference and
raw/Holm exact McNemar `p=1.0`; no task-success improvement is supported. The
exploratory motion-seam differences are `-0.023640` for hard projection and
`-0.019524` for RTC guidance, with task-block intervals excluding zero. They
remain process metrics rather than efficacy or safety endpoints.

Validate and open the current evidence from any Windows directory:

```powershell
D:\arm-planning-control-project\project\scripts\rtc_primary_acceptance.cmd
```

Use `-NoOpen` for a noninteractive validation.

## Evidence ladder

| Stage | Matrix | Status and purpose |
| --- | --- | --- |
| G0 | 20 fixed-observation warm queries | Passed: parity, direction, latency, memory, determinism |
| v2 three-method attempt | 10 tasks x 2 states x 3 methods | Rejected: environment reuse broke query-0 pairing |
| Pairing remediation | tasks 3/8/9 smoke | Passed: fresh environments and four-hash query-0 gate |
| Corrected-v3 held-out primary | 10 tasks x 5 states x 2 noise seeds x 3 methods | Completed: 300 rollouts; no task-success superiority; exploratory seam decrease |
| Runtime extension | independent inference/control clocks | Pending |
| Generalization | second VLA and hardware | Pending |

The current protocol is
[`RTC_OVERLAP_PRIMARY_300_V3_PROTOCOL.md`](research/RTC_OVERLAP_PRIMARY_300_V3_PROTOCOL.md).
All v2 outcome rows are excluded from the corrected-v3 analysis.

## Remaining gap

The implementation now reaches inside a real pi0.5 flow sampler and has
checkpoint-backed closed-loop evidence. The 300-rollout held-out comparison did
not establish a task-success advantage. The remaining publication gap is no
longer "implement VJP guidance" or "run more of the same matrix." It is to test
causal asynchronous execution with independently ticking inference and control,
a second policy family, physical feasibility constraints, broader
perturbations, and hardware.

The present LIBERO evaluator blocks on each policy response and injects a fixed
four-step overlap delay. It is suitable for matched method analysis, but it is
not a hard real-time system or a deadline guarantee.
