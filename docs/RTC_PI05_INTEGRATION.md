# RTC-guided pi0.5 integration

## Status

ArmBench now contains a policy-internal RTC-style VJP guidance path for the
official `pi05_libero` checkpoint. The integration changes the pinned OpenPI
flow sampler without fine-tuning pi0.5. A fixed-observation G0 gate passed, and
a separate LIBERO-10 three-method pilot completed 60/60 closed-loop rollouts.

These stages establish sampler integration, correction direction, latency,
memory, auditability, and a small simulation pilot. They do not establish full
equivalence to the RTC paper, task-success improvement, independently ticking
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
shared by all compared methods. Relabeling the earlier suffix-selection result
as RTC would therefore be methodologically wrong.

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

## Completed three-method pilot

The exploratory pilot used all ten LIBERO-10 tasks, initial-state indices
`0,1`, and all three methods: 20 matched triplets and 60 rollouts at
`H=10`, `E=5`, `d=4`. Method order used a three-way Latin rotation, and every
triplet shared explicit keyed pi0.5 sampling noise. The raw artifact contains
60 manifest-protected H.264 videos and a float32 transition transcript.

| Method | Success | Episode-equal motion seam | Episode-equal gripper seam | Diagnostic |
| --- | ---: | ---: | ---: | ---: |
| Unconditioned overlap | 20/20 | 0.104452 | 0.058052 | - |
| Hard projected overlap | 19/20 | 0.083129 | 0.045345 | max hard residual 0 |
| RTC-guided overlap | 19/20 | 0.084002 | 0.039617 | weighted model RMSE 0.024735 |

Both conditioned methods had `0/1/19` wins/losses/ties against unconditioned
overlap, with raw exact McNemar `p=1.0`. The pilot therefore does not support a
task-success improvement. The seam values above come from the independent
report, which aggregates query-level records first within episode. Task-block
95% motion-seam intervals exclude zero in the negative direction for both
methods, while the gripper intervals cross zero. Seam remains an exploratory
process metric.

Validate the immutable raw artifact locally:

```powershell
& '..\.venv\Scripts\python.exe' -m integrations.openpi.rtc_overlap_pilot validate `
  evidence\pi05_rtc_overlap_pilot_001\evaluation
```

## Evidence ladder

| Stage | Matrix | Status and purpose |
| --- | --- | --- |
| G0 | 20 fixed-observation warm queries | Passed: parity, direction, latency, memory, determinism |
| Pilot | 10 tasks x 2 states x 3 methods | Completed: integration pilot, no efficacy claim |
| Held-out primary | 10 tasks x 5 states x 2 noise seeds x 3 methods | 300-rollout protocol frozen separately; outcome pending |
| Runtime extension | independent inference/control clocks | Pending |
| Generalization | second VLA and hardware | Pending |

The held-out protocol is
[`RTC_OVERLAP_PRIMARY_300_PROTOCOL.md`](research/RTC_OVERLAP_PRIMARY_300_PROTOCOL.md).
The 60 pilot rollouts are excluded from its 300-rollout matrix.

## Remaining gap

The implementation now reaches inside a real pi0.5 flow sampler and has
checkpoint-backed closed-loop evidence. The remaining publication gap is no
longer "implement VJP guidance." It is to establish a powered held-out effect,
then test causal asynchronous execution with independently ticking inference
and control, a second policy family, broader perturbations, and hardware.

The present LIBERO evaluator blocks on each policy response and injects a fixed
four-step overlap delay. It is suitable for matched method analysis, but it is
not a hard real-time system or a deadline guarantee.
