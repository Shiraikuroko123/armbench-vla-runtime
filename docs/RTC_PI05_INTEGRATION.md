# RTC-guided pi0.5 integration

## Status

Stage B0 is implemented as a tested reference contract, not as a completed
pi0.5 RTC result. `integrations/openpi/realtime_chunking.py` now reproduces the
public RTC evaluator's overlap scheduler and prefix-attention weights from
official commit `9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b`. It also supplies a
hard projected flow-inpainting reference under OpenPI's reverse-time
convention. No current evidence artifact used these functions.

## The scheduler difference that matters

For action horizon `H=10`, fixed execute horizon `E=5`, and inference delay
`d=4`, the existing measured-age dispatcher and RTC do not advance the same
number of simulation steps:

```text
current suffix selection:
  catch up for d steps, then execute new[d:d+E]  -> d+E total steps

RTC overlap scheduling:
  execute old[0:d], then execute new[d:E]        -> E total steps
```

Afterward RTC shifts `new[E:H]` to the front of a zero-padded `H`-step
reference chunk. A same-checkpoint RTC comparison therefore needs a new
overlap evaluator; relabeling the existing suffix-selection result would be
methodologically wrong.

## Model-side target

The pinned OpenPI snapshot is
`15a9616a00943ada6c20a0f158e3adb39df2ccac`. Its pi0.5 sampler is in
`src/openpi/models/pi0.py::Pi0.sample_actions` and integrates from `t=1` noise
to `t=0` actions. `pi05_libero` uses model-space shape `10 x 32`, while LIBERO
exposes raw `10 x 7` actions after output transforms.

The next extension must carry raw reference actions through the normal LIBERO
input transforms so normalization and 25 padded dimensions exactly match
training. It must then pass model-space reference actions, a hard committed
prefix mask, and RTC guidance weights into the flow sampler. The server must
attest the OpenPI extension commit and hash every condition/mask used in scored
queries.

Two sampler methods must remain distinctly named:

- `projected_flow_inpainting`: project hard committed slots onto
  `x_t = t * noise + (1 - t) * condition` before and after each Euler step;
- `rtc_pseudoinverse_guidance`: use the RTC denoised-action VJP correction and
  time-dependent guidance weight.

The first is a low-cost ablation and can guarantee zero final residual on hard
slots. It is not the RTC paper's pseudoinverse guidance method.

## Acceptance gates before rollouts

1. An empty condition and zero RTC weights are numerically identical to the
   official sampler under the same explicit noise.
2. Raw `10 x 7` reference actions become normalized/padded `10 x 32` model
   actions; padded dimensions are zero.
3. Hard projected slots have final residual below `1e-6`.
4. Nonfinite values, wrong dimensions, nonprefix hard masks, `d > E`, and
   `E > H` fail before JIT or policy inference.
5. An indexed fake policy proves exact execution order
   `old[:d] + new[d:E]` and a fixed `E` environment steps per query.
6. Peak GPU memory, warm inference P50/P95, and conditioning residual pass a
   20-query gate before any task-success pilot.

## Proposed evidence ladder

| Stage | Matrix | Purpose |
| --- | --- | --- |
| G0 | 20 warm queries per sampler method | Compilation, memory, latency, residual, and no-condition parity |
| Pilot | 10 tasks x 2 states x 3 methods at `d=4` | Detect integration failures without an efficacy claim |
| Primary | 10 tasks x 5 states x 3 noise keys x 2 methods | Suffix selection versus RTC-style continuation |
| Secondary | 10 tasks x 5 states x `d in {0,2}` x 2 methods | Delay sensitivity |
| Ablation | 10 tasks x 5 states with projected inpainting | Separate hard projection from RTC guidance |

The primary comparison should use one prespecified exact paired test. Task
clusters, condition order, motion discontinuity, acceleration jump, gripper
mismatch, condition residual, inference latency, and deadline misses remain
prespecified secondary analyses.

## Remaining gap

This reference layer closes an implementation ambiguity, not the publication
gap. A credible Stage B result still requires a clean OpenPI extension commit,
policy-internal JAX execution on the official checkpoint, an overlap scheduler
inside closed-loop LIBERO, fresh frozen protocols, and new evidence. True
measured-wall RTC later also needs independently ticking inference and control;
using response arrival time to condition the same completed inference would be
causally impossible.
