# pi0.5 projected-conditioning G0

This artifact is the pre-rollout integration gate for policy-internal projected
flow conditioning. It uses the official `pi05_libero` checkpoint through a
clean OpenPI extension commit and fixed explicit sampling noise.

## Result

- official no-condition path was exactly repeatable across 20 warm queries;
- an all-false condition mask was bitwise identical to the official path;
- four committed raw action steps had model-space and raw-space residual `0.0`;
- baseline model P95 was `54.919 ms` and conditioned model P95 was `54.527 ms`;
- conditioned/baseline model P95 ratio was `0.9929`, below the frozen `1.25` gate;
- peak JAX device memory was `6,906,790,912` bytes, below the frozen `23 GiB` gate.

The checkpoint contains 16 files totaling 12,439,085,481 bytes, with content
manifest SHA-256
`9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5`.

## Scope

This gate proves model loading, transform routing, JAX compilation, legacy
parity, hard-prefix enforcement, latency, and memory feasibility. It is not a
task-success experiment and does not establish RTC efficacy. The next evidence
stage must compare overlap scheduling in closed-loop LIBERO rollouts.

Validate from the project root:

```powershell
& '..\.venv\Scripts\python.exe' -m integrations.openpi.validate_projected_conditioning_g0 `
  evidence\pi05_projected_conditioning_g0_001
```
