# pi0.5 RTC guidance G0

This artifact is a 20-query feasibility gate on the official
`pi05_libero` checkpoint. It is not a task-success or publication claim.

The run used one fixed observation, one explicit `10 x 32` sampling-noise
array, a shifted `10 x 7` RTC reference chunk, the official exponential
prefix schedule for `H=10`, `E=5`, `d=4`, and maximum guidance weight 5.0.
The baseline, zero-weight, zero-maximum, and active VJP calls therefore share
the checkpoint, observation, and sampling noise.

Measured gates:

- zero-weight output was exactly equal to the legacy sampler output;
- baseline weighted model RMSE: `0.0814669346`;
- guided weighted model RMSE: `0.0270450617` (`33.20%` of baseline);
- 20-query guided wall-clock P95: `108.0617 ms` (`1.3583x` baseline);
- JAX peak bytes in use: `6,906,793,216` (`6.43 GiB`);
- baseline and guided outputs were exactly repeatable under explicit noise.

Validate without a GPU from the repository root:

```powershell
python -m integrations.openpi.validate_rtc_guidance_g0 `
  evidence/pi05_rtc_guidance_g0_001
```
