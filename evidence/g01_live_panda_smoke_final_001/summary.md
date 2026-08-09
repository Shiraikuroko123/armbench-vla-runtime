# Attested pi0.5-LIBERO to Panda MuJoCo runtime smoke

This mechanism gate verifies that the official OpenPI `pi05_libero`
checkpoint can drive the ArmBench asynchronous Panda runtime without policy
fine-tuning. It is an integration and runtime-systems result, not an official
LIBERO task-success evaluation.

## Executed path

```text
MuJoCo Panda RGB/state observation
  -> attested OpenPI pi0.5-LIBERO checkpoint
  -> native Hx7 Cartesian/axis-angle action chunk
  -> explicit Hx8 Panda differential-IK adapter
  -> latest-only asynchronous worker and measured-age dispatcher
  -> braking-invariant repair
  -> torque-controlled MuJoCo physics
```

The server loaded 16 checkpoint objects totaling `12,439,085,481` bytes.
Their GCS CRC32C values matched the public object metadata, and the server's
canonical checkpoint inventory SHA-256 was
`9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5`.

## Result

| Metric | Value |
| --- | ---: |
| Live policy responses accepted | 35 |
| Mean / P95 / max policy latency | 82.75 / 89.56 / 112.42 ms |
| Control ticks | 311 |
| Control ticks while inference was active | 290 |
| P95 / max control lateness | 0.075 / 22.09 ms |
| Hold-boundary rate | 4.76% |
| Planned repair interventions | 15 steps |
| Collision / self-collision / joint-limit steps | 0 / 0 / 0 |
| Abrupt-stop / unsafe-plan violations | 0 / 0 |
| Decodable video frames | 94 at 640x480 |

The run used ArmBench commit
`2f92db28e0bf3b30ad5482bb519377bd4d43b927`, upstream OpenPI commit
`15a9616a00943ada6c20a0f158e3adb39df2ccac`, and MuJoCo Menagerie commit
`71f066ad0be9cd271f7ed58c030243ef157af9f4`. The final live response digest
was `2904ddacfb2c0643d3b334378772cdb90a4a37eba0d41935811ae7b564deb8ab`.

## Claim boundary

The simulated robot remained within the registered collision, self-collision,
joint-limit, acceleration-stop, and plan-validity checks. This is not a
physical safety certification or an OS hard-real-time guarantee.

`target_reached` was false. The free-space Panda goal and prompt are an
integration probe outside the checkpoint's official LIBERO task protocol, so
this artifact must not be cited as evidence of pi0.5 task competence or as an
official LIBERO success rate. Its defensible claim is narrower: a real,
content-attested checkpoint response crossed the Hx7-to-Hx8 semantic boundary,
was scheduled off the control thread, and was executed under the registered
runtime constraints.

## Validate

From the repository root:

```bash
python -m integrations.openpi.validate_live_panda_smoke \
  evidence/g01_live_panda_smoke_final_001 --json
```

The validator independently checks the run manifest, live-provider identity,
checkpoint attestation, GCS CRC inventory, event lifecycle, NPZ clocks and
state traces, and full MP4 decoding/nonblank motion. It does not rerun the
checkpoint.

Review files:

- `run/summary.json`: protocol, implementation hashes, identity, and metrics.
- `run/events.json`: request, response, dispatch, repair, and command events.
- `run/trace.npz`: measured wall/simulation clocks and Panda state traces.
- `run/panda_trace.mp4`: post-hoc rendering of the measured trajectory.
- `server/checkpoint_attestation.json`: OpenPI and checkpoint provenance.
- `checkpoint_crc32c_verification.json`: public GCS object integrity audit.
- `validation.json`: saved independent validation report.
