# pi0.5 projected-overlap pilot

## Question

This exploratory study asks whether hard projected flow conditioning improves
closed-loop pi0.5 execution when both methods use the same fixed-width overlap
scheduler and explicit policy noise. It is an integration and mechanism pilot,
not a confirmatory efficacy study and not an implementation of RTC
pseudoinverse guidance.

## Frozen comparison

- Policy: official OpenPI `pi05_libero` checkpoint.
- Checkpoint content SHA-256:
  `9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5`.
- OpenPI upstream: `15a9616a00943ada6c20a0f158e3adb39df2ccac`.
- Projected-conditioning extension:
  `2c8e61d5fbfde4b670ae428ef4b8440d35c1c7fd`.
- ArmBench run commit:
  `baceec016c89bf1def6ea156f63d13c2c7f65d6a`.
- Matrix: 10 LIBERO-10 tasks x 2 fixed initial states x 2 methods.
- Scheduler: action horizon `H=10`, execute horizon `E=5`, fixed delay
  `d=4` actions, equivalent to 200 ms at 20 Hz.
- Query zero is an identical unconditioned bootstrap in both methods.
- Every matched method/query receives the same explicit `10 x 32` pi0.5
  sampling noise.
- Condition order alternates within adjacent pairs.
- All 40 assigned rollouts and videos remain in the artifact.

Both modes execute exactly `old[:d] + new[d:E]` after bootstrap. The only
intervention is whether the new flow sample is hard-conditioned on the
committed prefix inside the pi0.5 sampler.

## Result

| Method | Success | Mean queries | Mean motion seam | Mean gripper seam | Max model residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| `overlap_unconditioned` | 19/20 | 53.20 | 0.109373 | 0.048326 | n/a |
| `projected_overlap` | 18/20 | 55.05 | 0.086893 | 0.059185 | 0.0 |

The paired success difference is -5 points. There were 0 projected-only wins,
1 projected-only loss, and 19 ties; the exact two-sided McNemar value is
`p=1.0`. Mean motion seam decreased by 20.55%, while mean gripper seam
increased by 22.47%.

The projected-only failure was LIBERO-10 task 8, initial state 1, "put both
moka pots on the stove". It reached the 520-action limit; the unconditioned
pair succeeded after 355 actions. Both methods reached the action limit on
task 9, initial state 0. These observations are diagnostic and post hoc, not
new outcome claims.

## Evidence integrity

The artifact contains 50 files, including 40 videos, 40 episode records,
2,165 query transitions, the exact protocol, environment and server
attestation, a compressed float32 transition archive, and a complete root
manifest.

The independent validator does not trust scheduler output hashes alone. It
recomputes:

- keyed pi0.5 sampling keys and realized noise hashes;
- raw conditioning-action and prefix-mask hashes;
- every executed `old[:d] + new[d:E]` window, including partial final windows;
- every `new[E:H] + zeros(E)` next reference;
- every cross-query reference chain;
- archive, file, and aggregate manifest SHA-256 values.

The accepted identities are:

- root manifest SHA-256:
  `88ad66d2fce1efedca3ac04200ac569c5fbc5108a6e4c7be65ae38b4d7ccb9f9`;
- transition archive SHA-256:
  `6d87323aae7a313394f4e235b3a2952b998928f188da4108cf56d108ef13cc2d`.

Validate from the repository root:

```powershell
& '..\.venv\Scripts\python.exe' -m integrations.openpi.projected_overlap_pilot `
  validate evidence\pi05_projected_overlap_pilot_001

& '..\.venv\Scripts\python.exe' `
  -m integrations.openpi.validate_projected_overlap_artifact `
  evidence\pi05_projected_overlap_pilot_001
```

## Research decision

This pilot does not justify a 300-rollout confirmation of the same hard
projection. The useful finding is a mechanism split: exact prefix enforcement
improves translational/rotational handoff continuity but does not preserve or
improve task progress by itself.

The next direct comparison should implement RTC's soft VJP/pseudoinverse
guidance and evaluate three adjacent conditions under shared noise:

1. unconditioned overlap;
2. hard projected overlap;
3. soft RTC-style guidance.

Only after that method comparison should the project spend a larger rollout
budget. Independently ticking inference and control loops, a second model
family such as SmolVLA or OpenVLA-OFT, and real SO101/Panda evidence remain
separate generality and deployment stages.
