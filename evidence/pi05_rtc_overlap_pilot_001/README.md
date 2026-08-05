# pi0.5 RTC overlap pilot evidence

This directory preserves an exploratory three-method closed-loop pilot on the
official `pi05_libero` checkpoint. It is an integration and mechanism study,
not a confirmatory efficacy claim.

## Frozen matrix

- 10 LIBERO-10 tasks x initial-state indices `0,1` x 3 methods
- 20 matched triplets / 60 rollouts
- `H=10`, `E=5`, injected overlap delay `d=4`
- methods: unconditioned overlap, hard projected overlap, RTC-guided overlap
- reference-only bootstrap and three-way Latin method order
- identical explicit pi0.5 sampling-noise key within each triplet
- ArmBench evaluator commit: `2aef062256fc3f6257f9f58d68c3f18c07d1b0b8`
- OpenPI extension commit: `54592c7148ba69bf52757385502782f80f2285e0`
- checkpoint content SHA-256:
  `9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5`

## Results

The independent report validates the raw artifact first, rejects incomplete or
duplicate triplets and initial-state mismatches, aggregates seam values within
episode, and only then compares methods.

| Method | Success (Wilson 95%) | Episode-equal motion seam mean / median | Episode-equal gripper seam mean / median | Scored transitions |
| --- | ---: | ---: | ---: | ---: |
| Unconditioned overlap | 20/20 [0.839, 1.000] | 0.104452 / 0.105460 | 0.058052 / 0.046187 | 994 |
| Hard projected overlap | 19/20 [0.764, 0.991] | 0.083129 / 0.082088 | 0.045345 / 0.042563 | 1,039 |
| RTC-guided overlap | 19/20 [0.764, 0.991] | 0.084002 / 0.080663 | 0.039617 / 0.035865 | 1,053 |

| Contrast vs unconditioned | Success difference | Wins / losses / ties | Exact McNemar / Holm | Motion seam mean difference (task-block 95%) | Gripper seam mean difference (task-block 95%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hard projected | -0.050 | 0 / 1 / 19 | 1.0 / 1.0 | -0.021322 [-0.032036, -0.012622] | -0.012707 [-0.029111, +0.002804] |
| RTC guided | -0.050 | 0 / 1 / 19 | 1.0 / 1.0 | -0.020450 [-0.030301, -0.011399] | -0.018435 [-0.037893, +0.000875] |

The two success contrasts do not support improvement. Smaller motion seams are
consistent across the descriptive whole-task bootstrap intervals; gripper
intervals cross zero. Seam is an exploratory process metric and receives no
superiority p-value.

## Integrity and video audit

The immutable [`evaluation`](evaluation) directory contains 70 protected files:
JSON protocol and traces, a compressed float32 transition archive, root
manifest, and all 60 videos. The downloaded cloud archive was 9,972,551 bytes;
its remote and local SHA-256 values both equal:

```text
0397944996316aa15d2e304cd2085dea095e0ab9bfcf513b07f7f221fa5f18f6
```

The source manifest byte SHA-256 is
`b9505af7c5185e1d5bf3af22a273708ec51b0a40a3c387a7865cc6777d6b07bb`.
All 60 MP4 files decode as H.264 `224x224` at 10 fps, with 155-520 frames per
video and 15,322 frames in total.

The [`analysis`](analysis) directory binds its JSON, CSV, and Markdown report to
their exact byte hashes. The analyzer records Python 3.10.8, NumPy 1.26.4, and
its source SHA-256
`0b1f2eba758772377477c2bf095befc92786dcedee83c8b6e6e880af49dcf133`.

## Reproduce

From the repository's `project` directory:

```powershell
& '..\.venv\Scripts\python.exe' -m integrations.openpi.rtc_overlap_pilot `
  validate evidence\pi05_rtc_overlap_pilot_001\evaluation

& '..\.venv\Scripts\python.exe' -m integrations.openpi.rtc_overlap_analysis `
  evidence\pi05_rtc_overlap_pilot_001\evaluation `
  --output-directory evidence\pi05_rtc_overlap_pilot_001\analysis_rebuilt
```

The analyzer refuses to overwrite an existing output directory. Compare the
rebuilt report with [`analysis/manifest.json`](analysis/manifest.json); the
portable source label and output bytes are deterministic for the pinned code.

## Claim boundary

This is a simulation-only pilot with two development initial states per task.
It does not establish task-success efficacy, full RTC-paper equivalence,
independently ticking inference/control, hard real-time behavior, deadline or
collision safety, a second checkpoint, or real-robot validity. The disjoint
300-rollout study is specified in
[`docs/research/RTC_OVERLAP_PRIMARY_300_PROTOCOL.md`](../../docs/research/RTC_OVERLAP_PRIMARY_300_PROTOCOL.md).
