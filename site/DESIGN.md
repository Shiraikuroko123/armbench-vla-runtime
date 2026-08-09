# ArmBench Website Design System

## Product archetype

Primary archetype: research / engineering project. Secondary archetype:
platform-style reproducibility surface. The page answers, in order:

1. Did a real checkpoint response reach the Panda runtime?
2. What does the Panda runtime boundary check?
3. How can a visitor run the acceptance script locally?
4. Where does the evidence stop?

The first viewport uses repository-authentic footage from the final G01
checkpoint-to-Panda integration probe. The page does not use generated robot
imagery, a generic VLA hero, or a marketing claim.

## Content contract

- **Identity:** ArmBench, an attested pi0.5-to-Panda runtime and evaluation
  platform.
- **Claim:** a real H×7 pi0.5-LIBERO response crosses an explicit H×8 Panda
  adapter and asynchronous measured-age runtime; task competence remains a
  separate question.
- **Evidence:** `pi05-panda-live-smoke.mp4`, the final G01 bundle, the 17-case asynchronous provider
  boundary, the 27-case fault matrix, two saved MuJoCo task traces, and the
  separate π0.5-LIBERO comparison pair, plus the official LeRobotDataset v3.0
  Panda H×8 roundtrip.
- **Current facts:** 35 accepted live responses, mean/P95 latency 82.75/89.56
  ms, 290/311 control ticks concurrent with inference, 94 video frames, and
  zero registered G01 simulation violations; 17/17 asynchronous boundary outcomes, 6 complete plans,
  10 holds, 1 unrecoverable stop, and zero partial prefixes; 27/27 expected
  fault outcomes, 12 accepted, 6 verified
  brakes, 7 holds, 2 unrecoverable stops; 2/2 task targets; 351/351 edges and
  braking boundaries; zero registered contacts, limit violations, or torque
  saturation; offline supervision of 5.27 s and 10.20 s.
- **Boundary:** the 2/2 reached Panda task source is scripted RRT-Connect. G01
  is a single free-space integration probe with `target_reached=false`, not an
  official LIBERO/Panda task score. All evidence is MuJoCo simulation with
  best-effort Python scheduling, not hard real time or safety certification.

## Visual direction

Graphite workbench neutrals are taken from the simulator frame. Signal yellow
marks measurements and provenance; green, blue, amber, and red encode the four
supervisor outcomes. There are no gradients, glows, decorative orbs, stock
robots, remote fonts, analytics, or runtime dependencies.

Sections are full-width bands with a constrained inner rail. Cards are limited
to the two evidence ledgers and repeated technical boundary items. Tables are
real semantic tables inside horizontally scrollable wrappers, so the method
column and long case identifiers remain inspectable on mobile.

## Interaction contract

- Chinese / English switching persists for the session and local storage.
- The mobile navigation exposes `aria-expanded`, closes on Escape and outside
  click, and returns focus to its trigger.
- The hero is the only autoplay candidate. It is muted, inline, poster-backed,
  and paused for reduced-motion or data-saving preferences.
- π0.5 comparison videos are below the fold and lazy-attached. One controller
  synchronizes play, pause, restart, scrub, and speed for both clips. The
  shorter success clip holds its last decodable frame while the 22-second
  baseline completes, and the disclosure states this explicitly.
- Acceptance commands expose copy buttons with a clipboard fallback.
- All media and dynamic controls have adjacent text labels and stable aspect
  ratios; no information depends on hover.

## Responsive and performance targets

- Desktop target: 1440 × 900; intermediate target: 900–1100 px; mobile target:
  390 × 844 (minimum supported width 320 px).
- Hero height reserves a visible artifact-bar cue below the first viewport.
- The hero video is under 8 MB and uses `preload="none"`; comparison videos
  load on intersection or user intent.
- HTML, CSS, and JavaScript are dependency-free and below the 500 KB source
  target. Media slots reserve dimensions to avoid layout shift.
- `prefers-reduced-motion`, forced colors, and high contrast receive explicit
  handling.

## Provenance

- Hero media: `evidence/g01_live_panda_smoke_final_001/run/panda_trace.mp4`,
  copied to `site/assets/media/pi05-panda-live-smoke.mp4` with a derived poster.
- π0.5 media: `evidence/pi05_libero_measured_age_confirmatory_001`, copied to
  the two comparison clips and posters.
- Numerical evidence: `docs/INTEGRATED_PANDA_ASSURANCE_ZH.md`,
  `docs/RESULTS.md`, and manifests under
  `reports/cpu_runtime_completion_001/`,
  `reports/integrated_panda_fault_matrix_001/` and
  `reports/integrated_panda_task_001/`, plus
  `reports/official_lerobot_roundtrip_001/`.
- Code and artifact links point to the public repository
  `Shiraikuroko123/armbench-vla-runtime`.
- Code license and third-party asset terms are linked in the footer.

The asynchronous CPU artifact is presented as a provider-to-supervisor boundary
result. Its recorded 438 ms assurance-worker P95 is explicitly not a deadline
guarantee, learned-policy result, closed-loop task score, or real-robot claim.
