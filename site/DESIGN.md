# ArmBench Website Design System

## Direction

**Graphite runtime console with signal-yellow timing marks.**

The interface is a compact project report rather than a paper template or a
generic AI landing page. MuJoCo and LIBERO media lead; controller ticks, action
indices, exact statistics, result files, and project limits form the visual
language.

## Palette

The site theme is consistently dark so media, measurements, and state
changes can be compared without a visual mode switch. Yellow is reserved for
timing, selection, and provenance. Red and green are semantic only.

```css
--bg: #0b0d0c;
--bg-deep: #070908;
--surface: #121513;
--surface-raised: #191d1a;
--surface-strong: #222720;
--text: #f2f0e8;
--text-soft: #cbc9c0;
--muted: #969b93;
--line: #30352f;
--line-strong: #50564d;
--signal: #f1c84b;
--danger: #f47d73;
--success: #83d09d;
```

There are no gradients, decorative orbs, neon circuitry, or stock imagery.

## Typography

- Display and body: Segoe UI Variable / Segoe UI with Chinese system fallbacks.
- Measurements, commands, status labels, and action indices: Cascadia Code or
  the local monospace fallback.
- The 96 px display size is reserved for the ArmBench name on wide screens.
- Section headings remain compact; letter spacing is always zero.
- Font size changes at discrete breakpoints and never scales with viewport
  width.

## Layout

- Maximum content rail: 1240 px with 32 px desktop gutters.
- Full-bleed hero footage with identity and capability over the actual scene.
- Unframed full-width sections separated by one-pixel rules.
- A four-cell artifact dock supplies the visible next-section cue.
- Cards are limited to repeated testbeds, suites, and fault cases.
- Media, charts, action tracks, and controls have stable aspect ratios or grid
  tracks to prevent layout shift.
- Corner radius is 4 px; there are no nested decorative cards.

## Core Components

- **File dock:** direct entry to code, the generated result-file index,
  results, and debugging.
- **Result tabs:** measured-age experiment, cross-suite run, RTC null result.
- **Matched player:** aligned square videos, one play state, restart, progress,
  and speed control. The seven-second clip holds its final frame on the shared
  22-second timeline.
- **Timing lab:** an observation-age range input that renders stale actions,
  execution window, and execute/refresh state from the frozen protocol.
- **Testbed split:** π0.5-LIBERO and Panda-MuJoCo stay visually and
  semantically separate, followed by the integration gap.
- **Implemented modules:** provider adaptation, LeRobot-style frame replay,
  official LeRobotDataset round-trip, constrained projection, braking repair,
  asynchronous Panda control, continuous collision edges, dynamics braking
  audit, and result-file checks.
- **Fault map:** scripted non-learned detection and response rows with links to
  saved files.
- **Scope block:** checked items and unfinished items receive equal visual
  weight.

## Media Rules

- Hero: repository-authentic Panda MuJoCo MP4 and WebP poster.
- Comparison: repository-authentic π0.5-LIBERO matched pair and posters.
- Every clip names simulation, embodiment, task or scene, outcome, conditions,
  and playback rate where applicable.
- The qualitative pair is disclosed as one selected aligned-only win; aggregate
  rates are explicitly tied to all 120 pairs.
- Only the hero may autoplay, always muted and inline. Below-fold videos load
  near the viewport or after user intent.
- Reduced motion pauses the hero. Data-saving mode requests no MP4 until intent.

## Responsive Behavior

- `1100 px`: reduce display scale and stack the timing lab.
- `860 px`: replace desktop navigation with a keyboard-accessible menu; stack
  evidence where comparison width is insufficient.
- `620 px`: use a fixed 720 px hero, two-column artifact dock, compact controls,
  and mobile-safe media labels.
- `380 px`: tighten language controls and player layout while preserving a
  320 px minimum page width.

The 320 px composition keeps the project name, core claim, Panda evidence,
simulation label, primary action, and the beginning of the artifact dock in
the first viewport without horizontal overflow.

## Interaction And Accessibility

- Chinese and English switch in place and persist locally.
- Tabs support Left, Right, Home, and End keys with roving focus.
- The mobile menu exposes accurate expanded state and closes on navigation,
  Escape, or an outside click.
- All state-changing controls are native buttons, ranges, or selects.
- Focus uses a visible two-pixel signal-yellow outline.
- Horizontally scrollable command text is keyboard focusable.
- Semantic landmarks, one `h1`, sequential headings, captions, alt text,
  accessible tab relationships, and a skip link are required.
- Content remains visible when JavaScript or reveal animation is unavailable.
- `prefers-reduced-motion` removes reveal motion and smooth scrolling.

## Performance And Deployment

- No frontend framework, external font, analytics script, or runtime CDN.
- HTML, CSS, and JavaScript remain below the 500 KB source target.
- Hero video is below 8 MB and uses `preload="none"` with a real poster.
- Below-fold media is lazy-attached; posters reserve dimensions.
- Metadata includes canonical URL, Open Graph, Twitter card, favicon, and
  `SoftwareSourceCode` JSON-LD.
- GitHub Pages is the deployment target; all local paths are relative to the
  repository root served from `site/`.

## Provenance

- Panda footage: `evidence/mujoco_formal_20260803`.
- Matched π0.5-LIBERO media:
  `evidence/pi05_libero_measured_age_confirmatory_001`.
- Numerical claims: `docs/RESULTS.md`, repository evidence manifests, and the
  cross-suite evidence Release.
- Runtime extension claims: `docs/PROVIDER_CONTRACT.md`,
  `docs/LEROBOT_RUNTIME_BRIDGE.md`, `docs/OFFICIAL_LEROBOT_ROUNDTRIP.md`,
  `docs/PI05_PANDA_BRAKING_REPAIR.md`, `docs/MUJOCO_SWEPT_AUDIT.md`,
  `docs/DYNAMICS_BRAKING_AUDIT.md`, and `docs/ASYNC_PANDA_CLOSED_LOOP.md`,
  plus their checked-in reports.
- Licensing: repository `LICENSE` and `THIRD_PARTY_NOTICES.md`.
