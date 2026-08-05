# ArmBench Website Design System

## Direction

**Runtime observatory: a calibrated robotics instrument in a dark lab, with
white evidence sheets and brass timing marks.**

The page should feel exact, inspectable, and operational. It must not resemble
a generic AI landing page. The visual language comes from controller clocks,
action indices, status lamps, trace lines, hashes, and audit stamps. Real
MuJoCo and LIBERO footage is the primary imagery.

## Palette

The Impeccable palette seed is hue 230. The strategy is restrained: a neutral
near-black hero and navigation, literal white evidence bands, a cobalt timing
color, and a warm brass accent that distinguishes deadlines from policy data.
All production colors use OKLCH.

```css
--black: oklch(0.085 0 0);
--black-soft: oklch(0.145 0.012 230);
--white: oklch(1 0 0);
--paper: oklch(0.965 0.006 95);
--ink: oklch(0.19 0.018 230);
--muted: oklch(0.50 0.025 230);
--primary: oklch(0.67 0.14 230);
--primary-dark: oklch(0.45 0.086 230);
--accent: oklch(0.78 0.14 83);
--danger: oklch(0.59 0.18 28);
--success: oklch(0.61 0.13 155);
```

## Typography

- Display and body: system UI with Segoe UI / PingFang SC / Microsoft YaHei
  fallbacks. This avoids remote font dependencies in China.
- Technical labels and measurements: ui-monospace.
- Hero type is reserved for the project name and literal runtime proposition.
  Panel and section headings stay compact.
- Letter spacing is zero. Font size does not scale directly with viewport width.

## Layout

- Maximum reading width: 1180 px.
- Full-bleed hero media; constrained text and instrument overlays.
- Full-width alternating black, white, and paper bands instead of floating
  section cards.
- Cards are used only for repeated evidence items and video cases, with 6 px
  corner radii.
- Stable tracks and aspect ratios prevent video, chart, and control shifts.

## Interaction

- Language control switches Chinese and English without navigation and saves
  the choice locally.
- Observation-age range input recomputes the frozen protocol decision using a
  50 ms period, ceil rounding, ten-action chunk, five-action replan horizon,
  and 250 ms deadline.
- Video controls use familiar play/pause and comparison labels. Autoplay is
  muted and disabled when reduced motion is requested.
- Copy buttons expose the two offline acceptance commands.
- Motion is limited to status pulses, timeline transitions, and section reveal.
  Nothing essential depends on animation.

## Accessibility

- Semantic landmarks, headings, table captions, and descriptive alt text.
- Visible focus indicators and a skip link.
- Native range and button controls with accessible names.
- Minimum 44 px interactive targets on touch layouts.
- Text contrast meets WCAG AA; body text targets 7:1 where practical.
- `prefers-reduced-motion` disables autoplay-like motion and smooth scrolling.

## Public Asset Provenance

- Panda footage: `evidence/mujoco_formal_20260803`.
- Matched pi0.5-LIBERO videos:
  `evidence/pi05_libero_measured_age_confirmatory_001`.
- Research landscape figure: `docs/research/figures/armbench_vla_method_gap`.
- Numerical claims: `docs/RESULTS.md` and immutable evidence manifests.

No stock imagery, generated robot imagery, personal data, or unsupported result
is used.
