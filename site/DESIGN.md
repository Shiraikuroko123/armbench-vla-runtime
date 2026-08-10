# VLA-Sync website design system

## Archetype

Primary archetype: research project page. Secondary archetype: engineering
reproducibility surface. The page follows a paper-like proof sequence while
keeping the commands and artifact trail directly usable by an interviewer.

## Visual thesis

The visual system represents the project's central mechanism rather than a
generic AI aesthetic:

- near-black graphite is the simulator and workbench substrate;
- coral is the inference clock, response age, and stale-action risk;
- teal is the control clock, accepted action suffix, and verified execution;
- amber is provenance, caution, and claim-boundary evidence;
- red is reserved for retained failures.

Large directional color fields separate inference and control without using
purple branding, stock robots, decorative orbs, or generated concept art. A
faint control lattice appears only where it supports the timing/runtime idea.
Each full-width section has a distinct field or grid treatment so the page has
depth and progression, while the authentic rollout pixels remain unfiltered.

The visual rhythm borrows the useful part of modern research sites such as
TwinRL: a confident centered identity, immediate physical evidence, and broad
background transitions. It does not copy TwinRL assets, purple/pink identity,
source code, or card styling.

## Layout contract

- The first viewport contains VLA-Sync, a literal technical claim, artifact
  actions, and the beginning of the authentic teaser.
- Media uses a wider evidence rail than prose.
- Sections are full-width bands; cards are limited to repeated rollout items
  and framed tools.
- Method, chart, table, and media dimensions remain stable across state
  changes.
- Desktop, tablet, and mobile are separately composed; no desktop figure is
  simply scaled until its labels become unreadable.

## Media contract

- Only the hero teaser may autoplay. It is muted, inline, poster-backed, and
  has a visible pause control.
- Below-fold videos attach their source near the viewport or when their tab is
  selected. Hidden-tab videos are paused.
- Captions identify simulator, embodiment, task/seed/episode, method, outcome,
  and whether the clip participates in a registered study.
- Failure clips are retained alongside success evidence. Aggregate claims come
  from full reports, never from selected footage.
- Provenance and inclusion status are recorded in `media-manifest.json`.

## Interaction contract

- Chinese and English switching persists in local storage.
- Mobile navigation exposes state through `aria-expanded` and closes after
  navigation.
- Rollout tabs support click plus Left/Right/Home/End keyboard movement.
- Copy controls provide visible success/failure feedback.
- Reduced-motion and data-saving users receive a paused hero and stable page.
- Content is visible without scroll-trigger JavaScript; enhancement failure
  must not produce an empty page or incomplete full-page capture.

## Accessibility and performance

- One `h1`, semantic landmarks, ordered headings, named controls, and visible
  focus states.
- Text and controls target WCAG AA contrast against their actual backgrounds.
- The site has no remote fonts or runtime framework dependencies.
- Initial HTML/CSS/JavaScript remains below 500 KB; hero video remains below
  8 MB; below-fold videos are not fetched at initial load.
- Supported verification viewports: 1440x900, an intermediate tablet width,
  390x844, and 320 px minimum width.

## Result and claim rules

- `π0.5` is written as Physical Intelligence's π0.5 VLA; OpenPI is the model
  implementation and serving stack, not another model.
- LIBERO checkpoint results and Panda substrate results stay separate.
- The hero is an equal-size, synchronized LIBERO smoke pair; both panes
  succeed and neither enters the frozen-240 statistics.
- The G01 π0.5-to-Panda probe remains integration evidence with
  `target_reached=false`, but it is not used in the hero.
- The Panda 2/2 result is scripted planning evidence.
- No text implies hard real time, hardware safety, training, fine-tuning, or
  cross-model generality.

## QA release gate

Before publication, verify HTML and JavaScript syntax, desktop/tablet/mobile
screenshots, horizontal overflow, keyboard tabs, language/menu/copy controls,
posters and nonblank video frames, lazy media loading, console/network errors,
metadata, and the public GitHub Pages URL.
