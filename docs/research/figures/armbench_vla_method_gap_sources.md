# ArmBench VLA method-gap figure: sources and reproduction

This note records the factual basis for `armbench_vla_method_gap.{png,pdf,svg}`.
The figure is a qualitative method taxonomy, not a performance ranking. A blank
or "not central" capability statement means that the cited work does not make
that question its main intervention; it is not proof that no implementation of
the work can address it.

## Reproduce

From the repository's `project` directory:

```powershell
& '..\.venv\Scripts\python.exe' `
  'docs\research\figures\generate_vla_method_gap.py'
```

The script uses the repository-pinned Matplotlib dependency and writes PNG,
vector PDF, and SVG versions beside itself. It contains no network calls; the
literature status was checked separately before freezing the labels below.

## ArmBench evidence boundary

The ArmBench row is grounded in the repository's frozen evidence and claim
boundaries:

- `README.md`, sections "Cross-suite external validation", "Confirmatory
  pi0.5 result", and "Claim boundaries";
- `docs/RESULTS.md`, verified pi0.5-LIBERO result snapshots;
- `docs/PI05_ALIGNMENT_CONFIRMATORY_FREEZE.md`;
- `docs/PI05_CROSS_SUITE_EXTERNAL_VALIDATION_FREEZE.md`.

The row combines the 300-rollout Spatial study and the separately frozen
300-rollout Object/Goal/LIBERO-10 external-validation family. The figure does
not call ArmBench a paper, preprint, real-robot deployment, RL method, or VLA
training method.

## Comparison works

Status was checked on 2026-08-05. "Formal paper" means a venue proceedings or
journal version was located, even when an arXiv version also exists.

| Route | Formal status used in figure | Identifier | Claim basis used |
| --- | --- | --- | --- |
| OpenVLA-OFT | Robotics: Science and Systems 2025 | DOI `10.15607/RSS.2025.XXI.017`; arXiv `2502.19645` | Offline VLA fine-tuning; parallel decoding, action chunking, continuous actions, and L1 objective; LIBERO and real ALOHA evaluations. |
| DPPO | ICLR 2025 | arXiv `2409.00588`; official ICLR 2025 paper listing | Policy-gradient RL fine-tuning for diffusion policies; simulation and zero-shot hardware deployment. |
| HIL-SERL | Science Robotics 2025 | DOI `10.1126/scirobotics.ads5033`; arXiv `2410.21845` | Real-world vision-based RL using demonstrations and human corrections on dexterous manipulation tasks. |
| RTC | NeurIPS 2025 | arXiv `2506.07339`; official NeurIPS 2025 paper listing | Training-free inference-time execution for diffusion/flow action chunks; freezes committed actions and inpaints the continuation; dynamic simulation and real bimanual evaluation. |

All four comparison routes now have formal versions. In particular, RTC was a
preprint when first released in June 2025, but its current arXiv record states
that it was published at NeurIPS 2025, and the title appears in the official
NeurIPS 2025 paper list. The figure therefore does not label RTC as a current
preprint.

## Reproducible literature lookup

The lookup was targeted, not exhaustive.

- OpenAlex works search, `per_page=5` or `20`, title queries
  `OpenVLA-OFT`, `Diffusion Policy Policy Optimization`,
  `Precise and Dexterous Robotic Manipulation via Human-in-the-Loop
  Reinforcement Learning`, and `"RTC" "action chunking"`.
- arXiv Atom API:
  `https://export.arxiv.org/api/query?id_list=2502.19645,2409.00588,2410.21845,2506.07339&max_results=4`.
- Official venue lists:
  `https://iclr.cc/virtual/2025/papers.html` and
  `https://neurips.cc/virtual/2025/papers.html`; each returned HTTP 200 and
  contained the corresponding exact title.
- OpenAlex returned the published RSS DOI for OpenVLA-OFT and the published
  Science Robotics DOI for HIL-SERL. It still represented DPPO and RTC chiefly
  through their arXiv records, so the official conference lists were used to
  establish formal publication status.
- An auxiliary Semantic Scholar record request returned HTTP 429 without an API
  key, and an auxiliary DBLP search returned HTTP 500. Neither response was
  used for a status or method claim; the identifiers and official venue pages
  above provide the recorded provenance.

No citation counts, success rates, training-time comparisons, or relative
performance numbers from these papers are encoded in the figure.
