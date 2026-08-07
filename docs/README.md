# Documentation

This index separates current engineering guidance from frozen experimental
records. Frozen protocols and audit artifacts preserve the wording and state
that governed their corresponding runs; they are not retroactively rewritten
to match later results.

## Status labels

| Label | Meaning |
| --- | --- |
| Current | Maintained description of the present implementation or evidence |
| Operational | Instructions intended for direct use |
| Frozen | Pre-execution protocol retained for provenance |
| Audit | Immutable diagnosis or remediation record |
| Historical | Superseded plan or implementation retained for traceability |

## Start here

| Document | Status | Purpose |
| --- | --- | --- |
| [Architecture and claim boundaries](PROJECT_ARCHITECTURE.md) | Current | Two execution paths, shared runtime layer, terminology, and integration boundary |
| [架构与主张边界](PROJECT_ARCHITECTURE_ZH.md) | Current | 中文架构、术语和可公开主张范围 |
| [Results](RESULTS.md) | Current | Verified outcomes, provenance, statistics, and study-specific limitations |
| [Evidence catalog](EVIDENCE_CATALOG.md) | Current | Complete artifact inventory, evidence classes, review links, validators, and claim boundaries |
| [Evaluation methodology](METHODOLOGY.md) | Current | Runtime contracts, metrics, experimental design, and claim scope |
| [Troubleshooting](DEBUGGING.md) | Operational | Boundary-oriented diagnosis for installation, protocol, runtime, and artifacts |
| [Local CPU setup](LOCAL_SETUP.md) | Operational | Portable Windows/Linux installation, model resolution, and offline commands |
| [本地 CPU 安装](LOCAL_SETUP_ZH.md) | Operational | Windows/Linux 安装、模型路径解析与离线运行说明 |
| [Technical review guide](TECHNICAL_REVIEW.md) | Operational | Architecture walkthrough, review questions, and maintainer readiness checks |
| [OpenPI/LIBERO operations](OPENPI_LIBERO_OPERATIONS.md) | Operational | Linux/NVIDIA execution, validation lifecycle, retention, and cost controls |

## Runtime methods

| Document | Status | Purpose |
| --- | --- | --- |
| [Measured-age temporal alignment](MEASURED_LATENCY_RUNTIME.md) | Current | Timestamp-based suffix selection and held-out confirmation |
| [Non-blocking runtime harness](ASYNC_RUNTIME.md) | Current | Threaded inference mailbox, latest-only queue, deadline dispatch, and limitations |
| [非阻塞运行时验收](ASYNC_RUNTIME_ZH.md) | Current | 独立推理线程、latest-only 队列、deadline 调度与证据边界 |
| [RTC-guided pi0.5 integration](RTC_PI05_INTEGRATION.md) | Current | Scheduler contract, reverse-time VJP mapping, and corrected-v3 outcome |
| [Projected-overlap pilot](PI05_PROJECTED_OVERLAP_PILOT.md) | Current | Exploratory hard-conditioning study and its result boundary |
| [Top-venue engineering gap analysis](research/VLA_TOP_VENUE_GAP_ANALYSIS_2026.md) | Current | Method positioning, evidence classes, and remaining research gaps |

## Acceptance workflows

| Document | Status | Purpose |
| --- | --- | --- |
| [RTC corrected-v3 acceptance](RTC_OVERLAP_PRIMARY_V3_ACCEPTANCE.md) | Operational | Validate two raw v3 artifacts, rebuild the combined analysis, and open the dashboard |
| [RTC corrected-v3 acceptance — 中文](RTC_OVERLAP_PRIMARY_V3_ACCEPTANCE_ZH.md) | Operational | Corrected-v3 中文验收与结果边界 |
| [Temporal-alignment evidence acceptance — 中文](ALIGNMENT_ACCEPTANCE_ZH.md) | Operational | 验证确定性时序对齐 artifact 并检查匹配视频 |

## Frozen protocols and audits

| Document | Status | Bound study |
| --- | --- | --- |
| [Temporal-alignment confirmatory freeze](PI05_ALIGNMENT_CONFIRMATORY_FREEZE.md) | Frozen | 300-rollout LIBERO Spatial deterministic-delay study |
| [Measured-age confirmatory freeze](PI05_MEASURED_AGE_CONFIRMATORY_FREEZE.md) | Frozen | 240-rollout measured-age held-out study |
| [Cross-suite external-validation freeze](PI05_CROSS_SUITE_EXTERNAL_VALIDATION_FREEZE.md) | Frozen | 300-rollout Object, Goal, and LIBERO-10 validation |
| [Corrected RTC primary protocol](research/RTC_OVERLAP_PRIMARY_300_V3_PROTOCOL.md) | Frozen | 300-rollout corrected-v3 RTC comparison |
| [RTC query-zero pairing audit](research/RTC_OVERLAP_PAIRING_AUDIT_20260805.md) | Audit | Detection and remediation of v2 environment carryover |

## Historical records

| Document | Status | Notes |
| --- | --- | --- |
| [Original pi0.5-LIBERO runbook](PI05_LIBERO_STUDY.md) | Historical | Planning-era budget and evidence ladder; later studies report their completed outcomes elsewhere |
| [Superseded RTC primary protocol](research/RTC_OVERLAP_PRIMARY_300_PROTOCOL.md) | Historical | v2 protocol rejected after the pairing audit |

## Evidence and generated reports

The evidence/ tree contains preserved experiment artifacts. The reports/ tree
contains generated offline dashboards. Treat evidence directories as read-only:
create a new run ID for diagnostics or reruns.

The generated [evidence catalog](EVIDENCE_CATALOG.md) covers every top-level
artifact and links its machine-readable result, protocol, manifest, raw review
files, validator command, and claim boundary. Run
`python scripts/build_evidence_catalog.py --check` to verify that the catalog
matches all Git-tracked evidence bytes.

The active result summary is [Results](RESULTS.md). When a summary and a frozen
protocol use different tense or status language, the protocol records what was
declared before execution; the current result summary records what was
subsequently observed and validated.
