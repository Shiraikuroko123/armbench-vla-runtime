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
| [Live pi0.5-to-Panda integration gate](../evidence/g01_live_panda_smoke_final_001/summary.md) | Current | Attested checkpoint identity, live Hx7-to-Hx8 Panda bridge, asynchronous timing evidence, video, validator, and strict claim boundary |
| [G02 independent-clock pi0.5-LIBERO pilot](G02_INDEPENDENT_CLOCK_PILOT.md) | Current | 40-rollout official-checkpoint simulation pilot with independent simulation/inference clocks, complete failures, and GPU-free validator |
| [G02 独立时钟 pilot](G02_INDEPENDENT_CLOCK_PILOT_ZH.md) | Current | 40 次 LIBERO Spatial 运行、请求生命周期、时钟重叠证明、失败视频与验收命令 |
| [Architecture and claim boundaries](PROJECT_ARCHITECTURE.md) | Current | Two execution paths, shared runtime layer, terminology, and integration boundary |
| [架构与主张边界](PROJECT_ARCHITECTURE_ZH.md) | Current | 中文架构、术语和可公开主张范围 |
| [中文代码导读](CODE_WALKTHROUGH_ZH.md) | Operational | 从 CLI、provider/worker、调度器到 Panda 保障链和 artifact validator 的断点地图 |
| [本地 CPU 一键验收](CPU_ACCEPTANCE_ZH.md) | Operational | 单命令重跑碰撞、动力学、provider、LeRobot、Panda 与异步证据，输出本地汇总 |
| [完整改进路线与采购表](ROADMAP_ZH.md) | Current | 后续工作、前置条件、预算、验收标准、优先级与采购边界 |
| [Results](RESULTS.md) | Current | Verified outcomes, provenance, statistics, and study-specific limitations |
| [Evidence catalog](EVIDENCE_CATALOG.md) | Current | Complete artifact inventory, evidence classes, review links, validators, and claim boundaries |
| [Evaluation methodology](METHODOLOGY.md) | Current | Runtime contracts, metrics, experimental design, and claim scope |
| [Troubleshooting](DEBUGGING.md) | Operational | Boundary-oriented diagnosis for installation, protocol, runtime, and artifacts |
| [Local CPU setup](LOCAL_SETUP.md) | Operational | Portable Windows/Linux installation, model resolution, and offline commands |
| [本地 CPU 安装](LOCAL_SETUP_ZH.md) | Operational | Windows/Linux 安装、模型路径解析与离线运行说明 |
| [MuJoCo swept collision audit](MUJOCO_SWEPT_AUDIT.md) | Current | Clearance-backed static-obstacle subdivision, dense-oracle audit, and claim boundary |
| [Panda continuous self-collision audit](MUJOCO_SELF_COLLISION_AUDIT.md) | Current | Registered self-collision certificate, dense-oracle comparison, and fail-closed boundary |
| [Panda dynamics braking audit](DYNAMICS_BRAKING_AUDIT.md) | Current | MuJoCo inverse-dynamics stop feasibility across payload, damping, velocity, and continuous collision edges |
| [Panda 动力学制动审计](DYNAMICS_BRAKING_AUDIT_ZH.md) | Current | 负载、阻尼、初速度矩阵中的逆动力学、连续碰撞边与主张边界 |
| [Integrated Panda action assurance](INTEGRATED_PANDA_ASSURANCE.md) | Current | Atomic OSQP, continuous-collision, stop-invariant supervision plus rerunnable MuJoCo task evidence |
| [Panda 集成动作保障链](INTEGRATED_PANDA_ASSURANCE_ZH.md) | Current | 原子动作监管、注册故障矩阵、闭环任务证据、本地验收与在线耗时边界 |
| [Panda 异步动作保障 CPU 收口](CPU_RUNTIME_COMPLETION_ZH.md) | Operational | provider、双 worker、完整监督与 reset-safe 原子发布的 17 案例验收 |
| [冻结 pi0.5 响应的完整 Panda CPU 回放](PI05_INTEGRATED_PANDA_CPU_REPLAY_ZH.md) | Current | 真实保存响应、三模式成对矩阵、20 ms go/no-go、候选/发布轨迹和独立 validator |
| [优化后的 pi0.5 到 Panda CPU 保障回放](PI05_OPTIMIZED_CPU_REPLAY_ZH.md) | Current | persistent OSQP、broad phase、workspace/证书复用、180 案例审计与边界性 `go` |
| [优化后 CPU 保障链的重复性审计](PI05_OPTIMIZED_CPU_REPEATABILITY_ZH.md) | Current | 3 次空闲与 3 次 CPU 负载冷启动、嵌套 artifact 校验，以及不可重复的 20 ms execute 结论 |
| [Panda 原子窗口 CPU 重复性审计](PI05_WINDOWED_CPU_REPEATABILITY_ZH.md) | Current | `H=10`/`H=1` 成对窗口、6 轮 fresh-process idle/load 重跑与窗口级原子发布证据 |
| [Official LeRobotDataset round-trip](OFFICIAL_LEROBOT_ROUNDTRIP.md) | Current | Pinned LeRobot v3.0 dataset export, isolated loader reload, and exact Panda feature semantics |
| [官方 LeRobotDataset round-trip](OFFICIAL_LEROBOT_ROUNDTRIP_ZH.md) | Current | 官方 loader 版本锁定、Panda Hx8 动作语义与逐字段 round-trip |
| [MuJoCo swept 碰撞审计](MUJOCO_SWEPT_AUDIT_ZH.md) | Current | 基于 clearance 的静态障碍细分、dense 对照和主张边界 |
| [Panda 连续自碰撞审计](MUJOCO_SELF_COLLISION_AUDIT_ZH.md) | Current | 注册自碰撞证书、dense 对照与 fail-closed 边界 |
| [Provider-neutral action contract](PROVIDER_CONTRACT.md) | Current | Frozen provider identity, exact action semantics, fail-closed gate, and second-family ABI audit |
| [Provider-neutral 动作契约](PROVIDER_CONTRACT_ZH.md) | Current | 冻结 provider 身份、精确动作语义、fail-closed 门禁与第二模型族 ABI 审计 |
| [LeRobot-style runtime bridge](LEROBOT_RUNTIME_BRIDGE.md) | Current | Frame mapping, actuator watchdog, episode export, and deterministic replay |
| [LeRobot 风格运行时桥接](LEROBOT_RUNTIME_BRIDGE_ZH.md) | Current | 帧映射、执行器 watchdog、episode 导出与确定性重放 |
| [Technical review guide](TECHNICAL_REVIEW.md) | Operational | Architecture walkthrough, review questions, and maintainer readiness checks |
| [OpenPI/LIBERO operations](OPENPI_LIBERO_OPERATIONS.md) | Operational | Linux/NVIDIA execution, validation lifecycle, retention, and cost controls |

## Runtime methods

| Document | Status | Purpose |
| --- | --- | --- |
| [Measured-age temporal alignment](MEASURED_LATENCY_RUNTIME.md) | Current | Timestamp-based suffix selection and held-out confirmation |
| [Non-blocking runtime harness](ASYNC_RUNTIME.md) | Current | Threaded inference mailbox, latest-only queue, deadline dispatch, and limitations |
| [Independent-clock runtime smoke](INDEPENDENT_CLOCK_RUNTIME.md) | Current | Spawned provider process, parent-owned wall-clock ticks, latest-only supersession, and auditable deadline decisions |
| [G02 official-checkpoint pilot](G02_INDEPENDENT_CLOCK_PILOT.md) | Current | Attested `pi05_libero` matrix, 38/40 result, failure retention, and validation boundary |
| [非阻塞运行时验收](ASYNC_RUNTIME_ZH.md) | Current | 独立推理线程、latest-only 队列、deadline 调度与证据边界 |
| [Panda 异步动作保障 CPU 收口](CPU_RUNTIME_COMPLETION_ZH.md) | Current | provider-neutral 输入、异步集成监督、原子发布与可重算故障矩阵 |
| [Asynchronous Panda closed loop](ASYNC_PANDA_CLOSED_LOOP.md) | Current | Live dual-camera capture, blocking policy worker, measured-age dispatch, braking fallback, and torque-level MuJoCo execution |
| [异步 Panda 闭环运行时](ASYNC_PANDA_CLOSED_LOOP_ZH.md) | Current | 双相机采集、异步策略、观测年龄调度、制动回退与 MuJoCo 力矩执行的本地验收 |
| [LIBERO-to-Panda Cartesian adapter](PANDA_CARTESIAN_ADAPTER.md) | Current | Jacobian-based component bridge from 7-D Cartesian chunks to guarded Panda joint velocity |
| [LIBERO 到 Panda 动作适配器](PANDA_CARTESIAN_ADAPTER_ZH.md) | Current | 7 维末端动作到受保护 Panda 关节速度的组件级转换与边界 |
| [Provider-neutral action contract](PROVIDER_CONTRACT.md) | Current | Model-family identity, semantic SHA-256 gate, frozen responses, and Panda adapter binding |
| [Provider-neutral 动作契约](PROVIDER_CONTRACT_ZH.md) | Current | 模型族身份、动作语义哈希门禁、冻结响应与 Panda adapter 绑定 |
| [LeRobot-style runtime bridge](LEROBOT_RUNTIME_BRIDGE.md) | Current | LeRobot-style frames, fail-closed command watchdog, and replayable episode evidence |
| [LeRobot 风格运行时桥接](LEROBOT_RUNTIME_BRIDGE_ZH.md) | Current | LeRobot 风格帧、fail-closed 命令 watchdog 与可重放 episode 证据 |
| [Frozen pi0.5 response replay](PI05_PANDA_ARCHIVE_REPLAY.md) | Current | Hash-verified official responses replayed through the offline Panda adapter and guard |
| [冻结 pi0.5 响应的 Panda 离线回放](PI05_PANDA_ARCHIVE_REPLAY_ZH.md) | Current | 官方冻结动作的严格校验、分层回放、结果与主张边界 |
| [Deadline-bounded braking-invariant repair](PI05_PANDA_BRAKING_REPAIR.md) | Current | Whole-chunk scale search, terminal braking validation, and paired frozen-response result |
| [延迟有界的终端制动不变量修复](PI05_PANDA_BRAKING_REPAIR_ZH.md) | Current | 冻结 `pi0.5` 响应上的轨迹级动作修复、可视化验收与主张边界 |
| [Frozen responses through integrated Panda assurance](PI05_INTEGRATED_PANDA_CPU_REPLAY.md) | Current | Direct/QP/full-assurance CPU matrix with a frozen 20 ms go/no-go rule |
| [冻结响应的 Panda 完整保障 CPU 回放](PI05_INTEGRATED_PANDA_CPU_REPLAY_ZH.md) | Current | 同一真实响应上的直接发布、QP 与完整连续碰撞/制动保障对照 |
| [Optimized pi0.5-to-Panda CPU assurance replay](PI05_OPTIMIZED_CPU_REPLAY.md) | Current | Persistent workspaces, conservative collision pruning, atomic publication, and the 180-case timing audit |
| [优化后的 pi0.5 到 Panda CPU 保障回放](PI05_OPTIMIZED_CPU_REPLAY_ZH.md) | Current | 20/100 ms 双 profile、独立 validator、最低 `go` 条件与稳定实时边界 |
| [Optimized CPU repeatability audit](PI05_OPTIMIZED_CPU_REPEATABILITY.md) | Current | Six cold-process idle/load reruns showing repeatable fail-closed publication but unstable 20 ms execution |
| [优化后 CPU 保障链的重复性审计](PI05_OPTIMIZED_CPU_REPEATABILITY_ZH.md) | Current | 6 轮独立进程重跑、受控 CPU 竞争、逐轮 artifact 与当前工程解释 |
| [Windowed Panda CPU repeatability audit](PI05_WINDOWED_CPU_REPEATABILITY.md) | Current | Six paired `H=10`/`H=1` fresh-process trials with nested validation and atomic-window publication evidence |
| [Panda 原子窗口 CPU 重复性审计](PI05_WINDOWED_CPU_REPEATABILITY_ZH.md) | Current | 单动作认证窗口、受控 CPU 竞争、逐轮 artifact 与源动作块原子性边界 |
| [RTC-guided pi0.5 integration](RTC_PI05_INTEGRATION.md) | Current | Scheduler contract, reverse-time VJP mapping, and corrected-v3 outcome |
| [Projected-overlap pilot](PI05_PROJECTED_OVERLAP_PILOT.md) | Current | Exploratory hard-conditioning study and its result boundary |
| [Top-venue engineering gap analysis](research/VLA_TOP_VENUE_GAP_ANALYSIS_2026.md) | Current | Method positioning, evidence classes, and remaining research gaps |

## Acceptance workflows

| Document | Status | Purpose |
| --- | --- | --- |
| [Local CPU acceptance — 中文](CPU_ACCEPTANCE_ZH.md) | Operational | Rerun every saved local CPU validator, optionally including official LeRobot and the full test suite |
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
| [Independent-clock deadline follow-up](research/pi05_deadline_followup_protocol_20260810.json) | Frozen | Spatial 150/155/175 ms multi-seed boundary matrix |
| [Independent-clock nominal references](research/pi05_deadline_nominal_reference_protocol_20260810.json) | Frozen | Registered 200 ms Spatial/Object reference cells |
| [Independent-clock Object transfer](research/pi05_object_deadline_transfer_protocol_20260810.json) | Frozen | Object 150/175 ms cross-suite deadline cells |
| [Held-out action-selection baseline](research/pi05_independent_clock_selection_baseline_protocol_20260810.json) | Frozen | Three-seed Spatial comparison with query-0 pairing gates |
| [Integrated Panda CPU replay protocol](research/pi05_integrated_panda_cpu_protocol_20260810.json) | Frozen | 30 responses x 3 scenes x 3 modes under a fixed 20 ms software budget |
| [Optimized CPU repeatability protocol](research/pi05_optimized_cpu_repeatability_protocol_20260811.json) | Frozen | Three idle and three four-worker cold-process reruns of the 180-case optimized audit |
| [Windowed CPU replay protocol](research/pi05_windowed_cpu_replay_protocol_20260811.json) | Frozen | Paired ten-action and one-action certification windows over the frozen 180-row audit |
| [Windowed CPU repeatability protocol](research/pi05_windowed_cpu_repeatability_protocol_20260811.json) | Frozen | Three idle and three four-worker fresh-process reruns of the paired window audit |

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

The current independent-clock deadline synthesis is the
[18-cell / 720-rollout report](../reports/pi05_deadline_multisuite_report_720_20260810_001/summary.md).
Its 50 ms stress control remains separate from the balanced report. The
held-out action-selection study is the
[six-cell / 240-rollout report](../reports/pi05_selection_heldout_report_240_20260810_001/summary.md);
all six source cells, 120 query-0 pairing gates, and the exact frozen-matrix
profile pass current-code validation.
