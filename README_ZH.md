[项目网页](https://shiraikuroko123.github.io/armbench-vla-runtime/) | [English](README.md) | [文档索引](docs/README.md)

[![CPU CI](https://github.com/Shiraikuroko123/armbench-vla-runtime/actions/workflows/ci.yml/badge.svg)](https://github.com/Shiraikuroko123/armbench-vla-runtime/actions/workflows/ci.yml)

# ArmBench

ArmBench 是一个面向动作块式视觉-语言-动作策略的运行时与评测平台。它研究的不是重新训练一个 VLA，而是 VLA 已经输出一段未来动作后，如何在推理延迟、状态变化和异常响应下可信地进入控制回路。

本文首次使用的规范名称为：**Physical Intelligence 的 `pi0.5`（pi-zero-point-five）视觉-语言-动作模型，简称 `pi0.5 VLA`**。`OpenPI` 是项目使用的模型与推理实现栈，不是另一个模型名称。

## 项目由什么组成

仓库包含两条分别验证的路径：

- **七自由度 Panda 执行基座**：在本地 MuJoCo 中实现规划、受限轨迹跟踪、协议验证、故障注入和动作级检查。
- **`pi0.5 VLA` 时序评测路径**：通过 OpenPI 调用官方 checkpoint，在 LIBERO 闭环仿真中研究 action chunk 的推理时序问题。

二者共享运行时契约、观测/状态封装、响应校验、日志、视频和证据工具；当前尚未形成经过验证的“`pi0.5` 直接控制 Panda”的端到端部署。

## 要解决的问题

VLA 会根据相机画面、机器人状态和文字指令一次预测多步未来动作。模型推理需要时间，响应返回时，动作块前几步对应的时刻可能已经过去；从第 0 步直接执行会产生过期动作。

ArmBench 根据观测年龄选择未过期动作后缀；超过 deadline 时进入受限的 hold/refresh 路径；对断连、超时、NaN、错误形状、过期响应和状态不一致响应执行 fail-closed。

```text
图像 + 指令 + 机器人状态
            |
            v
Physical Intelligence pi0.5 VLA / OpenPI
            |
            v
        action chunk
            |
            v
时序调度、响应校验与故障处理
            |
            +--> LIBERO 闭环评测
            |
            +--> Panda 执行基座
                 (规划、跟踪、guard、故障注入)
```

Panda 与 LIBERO 的动作契约不同，实验结果不会混合统计。完整设计与边界见[架构与主张边界](docs/PROJECT_ARCHITECTURE_ZH.md)。

## 已验证的证据

| 研究 | 证据 | 可以得出的结论 |
| --- | --- | --- |
| 观测年龄时序对齐 | 官方 `pi0.5`-LIBERO Spatial，120 组匹配试验：88/120 到 116/120，+23.33 个百分点，exact McNemar `p=1.94e-6` | 在该冻结仿真矩阵中，免训练的观测年龄后缀选择有效 |
| 跨任务集验证 | Object、Goal、LIBERO-10：300 rollouts / 150 pairs，83/150 到 141/150 | 将同一模型族和仿真套件内的确定性延迟证据扩展至三个任务集 |
| RTC-style continuation | 300 rollouts / 100 matched triplets：baseline 96/100，hard projection 97/100，RTC guidance 97/100 | 没有任务成功率优势；motion seam 仅为探索性过程指标 |
| 终端制动不变量修复 | 270 个成对离线案例：已注册约束从 264/270 提升到 270/270，6 个旧冲突全部解决，0 个回归 | 将冻结的 `pi0.5` 响应送入 Panda 适配器；不主张任务成功率或硬实时 |
| 异步 Panda 闭环 | 27 个 CPU 墙钟案例：制动不变量模式 9/9 通过物理谓词，突停违规 0，修复预算超限 0；legacy 为 266 次突停，unguarded 为 211 次 | 使用 scripted 非学习策略验证双相机、策略 worker、调度、修复和力矩控制集成；不是学习策略效果或实体安全认证 |
| Clearance-backed swept 审计 | 三个场景 72 条固定 seed 边：相对更密采样 oracle 的 false-safe 为 0 | 静态障碍保守审计；自碰撞和连续实体安全仍不在范围内 |

完整协议、验证器、统计和研究限制见[结果说明](docs/RESULTS.md)。

## 不应声称的内容

- 没有训练或微调 `pi0.5`。
- 官方 checkpoint 结果仅覆盖 LIBERO 仿真。
- 时序实验使用 blocking inference 加 simulator catch-up，不是操作系统级硬实时控制。
- Panda guard 不是碰撞安全认证，也不能证明 `pi0.5` 已控制 Panda。
- 没有集成 Isaac Lab、ROS2、真实 Franka Panda 或安全 PLC。

## 本地 CPU 快速开始

本地 Panda 路径需要 Git 和 CPython 3.10，不需要 GPU、OpenPI 服务器或
实体机器人。在全新克隆的 Windows 仓库根目录运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_local.ps1
& '.\.venv\Scripts\python.exe' -m armbench doctor
& '.\.venv\Scripts\python.exe' -m armbench mujoco-view --scenario narrow_gate
```

在 Ubuntu 或其他支持的 Linux 发行版中运行：

```bash
./scripts/setup_local.sh
./.venv/bin/python -m armbench doctor
./.venv/bin/python -m armbench mujoco-view --scenario narrow_gate
```

安装脚本会安装 CPU 依赖，并将固定版本的 Panda 模型检出到 `.cache/`。
只有需要轻量 OpenPI 客户端时才使用 PowerShell 的 `-WithVla` 或 Linux
的 `--with-vla`；该选项不会下载 `pi0.5` checkpoint。手动安装、模型路径
覆盖和无桌面环境说明见[本地安装与支持](docs/LOCAL_SETUP_ZH.md)。

Windows 上可以先执行一个有边界的本地验收：

```powershell
.\scripts\vla_demo.cmd -CheckOnly
```

下面的无模型异步验收会让阻塞策略调用在独立 worker 中运行，同时检查控制
侧是否继续按周期 tick：

```powershell
& '.\.venv\Scripts\python.exe' -m armbench vla-async-smoke
```

这只是组件级 scripted 证据，不是新的 `pi0.5` 任务成功率结果。设计边界见
[非阻塞运行时验收](docs/ASYNC_RUNTIME_ZH.md)。

下面的 CPU-only 动作适配验收会把脚本生成的 LIBERO 风格 `H x 7` 末端动作，
通过 MuJoCo Panda hand Jacobian 转为关节速度，并继续经过现有 guard：

```powershell
& '.\.venv\Scripts\python.exe' -m armbench vla-panda-adapter-smoke
```

它补上的是组件级动作语义边界，不运行 `pi0.5`，也不构成端到端部署证据。
实现与限制见 [LIBERO 到 Panda 的笛卡尔动作适配器](docs/PANDA_CARTESIAN_ADAPTER_ZH.md)。

下一步 CPU-only 验收在同一批冻结响应上，成对比较旧的逐步 guard 与轨迹级终端制动
不变量修复：

```powershell
& '.\.venv\Scripts\python.exe' -m armbench vla-panda-braking-repair `
  evidence\pi05_rtc_overlap_primary_v3_seed_20260807_001\evaluation `
  --output-directory results\pi05_panda_braking_repair_90_001 `
  --chunks 90 --selection-seed 20260807

& '.\.venv\Scripts\python.exe' -m armbench `
  vla-panda-braking-repair-validate `
  results\pi05_panda_braking_repair_90_001 `
  --source-directory `
  evidence\pi05_rtc_overlap_primary_v3_seed_20260807_001\evaluation
```

仓库中的[修复报告](reports/pi05_panda_braking_repair_90_001/summary.md)显示：270/270
案例满足已注册约束，6 个旧的避碰/加速度冲突全部解决，且没有回归。它仍是成对的
离线诊断，不重新运行 `pi0.5`，不形成 Panda 反馈闭环，也不是物理安全或硬实时证明。
可用 `mujoco-view` 查看 `raw_positions`、`legacy_positions` 和 `repair_positions`，
完整方法见[延迟有界的终端制动不变量动作修复](docs/PI05_PANDA_BRAKING_REPAIR_ZH.md)。

本地异步闭环阶段进一步把实时双相机采集、阻塞策略 worker、观测年龄后缀选择、
deadline 回退、制动修复和力矩控制 Panda 物理执行接到同一个控制循环中。内置策略
是 scripted、非学习策略，因此整条运行时可以只用 CPU 验收：

```powershell
& '.\.venv\Scripts\python.exe' -m armbench vla-panda-async-run `
  --output-directory results\async_panda_quick `
  --scenario single_block --quick --deadline-ms 400

& '.\.venv\Scripts\python.exe' -m armbench `
  vla-panda-async-validate results\async_panda_quick
```

每个案例都会保留墙钟事件与 MuJoCo 实测 trace。运行时添加 `--videos` 可以在计时
结束后渲染 MP4，也可以用 `mujoco-view` 回放 `actual_positions`。这个阶段补齐的是
本地运行时/控制反馈闭环；它没有执行学习式 VLA checkpoint，也不构成硬实时或
真机安全证明。详见[异步 Panda 闭环运行时](docs/ASYNC_PANDA_CLOSED_LOOP_ZH.md)。

保留的 [27 案例 v2 报告](reports/async_panda_closed_loop_400ms_20mm_v2_001/summary.md)
覆盖五档固定延迟、jitter、响应丢失、负载和持续动作故障。制动不变量模式记录到
0 次突停违规、9/9 条 trace 满足项目物理谓词，修复 P95 为 5.99 ms、最大值为
11.47 ms。它只在 1/9 个条件中到达目标，因此报告明确保留了安全/进度代价和本地
CPU 吞吐限制，而不是用汇总成功率掩盖。

## 验收已保存结果

以下命令只核验保留的实验数据并重建离线 dashboard，不重跑模型推理，也不需要 GPU：

```powershell
.\scripts\measured_age_confirmatory_acceptance.cmd
.\scripts\rtc_primary_acceptance.cmd
```

通过[证据目录](docs/EVIDENCE_CATALOG.md)可以明确区分正式结果、pilot、集成门槛、
被拒绝的旧实验和 scripted 运行时检查。[机器可读版本](docs/evidence_catalog.json)
将每个保留 artifact 连接到结果、协议、manifest、原始审阅文件、验证命令和明确的
主张边界。

安装、远程 OpenPI 执行、调试与环境支持请从[文档索引](docs/README.md)进入。

## 仓库结构

```text
src/armbench/        Panda 规划、控制、MuJoCo 与运行时代码
integrations/openpi/ 官方 checkpoint 评测、分析与补丁
tests/               单元与集成测试
scripts/             启动和验收命令
docs/                设计、操作、协议与审计记录
evidence/            保留的实验 artifact
reports/             根据证据生成的离线 dashboard
```

ArmBench 使用 MIT License；上游软件、模型与资产保持其原始许可证，见[第三方声明](THIRD_PARTY_NOTICES.md)。
