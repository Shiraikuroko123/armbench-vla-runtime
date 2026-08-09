[项目网页](https://shiraikuroko123.github.io/armbench-vla-runtime/) | [English](README.md) | [文档索引](docs/README.md) | [代码导读](docs/CODE_WALKTHROUGH_ZH.md) | [改进路线与采购表](docs/ROADMAP_ZH.md)

[![CPU CI](https://github.com/Shiraikuroko123/armbench-vla-runtime/actions/workflows/ci.yml/badge.svg)](https://github.com/Shiraikuroko123/armbench-vla-runtime/actions/workflows/ci.yml)

# ArmBench

ArmBench 是一个面向动作块式视觉-语言-动作策略的运行时与评测平台。它研究的不是重新训练一个 VLA，而是 VLA 已经输出一段未来动作后，如何在推理延迟、状态变化和异常响应下可信地进入控制回路。

本文首次使用的规范名称为：**Physical Intelligence 的 `pi0.5`（pi-zero-point-five）视觉-语言-动作模型，简称 `pi0.5 VLA`**。`OpenPI` 是项目使用的模型与推理实现栈，不是另一个模型名称。

## 项目由什么组成

仓库包含两条分别验证的路径：

- **七自由度 Panda 执行基座**：在本地 MuJoCo 中实现规划、受限轨迹跟踪、协议验证、故障注入和动作级检查。
- **`pi0.5 VLA` 时序评测路径**：通过 OpenPI 调用官方 checkpoint，在 LIBERO 闭环仿真中研究 action chunk 的推理时序问题。

二者共享运行时契约、观测/状态封装、响应校验、日志、视频和证据工具。当前已经新增一条内容可校验的 integration gate：真实 `pi0.5`-LIBERO 响应通过显式 Cartesian 动作适配器进入 Panda 异步闭环；但这仍不是任务对齐的 Panda 正式评测、真机部署或安全认证。

项目名称的变化对应工程层次的推进，而不是放弃原来的七自由度项目：原有采样规划、碰撞检查与轨迹跟踪成为 Panda 执行基座；ArmBench 在它前面增加动作块时序、响应校验、原子发布与可行回退，因此研究对象从“如何生成并跟踪轨迹”扩展为“VLA 动作如何可信地进入控制器”。

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
                 (规划、OSQP 投影、连续碰撞、停止不变量、
                  力矩执行和故障注入)
```

Panda 与 LIBERO 的动作契约不同，实验结果不会混合统计。完整设计与边界见[架构与主张边界](docs/PROJECT_ARCHITECTURE_ZH.md)。

## 已验证的证据

| 研究 | 证据 | 可以得出的结论 |
| --- | --- | --- |
| 真实 `pi0.5` 到 Panda 集成门 | 官方 OpenPI `pi05_libero` checkpoint，35 个响应被接收，策略时延平均/P95 为 82.75/89.56 ms，推理期间控制 tick 为 290/311，注册仿真违规为 0 | 内容校验过的 H×7 响应穿过 Panda H×8 适配器与异步 runtime；探针目标未到达，因此只证明接线与运行时机制，不证明任务能力 |
| 观测年龄时序对齐 | 官方 `pi0.5`-LIBERO Spatial，120 组匹配试验：88/120 到 116/120，+23.33 个百分点，exact McNemar `p=1.94e-6` | 在该冻结仿真矩阵中，免训练的观测年龄后缀选择有效 |
| 跨任务集验证 | Object、Goal、LIBERO-10：300 rollouts / 150 pairs，83/150 到 141/150 | 将同一模型族和仿真套件内的确定性延迟证据扩展至三个任务集 |
| RTC-style continuation | 300 rollouts / 100 matched triplets：baseline 96/100，hard projection 97/100，RTC guidance 97/100 | 没有任务成功率优势；motion seam 仅为探索性过程指标 |
| 终端制动不变量修复 | 270 个成对离线案例：已注册约束从 264/270 提升到 270/270，6 个旧冲突全部解决，0 个回归 | 将冻结的 `pi0.5` 响应送入 Panda 适配器；不主张任务成功率或硬实时 |
| Panda 集成 supervisor | 27/27 个注册故障结果可重算：12 个接受、6 个已验证制动、7 个 hold、2 个不可恢复停止；拒绝动作块均未泄漏部分前缀 | 将 OSQP 运动学投影、连续碰撞证书和停止不变量组合为原子 CPU 参考链；使用 scripted 动作，不是硬实时 |
| 带保障的 Panda 任务执行 | nominal 与 0.5 kg/80 ms 条件下 2/2 到达 MuJoCo 目标；351/351 条运动边和停止边界通过证书，注册的接触/限位/力矩饱和均为 0 | 先离线保障，再执行力矩控制关节路点任务；不是学习式 VLA、物体操作或真机结果 |
| 异步 Panda 闭环 | 27 个带 clearance-backed swept 静态障碍检查的 CPU 墙钟案例：制动不变量模式 9/9 通过物理谓词，突停违规 0，修复预算超限 0；legacy 为 311 次突停，unguarded 为 289 次 | 使用 scripted 非学习策略验证双相机、策略 worker、调度、修复和力矩控制集成；不是学习策略效果或实体安全认证 |
| Clearance-backed swept 审计 | 三个场景 72 条固定 seed 边：相对更密采样 oracle 的 false-safe 为 0 | 静态障碍保守审计；自碰撞和连续实体安全仍不在范围内 |
| 连续自碰撞审计 | Panda 关节空间 72 条固定 seed 边：70 条端点均有效，相对 0.002 rad 采样 oracle 的 false-safe 为 0，保守拒绝 21 条 | 针对线性插值的 fail-closed 几何审计；不是实体安全或硬实时证书 |
| MuJoCo 动力学制动审计 | 0/0.5/1 kg 负载、0.5/1/2 倍阻尼和 5 类初速度共 45 条条件，45/45 通过逆动力学与连续碰撞边检查 | 编译 MuJoCo Panda 的采样可行性证据；不是闭环跟踪、硬实时或真机急停证明 |
| Provider-neutral 动作契约 | OpenVLA-OFT 命名的 CPU fixture：`6x7` 转 `6x8`，精确绑定观测，5/5 类语义冲突均被拒绝 | 仅证明第二模型族 ABI；没有执行 OpenVLA-OFT checkpoint |
| LeRobot 风格执行器边界 | 5 帧确定性重放：执行 3 条命令，拒绝过期观测，保持锁存，并在显式 reset 后恢复 | 仅证明帧接口与软件 watchdog；未运行官方 LeRobot 或实体机器人 |
| 官方 LeRobotDataset round-trip | 隔离 `lerobot==0.4.4`、数据集代码 `v3.0`，3 帧中的图像、state、action、task、timestamp 均由官方 loader 重载并匹配 | 仅证明 Panda Hx8 数据集序列化边界；没有策略、驱动、SO-101 转换或真机 |

完整协议、验证器、统计和研究限制见[结果说明](docs/RESULTS.md)。

已经保存的 live bundle 可在无 GPU 环境中独立验收。该命令核对 checkpoint
身份、响应来源、事件、时钟、状态轨迹和完整 MP4，但不会重新运行模型推理：

```powershell
& '.\.venv\Scripts\python.exe' -m `
  integrations.openpi.validate_live_panda_smoke `
  '.\evidence\g01_live_panda_smoke_final_001' --json
```

## 不应声称的内容

- 没有训练或微调 `pi0.5`。
- 官方 checkpoint 结果仅覆盖仿真。LIBERO 研究遵循官方任务协议；Panda G01
  是单独的集成探针，并如实记录 `target_reached=false`。
- 时序实验使用 blocking inference 加 simulator catch-up，不是操作系统级硬实时控制。
- Panda 集成保障中的 2/2 到达任务仍来自 scripted RRT-Connect，不能借用 G01
  将其改写为 `pi0.5` 的任务结果；完整 horizon 检查也不能替代实体碰撞安全认证。
- 没有集成 Isaac Lab、ROS2、真实 Franka Panda 或安全 PLC。
- LeRobot-style bridge 仍是 CPU-only 的内存接口契约；单独的官方
  `LeRobotDataset` round-trip 只验证数据集读写，不等于策略、驱动或真机集成。
- 动力学制动审计使用编译 MuJoCo Panda 模型和采样的逆动力学，不能替代真实
  负载标定、硬实时或急停验证。

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

当前最完整的本地 CPU 验收会重算 27 案例同步故障矩阵、17 案例异步原子发布矩阵，
并重新运行两条 Panda 任务的规划、动作保障和 MuJoCo 物理。脚本会自行解析仓库和
Python 路径。在仓库根目录执行；若从其他目录启动，则向 `-File` 传入脚本的绝对路径：

如果需要一次检查全部已保存的本地证据，运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  '.\scripts\accept_cpu.ps1'
```

该入口会依次验证连续静态/自碰撞、动力学制动、provider 语义契约、LeRobot
watchdog、冻结 `pi0.5` 响应回放、Panda 任务、异步闭环与证据目录，并把不纳入 Git
的结果写入 `output/cpu_acceptance/summary.md` 和 `summary.json`。检测到隔离的官方
`lerobot==0.4.4` 环境时还会自动加入官方 Dataset round-trip；没有该环境时只将此项
标为 skipped，不影响其余 CPU 验收。提交前可追加 `--full-tests` 运行完整 pytest。

这是一条证据重算入口，不是 `pi0.5` 到 Panda 的端到端部署、真机实验、硬实时保证或
安全认证。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  '.\scripts\accept_integrated_panda.ps1'

powershell -NoProfile -ExecutionPolicy Bypass -File `
  '.\scripts\accept_integrated_panda.ps1' `
  -Visualize -Case narrow_gate_payload_delay_goal
```

第一条是数字验收，第二条还会播放保存的 MuJoCo 实测轨迹。方法、结果、固定夹爪
allowed-collision 规则和耗时边界见[Panda 集成动作保障链](docs/INTEGRATED_PANDA_ASSURANCE_ZH.md)。

新增的异步矩阵把 mock、冻结响应和 OpenPI-compatible 接口 fixture 接入同一条
provider -> policy worker -> 集成保障 worker -> 原子发布链。17/17 个注册结果匹配，
6 个完整计划被发布、10 个进入 hold、1 个进入不可恢复停止，拒绝结果暴露的部分动作
前缀为 0。它不执行学习式 checkpoint；完整调用图、调试断点和主张边界见
[Panda 异步动作保障 CPU 收口](docs/CPU_RUNTIME_COMPLETION_ZH.md)。

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

这个 CPU 命令补上的是组件级动作语义边界，不运行 `pi0.5`，也不构成端到端部署证据。
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

保留的 [27 案例 v3 报告](reports/async_panda_closed_loop_400ms_20mm_v3_001/summary.md)
覆盖五档固定延迟、jitter、响应丢失、负载和持续动作故障。制动不变量模式记录到
0 次突停违规、9/9 条 trace 满足项目物理谓词，修复 P95 为 7.81 ms、最大值为
19.01 ms。静态障碍边使用记录在 provenance 中的 20 mm clearance-backed swept
细分；独立的连续自碰撞审计见下文。它只在 1/9 个条件中到达目标，因此报告明确保留了
安全/进度代价和本地 CPU 吞吐限制，而不是用汇总成功率掩盖。

独立的[连续自碰撞审计](reports/mujoco_self_collision_audit_001/summary.md)对 72 条
固定 seed Panda 关节空间边与 0.002 rad 采样 oracle 对照：70 条边的两个端点均有效，
false-safe 为 0，保守拒绝 21 条。无需 GPU 即可重新验收：

```powershell
& '.\.venv\Scripts\python.exe' -m armbench mujoco-self-collision-validate `
  reports\mujoco_self_collision_audit_001
```

数字验收后，可以在 MuJoCo viewer 中播放固定的“边中间自碰撞”机制控制边：

```powershell
& 'D:\arm-planning-control-project\.venv\Scripts\python.exe' -m armbench mujoco-self-collision-view `
  'D:\arm-planning-control-project\project\reports\mujoco_self_collision_audit_001' `
  --stratum known_intermediate --edge-index 0 --speed 0.75 --skip-validation
```

接触点叠加只提供运动学可视证据，正式结论仍以 manifest 保护的 validator 为准。

证书只覆盖关节空间线性插值和编译后的 MuJoCo 几何，不是实体安全或硬实时结果。

Provider-neutral CPU 审计演示第二个 action-chunk 模型族如何进入现有 runtime，
并避免把所有 `Hx7` 张量误认为相同动作空间：

```powershell
& '.\.venv\Scripts\python.exe' -m armbench vla-provider-audit-validate `
  reports\provider_contract_audit_001
```

仓库中的 fixture 使用 `OpenVLA-OFT` 名称来覆盖该 provider ABI，但它是合成数据，
不是 checkpoint 输出。精确语义门禁和主张边界见
[Provider-neutral 动作契约](docs/PROVIDER_CONTRACT_ZH.md)。

最后一个 CPU-only 边界将 Panda 命令映射为 LeRobot 风格 `add_frame` 字段，在发送前
执行 command watchdog，并导出带哈希清单的 episode；离线重放会重新计算全部决定：

```powershell
& '.\.venv\Scripts\python.exe' -m armbench vla-lerobot-replay `
  reports\lerobot_style_watchdog_001
```

它验证软件接口、过期命令 hold、故障锁存和 reset 路径，但没有使用官方 LeRobot 包，
也没有连接实体机器人。详见
[LeRobot 风格运行时桥接](docs/LEROBOT_RUNTIME_BRIDGE_ZH.md)。

官方 LeRobotDataset 边界在一个隔离环境中单独验收。它固定
`lerobot==0.4.4` 与数据集代码 `v3.0`，导出并用官方 `LeRobotDataset` loader
重载 3 帧 Panda Hx8 episode：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\setup_official_lerobot.ps1

& '.\.venv-lerobot-0.4.4\Scripts\python.exe' -m armbench `
  vla-lerobot-official-smoke `
  --output-directory reports\official_lerobot_roundtrip_001

& '.\.venv-lerobot-0.4.4\Scripts\python.exe' -m armbench `
  vla-lerobot-official-validate reports\official_lerobot_roundtrip_001
```

这不是 SO-101 动作，也不运行学习策略。完整版本约束、字段语义和
manifest 见[官方 LeRobotDataset round-trip](docs/OFFICIAL_LEROBOT_ROUNDTRIP_ZH.md)。

MuJoCo 动力学制动审计则在主 CPU 环境中重放 45 条注册条件：

```powershell
& '.\.venv\Scripts\python.exe' -m armbench `
  mujoco-dynamics-braking-validate reports\dynamics_braking_audit_001
```

它检查限加速度停止轨迹、连续碰撞边和 `mj_inverse` 力矩余量；完整结果与限制见
[Panda 动力学制动审计](docs/DYNAMICS_BRAKING_AUDIT_ZH.md)。

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
