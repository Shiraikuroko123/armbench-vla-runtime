[English](README.md) | 简体中文

# ArmBench：面向 VLA 动作块的时序对齐与 RTC 运行时

ArmBench 是一个 VLA 部署与评测工程，研究异步推理返回的 action chunk
在真正开始执行时已经部分过期的问题。项目不训练 pi0.5，而是在模型与
机器人仿真器之间实现训练无关的运行时调度、动作约束、故障注入、配对实验
和可复现证据链。

当前仓库同时保留两条明确分开的路径：

- Linux/NVIDIA 正式实验路径：运行 Physical Intelligence 官方
  pi05_libero checkpoint，在 LIBERO 上评测时序对齐、projected overlap
  和 RTC-style VJP guidance。
- Windows 本地工程路径：在 MuJoCo Franka Panda 上验证 OpenPI/DROID
  协议、动作块保护、闭环反馈、碰撞与故障处理。该路径的内置策略明确标记为
  scripted_non_learned，不能冒充真实 VLA 结果。

两条路径的 observation、action semantics 和实验结论不混用。

## 项目解决了什么问题

VLA 通常一次输出一段动作，而不是单步动作。若推理消耗 d 个控制周期，返回
动作块的前 d 个动作在执行时已经对应过去的观测。直接从索引 0 执行会造成
时序错位、重复运动甚至任务失败。

ArmBench 的 latency_aligned dispatcher 会跳过与观测年龄相匹配的过期前缀，
只执行后面的有效动作，不修改 checkpoint 权重。measured-age 扩展进一步使用
客户端实际测得的端到端观测年龄，而不是依赖隐藏的注入延迟标签。

项目还研究了第二条、彼此独立的路线：进入 pi0.5 flow sampler 内部，对已经
承诺的 overlap 动作分别进行 hard projection 或 RTC-style denoised-action
VJP guidance。该路线研究动作连续性，不应与前缀跳过调度混称为同一种方法。

## 系统边界

~~~text
相机图像 + 语言指令 + 机器人状态
                  |
                  v
       官方 pi0.5 / OpenPI server
                  |
                  v
              action chunk
                  |
                  v
观测年龄对齐 / overlap guidance / deadline 与状态检查
                  |
                  v
       LIBERO 或 MuJoCo 机器人执行
                  |
                  v
视频、逐动作记录、统计分析、manifest 与离线 dashboard
~~~

ArmBench 位于策略与执行器之间。pi0/pi0.5 负责生成动作，LIBERO、MuJoCo
或 Isaac Lab 负责模拟动作产生的物理结果，它们不是互相替代的软件。

## 已实现的核心能力

- 官方 pi05_libero checkpoint 的容器化评测、checkpoint/source attestation
  和完整 LIBERO 任务矩阵。
- 训练无关的 action-chunk 时序对齐、短 chunk 拒绝、deadline 回退和
  measured-age 调度。
- pi0.5 flow sampler 内的 hard projected overlap 与 RTC-style VJP guidance，
  不微调模型权重。
- 配对随机种子、显式 flow sampling noise、McNemar 检验、bootstrap 区间、
  Holm 多重比较校正和任务块统计。
- 只读验证器、SHA-256 manifest、失败关闭规则、视频哈希绑定和可离线重建的
  可视化 dashboard。
- MuJoCo Franka Panda 双相机观测、15x8 DROID 动作协议、速度/加速度/关节
  约束、mesh-edge 检查、动作 backtracking 和 torque PD 执行。
- OpenPI WebSocket/MessagePack 客户端以及 wrong-shape、nonfinite、
  disconnect、timeout、冻结相机、状态跳变和 deadline miss 故障注入。
- 在线 receding-horizon 循环：每轮重新采集真实仿真状态和两路图像，再请求
  下一段动作。

## 当前最重要的实验结果

### 1. corrected-v3 RTC overlap 主实验

corrected-v3 使用 10 个 LIBERO-10 任务、每任务 5 个初始状态、2 个新采样
种子和 3 种方法，共 300 个 rollout、100 个匹配 triplet。

| 方法 | 成功率 | motion seam 均值 | gripper seam 均值 |
| --- | ---: | ---: | ---: |
| Unconditioned overlap | 96/100 | 0.106729 | 0.053754 |
| Hard projected overlap | 97/100 | 0.083089 | 0.055490 |
| RTC-guided overlap | 97/100 | 0.087204 | 0.043190 |

两种 conditioned 方法相对 baseline 都只增加 1 个百分点，raw/Holm exact
McNemar 均为 p=1.0，因此不能宣称任务成功率提升。

探索性的 motion seam 差异为：

- hard projection：-0.023640，任务块 95% CI
  [-0.028187, -0.019207]；
- RTC guidance：-0.019524，任务块 95% CI
  [-0.023128, -0.016142]。

seam 是过程指标，不是碰撞安全、任务成功率或真实部署有效性的证明。

#### 为什么 v2 作废而 v3 有效

实验阶段在工程目标上是层层递进的，但证据有效性取决于配对不变量。v2 复用
同一个 LIBERO environment；后续审计发现任务 3、8、9 在 reset 后仍残留不同
视觉状态。虽然 declared state、prompt、sampling key 和 sampling noise
相同，三种方法在 query-0 接收到的两幅策略图像及其动作却不相同。

这破坏了“唯一变化是被比较方法”的因果前提。v2 不是因为数值与 v3 不一致
而被删除，而是因为配对设计无效而被排除。corrected-v3 为每个 rollout
创建并关闭一个新 environment，并在写入根 manifest 前强制核对以下四类
query-0 哈希：

- policy input；
- response action；
- sampling key；
- sampling noise。

v3 重新运行完整 held-out 矩阵，没有合并、筛选或修补 v2 结果。详细中文
验收说明见
[RTC corrected-v3 验收指南](docs/RTC_OVERLAP_PRIMARY_V3_ACCEPTANCE_ZH.md)。

### 2. pi0.5-LIBERO 确认性时序对齐实验

冻结实验覆盖全部 10 个 LIBERO Spatial 任务、每任务 5 个初始状态、两种
dispatch mode 和 0/100/200 ms 三档确定性延迟，共 300 个 rollout。

| 延迟 | Async 成功 | Aligned 成功 | 配对差异，bootstrap 95% CI | McNemar raw / Holm |
| ---: | ---: | ---: | ---: | ---: |
| 0 ms | 49/50 | 50/50 | +2 点 [0, +6] | 1.000 / 1.000 |
| 100 ms | 41/50 | 48/50 | +14 点 [+2, +26] | 0.0654 / 0.1309 |
| 200 ms，主条件 | 18/50 | 50/50 | +64 点 [+50, +76] | 4.66e-10 / 1.40e-9 |

该结论仅支持“在 LIBERO 仿真中，面对确定性注入的 200 ms 延迟，跳过过期
动作前缀优于从索引 0 执行”。它不是硬实时保证或真机结论。

### 3. measured-age held-out 确认实验

240-rollout held-out 实验将 response jitter 和显式 10x32 pi0.5 flow
sampling noise 在两种模式间配对。成功率从 88/120 提升到 116/120：

- 配对差异 +23.33 个百分点；
- pair bootstrap 95% CI [+15.00, +31.67]；
- 32/4/84 wins/losses/ties；
- exact McNemar p=1.94e-6；
- mean policy queries 从 22.75 降到 15.14。

实验仍然采用 blocking inference 和响应后的 simulator catch-up，不代表
真正独立运行的控制与推理线程。

### 4. 跨 suite 外部验证

冻结同一 checkpoint、200 ms 延迟和五步执行 horizon 后，在 LIBERO Object、
Goal 和 LIBERO-10 上额外运行 300 个 rollout：

| Suite | Async | Aligned | 配对差异，bootstrap 95% CI | McNemar raw / Holm |
| --- | ---: | ---: | ---: | ---: |
| LIBERO Object | 25/50 | 49/50 | +48 点 [+34, +62] | 8.05e-7 / 2.41e-6 |
| LIBERO Goal | 35/50 | 47/50 | +24 点 [+10, +38] | 0.00418 / 0.00418 |
| LIBERO-10 | 23/50 | 45/50 | +44 点 [+26, +60] | 2.74e-5 / 5.49e-5 |

三项预先指定的 suite-level 检验都在 Holm 校正后拒绝零假设。不能把该结果
外推到其他 checkpoint、真实网络抖动或真实机器人。

## 最快的可视化验收

所有正式实验已经保存到仓库或 GitHub Release。验收不需要重新租 GPU、
训练模型或重跑 rollout。

### RTC corrected-v3

在任意 Windows 目录运行：

~~~powershell
D:\arm-planning-control-project\project\scripts\rtc_primary_acceptance.cmd
~~~

只验证、不自动打开浏览器：

~~~powershell
D:\arm-planning-control-project\project\scripts\rtc_primary_acceptance.cmd -NoOpen
~~~

有效输出应包含：

~~~text
valid=true
rollouts=300
triplets=100
failure_videos_verified=10
tasks=10
~~~

生成的 dashboard 位于
[reports/pi05_rtc_overlap_primary_v3_300_001/index.html](reports/pi05_rtc_overlap_primary_v3_300_001/index.html)。

### 五分钟面试验收

~~~powershell
D:\arm-planning-control-project\project\scripts\interview_acceptance.cmd
~~~

该入口重新验证正式时序对齐证据及 300 个视频，然后打开 baseline/aligned
并排 dashboard。

### measured-age 确认实验

~~~powershell
D:\arm-planning-control-project\project\scripts\measured_age_confirmatory_acceptance.cmd
~~~

它会验证 271 个根文件、261 个嵌套 evaluation 文件和全部 240 个视频，
重新计算统计量并生成 120 对视频的离线页面。以上命令都支持 -NoOpen。

## pi0、pi0.5、Isaac Lab 与本项目的区别

| 组件 | 作用 | 本项目是否使用 |
| --- | --- | --- |
| pi0 / pi0.5 | 从图像、语言和状态生成 action chunk 的学习式 VLA | 是，正式实验使用官方 pi05_libero checkpoint |
| OpenPI | 官方模型代码、checkpoint、transforms 与远程协议 | 是，固定版本并验证 |
| LIBERO | 官方 pi0.5 使用的操作基准 | 是 |
| MuJoCo | CPU 可运行的刚体仿真器 | 是，本地 Panda 工程路径 |
| Isaac Gym | NVIDIA 旧版 GPU 仿真/RL 栈 | 否 |
| Isaac Lab | NVIDIA Omniverse 机器人仿真与训练框架 | 否 |
| ArmBench | 位于 VLA 与仿真执行器之间的运行时及评测层 | 本仓库 |

没有使用 Isaac Lab 不代表没有使用 VLA。Isaac Lab 适合 GPU 并行 rollout、
RL 和更大规模动力学实验；当前项目的主要贡献是 VLA 运行时、真实 pi0.5
闭环评测和证据工程，而不是训练策略。若申请 RL/大规模仿真岗位，Isaac Lab
或 RL 扩展仍是后续工作。

## 支持的设备

| 层 | 已验证或要求的环境 |
| --- | --- |
| 本地 guard benchmark | Windows、Python 3.10.8、Intel i9-12900H、Intel Iris Xe |
| MuJoCo physics/rendering | CPU + OpenGL，不要求 NVIDIA GPU 或 CUDA |
| OpenPI 客户端 | Windows 本机，WebSocket + MessagePack |
| pi0/pi0.5 server | 单独的 Ubuntu/NVIDIA 机器；官方说明推理需要超过 8 GB VRAM |
| Franka Panda 真机 | 尚未实现 ROS2、libfranka、标定、watchdog 或安全 PLC |
| 其他机器人本体 | 不能直接即插即用，需要修改 observation/action transform 和 MJCF |

已经保存的 dashboard 与证据验收只需要本地 CPU。只有重新运行官方
pi0.5 rollout 时才需要 NVIDIA GPU。

## Windows 本地配置

以下命令假定工作区位于 D:\arm-planning-control-project。不要在
C:\WINDOWS\system32 中使用相对路径寻找虚拟环境。

~~~powershell
$ArmbenchWorkspace = 'D:\arm-planning-control-project'
Set-Location $ArmbenchWorkspace
git clone --filter=blob:none --no-checkout https://github.com/google-deepmind/mujoco_menagerie.git '.\upstream\mujoco_menagerie'
git -C '.\upstream\mujoco_menagerie' sparse-checkout init --cone
git -C '.\upstream\mujoco_menagerie' sparse-checkout set franka_emika_panda
git -C '.\upstream\mujoco_menagerie' checkout 71f066ad0be9cd271f7ed58c030243ef157af9f4
py -3.10 -m venv '.venv'
$ArmbenchPython = Join-Path $ArmbenchWorkspace '.venv\Scripts\python.exe'
& $ArmbenchPython -m pip install --editable '.\project[test,vla]'
~~~

如果环境已经配置好，可以从任意目录使用绝对路径：

~~~powershell
& 'D:\arm-planning-control-project\project\scripts\vla_demo.cmd' -CheckOnly
& 'D:\arm-planning-control-project\project\scripts\vla_demo.cmd'
& 'D:\arm-planning-control-project\project\scripts\vla_demo.cmd' -Formal
~~~

- CheckOnly：检查依赖、场景、相机、协议、guard 和 artifact；
- 默认：运行单场景 smoke，不生成视频；
- Formal：运行两个场景并生成三个 MP4。

运行全部测试：

~~~powershell
& 'D:\arm-planning-control-project\.venv\Scripts\python.exe' -m pytest -q
~~~

## 调试与产物

先阅读 [调试指南](docs/DEBUGGING.md)。每次 online run 的关键产物包括：

| 问题 | 查看位置 |
| --- | --- |
| 策略实际收到了什么 | observations 图片、camera hash、VLAObservation.to_openpi_droid |
| server 是否返回正确 shape | probe.json、per_chunk.csv、OpenPIPolicyClient.infer |
| 哪个 chunk 超过 deadline | per_chunk.csv |
| 状态是否在每轮重新采集 | per_chunk.csv 与 NPZ |
| 哪个动作为什么被修改 | per_action.csv 的 scale、reason、raw/executed action |
| 预测路径是否通过检查 | ActionChunkGuard.guard 与 predicted_positions |
| 物理执行是否发生接触 | per_case.csv、MP4 与 actual_positions |

典型目录：

~~~text
results/<run_id>/
  config.json
  environment.json
  overview.png
  summary.md
  aggregate.json
  per_case.csv
  per_chunk.csv
  per_action.csv
  observations/*.png
  videos/*.mp4
  <case>.npz
  run.log
~~~

正式 evidence 目录是只读证据，不要直接修改其中的 CSV、JSON、manifest
或视频。调试时使用新的 run ID。

## 仓库阅读顺序

1. 本页：理解问题、方法、结果和边界。
2. [RTC corrected-v3 中文验收](docs/RTC_OVERLAP_PRIMARY_V3_ACCEPTANCE_ZH.md)：
   理解为什么 v2 作废以及如何一键验证 v3。
3. [完整结果快照](docs/RESULTS.md)：核对所有 provenance、统计量和限制。
4. [RTC/pi0.5 集成](docs/RTC_PI05_INTEGRATION.md)：理解 reverse-time flow、
   VJP 和 scheduler contract。
5. [调试指南](docs/DEBUGGING.md)：运行本地 MuJoCo/OpenPI 路径。
6. [英文完整 README](README.md)：查询全部低层命令和历史 artifact。

## 结论边界

- 项目评测官方 pi0.5 checkpoint，但没有训练或微调 pi0.5。
- 正式 checkpoint 实验发生在 LIBERO；本地 guard、camera、wire-fault
  和 replay 的旧 MuJoCo artifact 使用非学习式动作源。
- 所有结果仍是仿真结果，不是真机实验。
- blocking inference、注入延迟和 measured-age catch-up 不构成操作系统
  hard real-time 保证。
- seam 改善不等于任务成功、安全性或真实部署改善。
- MuJoCo mesh/interpolation 检查不是解析连续碰撞检测或形式化安全证书。
- 当前没有 Isaac Lab、RL 训练、ROS2 或真实 Franka 部署。

## 简历表述

在本人已经复现实验并理解代码的前提下，可以使用以下中文表述：

> 构建 OpenPI 兼容的 VLA 运行时与可审计 pi0.5-LIBERO 评测系统；实现训练
> 无关的 measured-age action-chunk 时序对齐，并对 response jitter 与
> pi0.5 flow sampling noise 进行配对。在冻结的 240-rollout held-out
> 实验中，将成功率从 88/120 提升到 116/120（+23.3 个百分点，exact
> McNemar p=1.94e-6；whole-task bootstrap 95% CI [+10.8,+38.3]），同时
> 将平均 policy query 从 22.75 降到 15.14；通过 checkpoint/source
> attestation、不可变 manifest、独立 validator 和一键视频 dashboard
> 保存并验收正式实验。

应写“评测并集成官方 pi0.5-LIBERO checkpoint”，不能写“训练了 pi0.5”
或“已部署真实机器人”。这是 VLA runtime/evaluation 项目，不是 VLA
训练项目。
