# ArmBench 技术加强路线与资源预算

状态：后续技术迭代的唯一执行表。最后核对日期：2026-08-08。

本文只保留技术工作。网页、措辞、简历包装、宣传材料和一般性仓库美化暂不进入路线。预算均为人民币估算，不含个人时间；云 GPU、硬件套件、运费和汇率会变化，实际购买前需要重新询价。

## 1. 当前资源与结论

| 资源 | 当前情况 | 能完成什么 | 缺口与处理方式 |
| --- | --- | --- | --- |
| CPU | i9-12900H，14 核 20 线程 | MuJoCo、规划控制、QP、碰撞检测、数据验证、官方 LeRobot loader | 足够，不需要购买 CPU 主机 |
| 内存 | 约 16 GB | 当前 CPU 工程与小规模仿真 | 属于最低可用；只有并行仿真/WSL 经常内存不足时才考虑 32 GB，先确认笔记本是否可升级 |
| GPU | Intel 集显，无 CUDA | 只能做渲染与普通 CPU 计算 | `pi0.5`、OpenVLA-OFT、Isaac Lab 和训练必须租 NVIDIA GPU |
| 磁盘 | D 盘约 58 GB、C 盘约 69 GB 可用 | 保存代码和压缩后的核心 evidence | 不适合同时保存多个模型、Docker image 和全部视频；云端使用 200-350 GB 临时盘，必要时再买 1 TB 移动 SSD |
| 实体设备 | 无机械臂、相机和急停 | 只能做仿真与离线验证 | 真机路线最后再采购，不影响前两阶段技术开发 |

当前不需要购买任何东西。第一阶段全部可在本机完成；只有进入真实 checkpoint 运行时才租 GPU。

## 2. 完成等级

| 等级 | 技术含义 | 验收要求 |
| --- | --- | --- |
| L0 | 代码实现 | 单元测试和明确输入输出契约 |
| L1 | 组件验证 | mock、scripted policy 或冻结响应可重复运行 |
| L2 | 真实模型 smoke | 真实 checkpoint 在线产生动作并进入执行循环 |
| L3 | 正式仿真实验 | 冻结协议、多任务/seed、原始 trace、统计和独立 validator |
| L4 | 跨模型或跨仿真 | 至少两个真实 checkpoint 家族，或有明确控制变量的第二仿真器 |
| L5 | 实体实验 | 标定机器人、硬件 watchdog/急停和重复故障注入 |

实现接口不能自动升级为 L2；一次成功视频不能自动升级为 L3。

## 3. 技术加强总表

优先级：P0 为当前必须完成；P1 为核心竞争力阶段；P2 为完成 P1 后的扩展；P3 只在明确论文问题或实验室合作下进行。

### 3.1 本地 CPU 核心

| ID | 技术加强与解决的问题 | 当前状态 | 前置条件 | 所需软件/数据 | 所需计算或设备 | 现金预算 | 开发时间 | 验收标准 | 优先级 |
| --- | --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| C01 | 统一 live/frozen/mock provider，使真实模型与测试后端进入同一运行时 | 已有语义 gate、合成 ABI fixture 和带身份/响应 hash 的 attested live OpenPI provider；尚未接入 Panda loop | 固定 observation、action semantics、checkpoint identity | Python 3.10、`openpi-client`、现有 provider contract | 当前电脑 | ¥0 | 2-4 天 | 三种 backend 可互换；动作语义、模型身份或响应 hash 不符时 fail closed；测试覆盖 worker/guard 链路 | P0 |
| C02 | 独立进程/时钟运行时，使控制在阻塞推理期间继续 tick | 已完成 spawn-safe process worker、latest-only mailbox 和 deadline smoke；正式 LIBERO 仍是 blocking + catch-up | C01；统一 monotonic clock、序列号和 mailbox | `multiprocessing`/websocket、现有 trace schema | 当前电脑先用阻塞 mock 测试 | ¥0 | 3-6 天 | 推理阻塞 0/40/80/160 ms 时控制 tick 不停止；response age、跳步、hold 和 deadline 均可从 trace 重算 | P0 |
| C03 | QP 动作投影，替代单纯缩放/greedy 回溯 | 已完成 OSQP 组件投影与 fail-closed infeasible smoke；尚未接入任务级在线 evaluator | 明确位置、速度、加速度、控制周期和不可行回退 | `OSQP`、`scipy`、NumPy | 当前电脑 | ¥0 | 3-7 天 | 1,000 个固定随机案例满足注册约束；报告 infeasible 率、任务误差、P95/max 求解时延；超预算进入 hold | P0 |
| C04 | 连续静态碰撞与连续自碰，补上当前 sampled self-collision 缺口 | 已完成保守 MuJoCo 连续边 checker 和 72 条静态障碍审计；自碰撞矩阵与任务级接线仍待完成 | 固定 Panda collision pairs、几何简化和独立 oracle | MuJoCo continuous pair-distance checker；可选独立 CCD oracle | 当前电脑 | ¥0-100 | 5-10 天 | 注册边集相对独立 CCD/更密 oracle 为 0 false-safe；静态碰撞和自碰分别报告；性能 P95 可复现 | P0 |
| C05 | 动力学可达停止，回答受限动作是否真的能在扭矩/负载下刹停 | 已完成：45 条注册条件通过 MuJoCo 逆动力学与连续碰撞边验证 | C03；Panda 质量、惯量、扭矩/速度限制和 payload 条件 | MuJoCo；可选 Pinocchio | 当前电脑 | ¥0 | 5-10 天 | 固定初速、payload、摩擦矩阵中，所有接受动作存在限时停止轨迹；报告停止时间/距离和保守拒绝率 | P0 |
| C06 | 官方 LeRobot loader 与 episode round-trip，补上“LeRobot 风格但未使用官方包” | 已完成：隔离 `lerobot==0.4.4`/v3.0 loader 对 3 帧 Panda episode 逐字段 round-trip | 固定一个官方 LeRobot release；保持 Panda 与未来 SO-101 动作语义分离 | 官方 `lerobot`、公开小 fixture | 当前电脑；建议 WSL2/Ubuntu | ¥0 | 2-5 天 | 官方 loader 读取导出 episode；图像、状态、动作、task、时间戳 round-trip 一致；版本被锁定 | P1 |
| C07 | 标准化 fault matrix，统一延迟、jitter、丢帧、旧响应、NaN、断连和模型失配 | 多数故障已有，但分散在多个 benchmark | C01-C05 的统一 runtime | 现有 fault/validator 模块 | 当前电脑 | ¥0 | 2-4 天 | 每类故障有注册输入、预期状态机转移和独立重放；任何旧命令均不能在 reset 后执行 | P1 |
| C08 | MoveIt 2/Servo 或成熟控制器基线，避免只与自研 guard 比较 | 未完成 | C03-C05；Panda URDF/SRDF；ROS2 路径可运行 | ROS2 Humble、MoveIt 2 | WSL 不适合最终实时测试；Ubuntu 环境 | ¥0-300 | 4-8 天 | 同一任务/约束矩阵对比成功率、违规、干预率和控制时延；动作语义映射留存 | P2 |

### 3.2 真实 VLA 与云 GPU

| ID | 技术加强与解决的问题 | 当前状态 | 前置条件 | 所需软件/数据 | 所需云资源 | 现金预算 | 开发/运行时间 | 验收标准 | 优先级 |
| --- | --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| G01 | 官方在线 `pi0.5 -> Panda` smoke，补上当前最关键的端到端断点 | 只有官方冻结响应离线回放；在线 Panda 使用 scripted policy | C01-C02；Panda 双相机 observation adapter；checkpoint attestation | 固定 OpenPI commit、官方 `pi05_libero` checkpoint、Panda MuJoCo | RTX 4090 24 GB；16 vCPU；64 GB RAM；200 GB SSD；6-12 GPU 小时 | ¥50-300 | 2-4 天开发，0.5 天运行 | 至少 10 条真实 checkpoint episode；请求、响应、适配、repair、watchdog、执行和反馈时间戳闭合；无 fixture | P0 |
| G02 | 真正独立时钟的 `pi0.5`-LIBERO pilot，替代 blocking inference + simulator catch-up | 未完成 | C02、G01；仿真与推理分进程；配对 jitter/noise | OpenPI、LIBERO、40-rollout 冻结 pilot 协议 | 4090 24 GB；16 vCPU；64 GB；200 GB；12-30 GPU 小时 | ¥100-600 | 4-8 天 | 仿真在推理期间持续推进；40 rollouts 完整；所有 age、offset、hold 和 failure 可独立重算 | P0 |
| G03 | 独立时钟正式矩阵，确认时序方法是否仍有效 | 未开始 | G02 无基础设施故障且存在可评估干预 | 预注册不少于 100 matched pairs；固定随机源 | 4090 24 GB；30-80 GPU 小时 | ¥300-1,500 | 2-5 天运行与分析 | 全部 assigned rollouts 入统计；成功、deadline、query、干预、时延、CI 和失败分类完整；validator 通过 | P1 |
| G04 | 在线任务级 QP/braking repair，验证满足约束的同时是否保留任务进度 | 只有 270 个离线案例与 scripted Panda loop | C03-C05、G02；候选与基线干预预算可比较 | 真实 `pi0.5` response、统一 fault matrix | 与 G03 共用或追加 20-50 GPU 小时 | ¥200-1,000 | 3-7 天 | 同一 policy 下报告任务成功、约束违规、hold、干预、进度和修复 P95/max；不筛除失败案例 | P1 |
| G05 | 真实 OpenVLA-OFT 原生 smoke，增加第二 checkpoint 家族 | 当前只有 OpenVLA-OFT 命名合成 fixture | 固定官方 commit/checkpoint；先跑原生 LIBERO evaluator | OpenVLA-OFT 官方仓库、模型与 LIBERO 数据 | 最低约 16 GB VRAM，建议 4090 24 GB；200 GB SSD；6-12 GPU 小时 | ¥50-400 | 2-4 天 | checkpoint 内容 hash、真实模型输出、至少一条成功/失败 episode 和原生 evaluator 记录完整 | P1 |
| G06 | OpenVLA-OFT 接入统一 provider/runtime，验证接口真正跨模型 | 未完成 | C01、G05；逐字段核对 frame、dt、rotation、gripper 和 normalization | 官方 action transform、现有 Panda adapter/guard | 16-24 GB GPU；10-30 GPU 小时 | ¥100-600 | 2-5 天 | 真实输出经过精确 semantic gate；任一不兼容字段 fail closed；完成小型在线矩阵 | P1 |
| G07 | 两真实模型家族的冻结对照，建立有限跨模型外部效度 | 未开始 | G03、G06；每个模型使用原生 evaluator | `pi0.5`、OpenVLA-OFT、共同 fault protocol | 24 GB GPU；40-120 GPU 小时；250-350 GB 临时盘 | ¥500-2,500 | 1-2 周 | 每模型至少 20-50 matched pairs；结果按模型分层，不混合动作空间；报告跨模型一致与冲突 | P1 |
| G08 | 忠实直接方法基线，避免把 RTC-style 近似写成论文方法 | 有内部条件化与 RTC-style 实验，但不是完整 RTC/VLASH/Action ControlNet | 对方公开实现/checkpoint 或可复现训练；统一独立时钟 | 固定上游 commit、真实实现所需模型 | 24-80 GB GPU；50-200 GPU 小时 | ¥500-3,000 | 1-3 周 | 真实实现与近似严格分名；在相同任务、噪声、时钟和统计协议下比较 | P2 |

`G01` 是系统连通性证据，不保证本地 Panda 任务成功。`pi05_libero` 的相机分布、控制点和本地场景不完全等同训练环境；任务效果必须通过 G02-G04 的注册实验判断。

### 3.3 不确定性、训练与仿真扩展

| ID | 技术加强与解决的问题 | 当前状态 | 前置条件 | 所需软件/数据 | 所需资源 | 现金预算 | 时间 | 验收标准 | 优先级 |
| --- | --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| A01 | 可校准拒绝执行，回答模型什么时候不应继续动作 | 只有规则 guard/watchdog，没有概率校准 | G03/G07 的成功与失败样本；独立 calibration/test split | 风险特征、统计校准、风险-覆盖评测 | CPU 可完成规则分数；模型特征/多采样需 20-80 GPU 小时 | ¥0-1,500 | 1-2 周 | 报告 risk-coverage、ECE/coverage、选择性成功/违规率与 CI；阈值只在 calibration 集确定 | P2 |
| A02 | Isaac Lab 域随机化，检验负载、摩擦、视觉和并行故障的 robustness | 未集成；现有 MuJoCo 已覆盖 CPU 控制闭环 | 必须先确定新增控制变量；G04 方法稳定 | Isaac Lab、Panda asset、统一 action/fault adapter | 4090 24 GB；16 vCPU；64 GB；200 GB；20-80 GPU 小时 | ¥200-1,500 | 4-10 天 | 同一控制问题在 MuJoCo/Isaac Lab 均可运行；差异来自注册变量，不是画质；完整对照报告 | P2 |
| A03 | 官方 LeRobot ACT 或 Diffusion Policy 训练，补足数据和策略训练链路 | 当前没有训练策略 | C06；公开或自采数据；train/val/test split | LeRobot dataset、ACT/DP recipe、50-150 GB 数据 | 12-24 GB GPU；20-80 GPU 小时 | ¥200-1,200 | 1-2 周 | 可复现训练曲线；至少 20-50 个未见初态 rollout；与未训练/BC 基线区分 | P2 |
| A04 | LoRA 或延迟感知 adapter，研究训练式方法能否超过免训练调度 | 未开始 | G03/G07 基线；数千条匹配演示或 latency 数据 | OpenPI/OpenVLA-OFT 训练 recipe、数据卡 | 建议 24 GB 起；部分配置需 A100 40/80 GB；50-200 GPU 小时 | ¥500-3,000 | 2-4 周 | 与 frozen checkpoint、suffix selection 和 no-adapter 同协议比较；held-out 场景和多 seed | P3 |
| A05 | 学习式恢复的行为克隆基线 | 未开始 | C07 失败分类；恢复 demonstration/oracle | LeRobot/自定义恢复数据 | 24 GB GPU；20-100 小时 | ¥200-1,500 | 1-3 周 | 与 hold、规则 repair 和 scripted recovery 比较；至少 3 个训练 seed | P3 |
| A06 | RL 恢复或延迟适应 | 未开始 | A05、稳定 reward、并行仿真、BC 和 no-adaptation baseline | Isaac Lab/ManiSkill、PPO/DPPO 类实现 | 24 GB+ GPU；100-500 GPU 小时 | ¥1,500-8,000 | 3-8 周 | 3-5 个 seed、学习曲线、相同 env steps、held-out fault matrix 和训练成本完整 | P3 |

RL 只有在“学习恢复/适应”成为明确研究问题时才做。它不能替代 G01-G07 的真实在线 VLA 和跨模型验证。

### 3.4 ROS2 与实体机器人

| ID | 技术加强与解决的问题 | 当前状态 | 前置条件 | 所需软件/数据 | 所需硬件 | 现金预算 | 时间 | 验收标准 | 优先级 |
| --- | --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| H01 | ROS2 transport/lifecycle，补上消息时钟、重连和执行器进程边界 | 未完成 | C01-C07；确定目标机器人动作语义 | Ubuntu 22.04、ROS2 Humble、rosbag2 | 可先用本机/另一台 CPU 主机 | ¥0-300 | 4-10 天 | observation/provider/guard/actuator 分节点；乱序、掉线、时钟回退均 fail closed；rosbag 可重放 | P1，真机前 |
| H02 | SO-101 类单 follower 实体接入 | 无硬件；Panda `H x 8` 不能直接驱动 SO-101 | C06、H01；新增 SO-101 position-action semantics 和 adapter | 官方 LeRobot driver、标定数据 | follower、双相机、支架、供电 hub、急停、备件 | ¥2,800-7,300 | 1-3 周 | 标定/限位/watchdog/断连/急停通过；至少 30 次重复 episode；旧命令不会恢复执行 | P2 |
| H03 | leader-follower 遥操作与数据采集 | 未完成 | H02；任务与数据质量规范 | 官方 teleoperation 和 dataset recorder | 在 H02 上增加 leader 与桌夹 | 追加 ¥1,200-2,500 | 1-2 周 | 50-200 条合格 demonstrations；官方 loader 可读；失败标签、相机标定和数据质检完整 | P2 |
| H04 | 实体学习策略/VLA 闭环 | 未完成 | H02-H03、A03 或可用 SO-101 checkpoint；远程 GPU 通道 | LeRobot/OpenPI client、ROS2/runtime | 复用机器人；远程 24 GB GPU | 追加 ¥300-1,500 | 1-3 周 | 真实策略、独立时钟、任务成功/违规/干预/时延全记录；断连进入硬件 hold/断电边界 | P2 |
| H05 | 实体故障矩阵 | 未完成 | H04；书面安全流程和软质任务环境 | 网络故障代理、故障注入器 | 急停/继电器、软障碍物、可控负载 | 追加 ¥200-1,000 | 3-7 天 | 延迟、丢帧、断连、旧响应、负载变化每类至少 10 次；测停止时间/距离和人工干预 | P2 |
| H06 | 七自由度 Panda/FR3 真机复现 | 无设备 | 实验室合作、libfranka、实时内核、场地与安全规范 | ROS2/libfranka、标定和真机协议 | Franka 本体及配套通常 ¥150,000+ | 不建议个人购买 | 2-6 个月 | MuJoCo 与实体使用同构任务；硬件时序、急停和多次故障实验完整 | P3，仅合作 |

SO-101 通常不是七自由度 Panda。使用它的技术意义是新增一个严格的 embodiment adapter，验证 provider、watchdog、数据和故障处理能否跨机器人迁移，不能把结果写成 Panda 真机实验。

## 4. 云服务器资源清单

### 4.1 推荐的单机配置

| 配置项 | 推荐值 | 原因 |
| --- | --- | --- |
| 操作系统 | Ubuntu 22.04 LTS | 与现有 OpenPI、LIBERO、ROS2 Humble 路径一致 |
| GPU | RTX 4090 24 GB | 满足 `pi0.5` 推理、OpenVLA-OFT 推理和大部分个人实验 |
| CPU | 16 vCPU，最低 8 vCPU | LIBERO/MuJoCo、视频、数据验证和模型服务并行 |
| 内存 | 64 GB，最低 32 GB | Docker、仿真器、模型服务和缓存同时运行 |
| 系统盘/数据盘 | 200 GB 起；两模型并存建议 300-350 GB | 镜像、checkpoint、缓存、视频和 artifact |
| 网络 | 可访问 GitHub/Hugging Face/模型存储；至少 10 Mbps | 国内网络失败会直接浪费 GPU 计费时间 |
| 计费 | 按小时；关闭自动续费 | 先 pilot，再决定是否扩大矩阵 |

### 4.2 GPU 小时与费用

以下按 4090 约 ¥3-15/小时并加入磁盘、下载和失败余量估算。

| 实验包 | GPU 小时 | 预算 | 产出 |
| --- | ---: | ---: | --- |
| `pi0.5 -> Panda` 在线 smoke | 6-12 | ¥50-300 | G01，真实 checkpoint 端到端 trace |
| 独立时钟 40-rollout pilot | 12-30 | ¥100-600 | G02，判断是否值得正式扩大 |
| `pi0.5` 正式矩阵和在线 repair | 30-100 | ¥500-2,000 | G03-G04 |
| OpenVLA-OFT smoke 与 provider 接入 | 16-42 | ¥150-800 | G05-G06 |
| 两模型正式矩阵 | 40-120 | ¥500-2,500 | G07 |
| Isaac Lab 对照 | 20-80 | ¥200-1,500 | A02，可选 |
| ACT/DP 小策略训练 | 20-80 | ¥200-1,200 | A03，可选 |
| LoRA/adapter | 50-200 | ¥500-3,000 | A04，可选 |
| RL 恢复 | 100-500 | ¥1,500-8,000 | A06，不属于近期主线 |

GPU 实例启动前必须在 CPU 环境完成：代码 checkout、配置生成、协议 freeze、容器构建检查、checkpoint 下载地址检查和自动关机脚本。重要 artifact 在释放云盘前至少保存到本地和一个远程仓库/对象存储。

## 5. 实体硬件采购清单

只在 H01 软件接口完成并决定进入 H02 后购买。

| 物品 | 必需性 | 估算 | 技术用途 |
| --- | --- | ---: | --- |
| SO-101 类 follower 套件 | 必需 | ¥1,500-3,000 | 实体执行器与夹爪 |
| SO-101 类 leader 套件 | 数据采集阶段必需 | ¥1,200-2,500 | 遥操作 demonstrations |
| 两个 1080p UVC 相机 | 必需 | ¥300-1,000 | 外部与腕部视觉 |
| 相机/机械臂支架、桌夹、标定板、照明 | 必需 | ¥300-900 | 固定外参与可重复场景 |
| 带供电 USB hub、电源和线材 | 必需 | ¥150-500 | 降低通信与供电故障 |
| 可锁存物理急停或断电继电器 | 必需 | ¥200-800 | 软件失效时切断执行器 |
| 备用舵机、齿轮、夹爪件和线材 | 建议 | ¥200-600 | 降低维修停工时间 |
| 软质任务物体、托盘和桌面防护 | 必需 | ¥100-500 | 低风险、可重复任务 |
| 内存升级到 32 GB | 条件购买 | ¥300-800 | 只有确认笔记本可升级且 WSL/并行仿真实测内存不足时购买 |
| 1 TB 移动 SSD | 条件购买 | ¥350-650 | 本机无法留出至少 150 GB 时保存数据 |
| RealSense 深度相机 | 现在不买 | ¥1,500-3,500 | 只有深度/3D 成为实验变量时购买 |
| Jetson Orin | 现在不买 | ¥2,000-8,000 | 大型 VLA 仍需云 GPU，当前没有必要 |
| 本地 RTX 3090/4090 主机 | 现在不买 | 约 ¥6,000-20,000+ | 预计累计使用超过约 300-500 GPU 小时后再核算 |

单 follower 完整路线约 ¥2,800-7,300；增加 leader 后约 ¥4,000-9,800。价格差异主要来自舵机套件、相机、急停方案和备件，不应只比较机械臂裸套件价格。

## 6. 技术预算方案

| 阶段 | 必做 ID | 所需资源 | 阶段预算 | 达成的技术等级 |
| --- | --- | --- | ---: | --- |
| 第一阶段：本地核心 | C01-C05，随后 C06-C07 | 当前电脑；可选 WSL2 | ¥0-100 | provider、独立时钟、QP、连续碰撞和动力学停止达到 L1 |
| 第二阶段：首次 GPU | G01-G02 | 4090 24 GB，约 18-42 小时，200 GB 盘 | ¥150-900 | 真实 `pi0.5` 在线链路和独立时钟 pilot 达到 L2 |
| 第三阶段：核心实验 | G03-G07 | 4090 24 GB，约 86-292 小时，250-350 GB 盘 | ¥1,150-6,000 | 正式独立时钟结果与第二真实模型达到 L3-L4 |
| 第四阶段：低成本真机 | H01-H05；可选 A03 | SO-101 pair、双相机、急停、云 GPU | ¥4,500-12,300；训练另加 ¥200-1,200 | 数据采集、真实策略和故障矩阵达到 L5 |
| 第五阶段：论文扩展 | A01-A06、G08、H06 中按问题选择 | 更多 GPU 或实验室机器人 | ¥2,000-10,000+ | 是否可投稿取决于新方法和实验结果，不由预算保证 |

## 7. 固定执行顺序

1. 先完成 C01-C05，不购买设备，不租 GPU。
2. 完成 C06-C07，把官方 LeRobot loader 和统一 fault matrix 接入同一契约。
3. 所有 CPU preflight 通过后，只租 6-12 小时 GPU 完成 G01。
4. G01 artifact 可验证后扩展到 G02；pilot 失败时先修时钟/协议，不直接购买更长时长。
5. G02 通过后再运行 G03-G04，形成真实异步核心实验。
6. 随后完成 G05-G07，加入第二个真实 VLA 家族。
7. 只有以上完成后，才在 Isaac Lab、训练、RL 和实体机器人之间按研究问题选择。

近期技术目标固定为：

> 在本地完成 provider、独立时钟、QP、连续碰撞和动力学停止；随后用一台 24 GB GPU 服务器完成真实 `pi0.5` 在线闭环与独立时钟 pilot，再接入真实 OpenVLA-OFT 做跨模型验证。
