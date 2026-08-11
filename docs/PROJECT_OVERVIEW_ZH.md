# VLA-Sync 项目概览

这份文档面向第一次接触仓库的读者。它先用直白的语言说明项目解决什么问题，
再给出对应的技术术语、代码入口和证据边界。阅读它不需要先了解 VLA、MuJoCo
或七自由度机械臂。

## 一句话定位

**VLA-Sync 是一个面向动作块式视觉-语言-动作（VLA）策略的异步运行时与评测
系统。它在模型输出进入相应控制器之前，判断动作是否仍然与当前时刻对应，
并在 Panda 保障路径中进一步检查已登记的执行约束，为每次执行或拒绝保留可复核
证据。**

项目使用 Physical Intelligence 的 `pi0.5 VLA` 官方 checkpoint 和 OpenPI
推理栈进行 LIBERO 仿真研究，但不训练或微调 `pi0.5`。它也不是实体机器人
安全认证、硬实时控制器或官方排行榜实现。

## 它解决什么问题

VLA 通常不会只返回一个关节命令，而是根据一帧图像、机器人状态和语言指令，
一次预测一段未来动作，这段序列称为 **action chunk（动作块）**。

假设模型在时刻 `t0` 看到画面并开始推理。推理期间，仿真或机器人控制器仍然
按自己的时钟推进；当响应在 `t1` 到达时，动作块的前几步可能已经对应过去的
状态。如果收到响应后无条件从动作索引 `0` 开始执行，就会把基于旧观测的动作
重新施加到已经变化的系统上。

这不是单纯的模型准确率问题，而是**模型时钟、控制时钟和传输时延之间的接口
问题**。如果评测在推理期间暂停仿真，就无法观察到这种真实的时钟错位。

## 系统如何处理

VLA-Sync 将推理进程和控制循环分开运行，并按以下顺序处理每个响应：

| 步骤 | 对外行的解释 | 工程实现 |
| --- | --- | --- |
| 1. 观测 | 相机、机器人状态和指令告诉模型当前要做什么 | `VLAObservation`，带有序列号和单调时钟时间戳 |
| 2. 推理 | 模型生成一段未来动作，而控制时钟不会暂停 | 独立 provider worker、latest-only mailbox |
| 3. 时间对齐 | 跳过响应返回前已经失效的动作前缀 | 按 measured observation age 计算 `action_index` |
| 4. 执行检查 | 确认响应新鲜、形状正确，并满足已登记的执行条件 | deadline、sequence、状态和动作语义门禁 |
| 5. 发布或保持 | 合格动作才交给控制器；不合格时不猜测动作 | `execute` 或 fail-closed `hold/refresh` |
| 6. 留存证据 | 之后可以检查当时为什么执行或拒绝 | trace、视频、manifest、源码哈希和 validator |

核心对照是：

- `age_aligned_suffix`：根据观测年龄，从动作块中选择仍对应当前时刻的后缀；
- `response_relative_chunk`：响应返回后仍从动作块第 `0` 项开始执行，但仍受
  相同的 deadline 和 hold 规则保护。

## 它与七自由度机械臂项目的关系

项目名称从“七自由度机械臂采样与受限轨迹跟踪基准”扩展为 VLA-Sync，表示研究
层次增加，并不是两个无关项目：

```text
七自由度 Panda 基座
  RRT-Connect / 路径平滑 / 时间参数化
  OSQP 受限投影 / 连续碰撞检查 / 动力学制动
  MuJoCo 状态、力矩执行和轨迹证据
                 |
                 v
VLA-Sync 运行时层
  provider 契约 / 独立时钟 / 动作年龄 / deadline
  latest-only 调度 / 原子发布 / fail-closed 回退
  LIBERO 任务评测、Panda 保障回放和证据验证
```

因此，原来的七自由度项目提供“动作怎样被约束和执行”的底层能力；VLA-Sync
增加“VLA 生成的动作何时可以进入控制器”的时序和验证能力。

两条证据路径必须分开理解：

| 路径 | 回答的问题 | 不能推出的结论 |
| --- | --- | --- |
| `pi0.5` + LIBERO | 延迟、动作选择和 deadline 如何影响仿真任务结果 | 不能推出 Panda 真机安全或跨模型普遍性 |
| Panda + MuJoCo | QP、连续碰撞、制动和原子发布是否满足登记约束 | 不能把 scripted Panda 成功写成 `pi0.5` 任务成功 |

## 仓库各部分的职责

| 目录或文件 | 作用 | 适合先看什么 |
| --- | --- | --- |
| `src/armbench/vla/types.py` | 定义观测、动作块、时间和身份字段 | 输入输出的形状与语义 |
| `src/armbench/vla/async_worker.py`、`process_worker.py` | 将可能阻塞的策略调用移出控制线程 | 推理和控制为何能并行推进 |
| `src/armbench/vla/async_dispatch.py` | 按动作年龄选择后缀并处理 deadline | 运行时最核心的决策 |
| `src/armbench/vla/integrated_panda_guard.py` | 组织 QP、碰撞和制动检查，形成原子决策 | Panda 侧如何拒绝不安全动作 |
| `src/armbench/vla/qp_projection.py` | 施加关节、速度和加速度约束 | 受限动作修复 |
| `src/armbench/mujoco_sim/` | Panda 几何、动力学、碰撞和轨迹执行 | 七自由度执行基座 |
| `integrations/openpi/` | `pi0.5`/LIBERO provider、请求和验证入口 | 真实 checkpoint 如何接入 |
| `evidence/`、`reports/` | 保存实验输入、输出、视频、统计和清单 | 结果是否可复核 |
| `docs/` | 协议、方法、边界、验收和调试说明 | 研究与工程文档 |

## 现有证据说明什么

当前最容易解释的三组结果是：

1. **动作选择对照**：冻结的 120 对 LIBERO 仿真中，观测年龄对齐方法完成
   `114/120`，响应后从第 0 项开始的方法完成 `100/120`；成对差值为
   `+11.67` 个百分点，exact McNemar `p=0.00936`。
2. **deadline 矩阵**：18 个独立验证单元、720 次 rollout 显示，服务时延跨过
   20 Hz 控制 tick 时，可发布动作占比会出现离散变化；这不是所有 VLA 的通用
   deadline 阈值。
3. **Panda 保障基座**：脚本化 MuJoCo Panda 任务的注册运动和制动边界为
   `351/351` 通过；这验证底层执行链，不是 `pi0.5` 物体操作成功率。

所有数字都对应仓库中的报告、manifest 和 validator，而不是网页手工填写的汇总。
完整结果见 [`docs/RESULTS.md`](RESULTS.md)。

## 如何验收

### 不需要 GPU：验证已经保存的证据

在仓库根目录完成本地安装后，可以运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\accept_cpu.ps1
& '.\.venv\Scripts\python.exe' -m pytest -q
```

这条路径会重算本地 CPU 证据、检查 manifest 和验证器；它不会重新执行
`pi0.5` checkpoint。

### 有 MuJoCo 图形界面：观看 Panda 轨迹

```powershell
& '.\.venv\Scripts\python.exe' -m armbench mujoco-view --scenario narrow_gate
```

视频只用于观察轨迹。判断结果是否正确时，应以 validator、CSV/NPZ trace 和
summary 中的状态、碰撞、限位与 deadline 字段为准。

### 重新运行 `pi0.5`

需要 Linux、NVIDIA GPU、OpenPI 环境和对应 checkpoint。没有这些资源时，仍然
可以完整复核仓库已经保存的 `pi0.5` 响应和统计证据。

## 面试或项目介绍的规范表述

> VLA-Sync 是一个面向动作块式 VLA 的异步运行时与评测平台。我针对模型推理
> 与机器人控制时钟不同步导致的过期动作问题，独立实现了 measured-age 动作
> 后缀选择、deadline/fail-closed 调度、Panda 受限动作保障和可重算证据链，
> 并在官方 `pi0.5`-LIBERO 仿真中完成冻结配对对照。项目不训练 VLA，当前证据
> 主要来自仿真，Panda 与 LIBERO 结果按动作契约分开统计。

## 当前边界

- 没有训练或微调 `pi0.5`。
- 官方 checkpoint 结果来自仿真；当前不是 ROS2、实体 Panda 或安全 PLC 部署。
- Python 运行时的时延是 best-effort 测量，不构成操作系统级硬实时保证。
- 动作选择结果覆盖已登记的 checkpoint、suite、seed 和协议，不能直接外推为
  跨模型或真机结论。
- Panda 的任务执行使用 scripted 参考动作；不能将其表述为 `pi0.5` 的任务成功。

## 推荐阅读顺序

1. 本概览：理解问题、系统角色和七自由度项目的关系。
2. [`PROJECT_ARCHITECTURE_ZH.md`](PROJECT_ARCHITECTURE_ZH.md)：确认组件和主张边界。
3. [`RESULTS.md`](RESULTS.md)：查看每组实验的协议、数字和限制。
4. [`CODE_WALKTHROUGH_ZH.md`](CODE_WALKTHROUGH_ZH.md)：按调用链阅读源码。
5. [`DEBUGGING.md`](DEBUGGING.md)：遇到环境、运行时或可视化问题时定位。
