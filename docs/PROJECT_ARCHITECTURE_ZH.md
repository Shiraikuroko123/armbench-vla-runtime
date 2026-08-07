# ArmBench 架构与主张边界

状态：Current。更新日期：2026-08-07。

## 项目目的

ArmBench 是一个研究动作块式 VLA 在“模型响应返回”和“机器人执行”之间如何运行的工程平台。它不是新的基础模型，不是 `pi0.5` 训练项目，不是真机部署，也不是碰撞安全认证系统。

平台由两条独立验证的执行路径和一层共用运行时/证据层组成。这个分离是刻意保留的：本地 Panda guard 的结果不能被表述成 `pi0.5`-LIBERO 结果，LIBERO 的任务成功率也不能被表述成 Panda 安全结果。

## 术语

- **`pi0.5`**：Physical Intelligence 的 pi-zero-point-five 视觉-语言-动作模型。首次完整说明后，使用 `pi0.5 VLA`。
- **OpenPI**：用于调用 `pi0.5` checkpoint 的上游模型与推理实现栈，不是另一个模型。
- **action chunk**：模型根据一次观测预测的一段未来控制动作。
- **观测年龄（measured age）**：从观测采集到策略响应可供控制器使用的耗时。
- **fail-closed**：响应无效或不可用时，执行已注册的 hold/fallback，而不是猜测动作。

## 组件

### 1. 七自由度 Panda 受限执行基座

本地 MuJoCo Panda 路径提供经典机器人学基础：

- RRT-Connect/RRT* 规划、路径平滑和时间参数化；
- PD/LQR 轨迹跟踪与扰动下的执行检查；
- 构型采样和边插值的碰撞检查；
- 关节、速度、夹爪和状态一致性检查；
- 双相机观测、传输测试和确定性故障注入。

这条路径用于验证七自由度机械臂上的运行时契约和受控失败行为。碰撞检查属于采样检查，不是连续碰撞证明。

### 2. `pi0.5 VLA` 时序评测路径

官方 checkpoint 路径通过 OpenPI 在 LIBERO 中运行经过 attestation 的 `pi05_libero` checkpoint。策略接收规范化图像、机器人状态和语言指令，输出 action chunk。

核心问题是：推理延迟后，这段动作是否仍具有时间上的可执行性。免训练的 measured-age 调度器根据响应年龄计算保守后缀偏移；只有后缀仍落在剩余 horizon 和 deadline 内才执行，否则进入有界的 hold/refresh。

### 3. 共用运行时与证据层

两条路径复用以下工程层：

```text
观测/状态/指令契约
          |
          v
策略传输与响应校验
          |
          v
观测年龄、deadline、状态检查与 fail-closed 监管
          |
          v
逐 query/逐 action trace、视频、manifest、验证器和 dashboard
```

共用工具不代表实验语义相同。LIBERO 和 Panda 的动作空间、策略、执行环境和结果主张保持分离。

当前维护的运行时还增加了一个组件级非阻塞验收：阻塞策略在独立 worker 中
运行，latest-only 待处理邮箱限制积压；控制侧拒绝乱序、失败、超过 deadline
或耗尽 action horizon 的响应。该验收使用 scripted policy 在 CPU 上验证，尚未
替换既有 `pi0.5` 实验采用的 blocking inference 加 simulator catch-up evaluator。

## 当前证据

| 组件 | 已验证结果 | 主张边界 |
| --- | --- | --- |
| Measured-age 调度器 | `pi0.5`-LIBERO Spatial，120 组匹配试验：88/120 到 116/120，+23.33 个百分点，exact McNemar `p=1.94e-6` | 一个冻结的官方 checkpoint 与仿真矩阵 |
| 跨任务集验证 | Object、Goal、LIBERO-10，300 rollouts / 150 pairs：83/150 到 141/150 | 同一模型族和仿真套件内的确定性延迟证据 |
| RTC-style sampler extension | 300 组匹配 triplet：baseline 96/100，hard projection 97/100，RTC 97/100 | 没有任务成功率优势；seam 是探索性指标 |
| Panda 运行时 | 本地 MuJoCo 中的协议、guard 和故障 trace | 不是官方 `pi0.5` 的效果证据，也不是物理安全证明 |
| 分线程运行时验收 | 独立 worker/control 线程、持续 control tick、latest-only 替换与 deadline 测试 | Scripted 组件证据；不主张 LIBERO 或 Panda 任务成功率 |
| 笛卡尔动作适配器 | 将 scripted `H x 7` LIBERO 风格动作经 Panda Jacobian 转为现有 `H x 8` guard 契约 | 仅为组件 smoke；不包含官方 checkpoint、任务成功率或控制器等价性主张 |
| 冻结响应 Panda 回放 | 核验 7,934 个官方响应哈希，并将 90 个动作块送入 3 个 Panda 场景 | 跨控制器离线诊断；未执行 checkpoint、反馈闭环或任务成功率评测 |

详细结果见[结果说明](RESULTS.md)，冻结协议和审计记录见[文档索引](README.md)。

## 当前已经集成了什么

项目现在已经实现组件级笛卡尔动作适配器：把声明为 LIBERO 风格的
`H x 7` 末端动作转换为 Panda 运行时的 `H x 8` 关节速度/夹爪契约，内部使用
MuJoCo Panda hand Jacobian、阻尼最小二乘微分逆运动学、关节限位缩放和现有
guard。确定性 CPU smoke 见
[LIBERO 到 Panda 的笛卡尔动作适配器](PANDA_CARTESIAN_ADAPTER_ZH.md)。

适配器现在还接受经过严格校验的官方 checkpoint 冻结响应离线回放。该流程核验全部响应哈希，
按 LIBERO 任务和运行时方法等额抽样，每个 Panda 案例独立重置，并生成可自校验的 CSV/JSON
报告。详见[冻结 pi0.5 响应的 Panda 离线回放](PI05_PANDA_ARCHIVE_REPLAY_ZH.md)。

但它仍不是经过验证的“官方 `pi0.5` 在线推理直接控制 Panda”链路。尺度、坐标系、
裁剪和夹爪语义已经与 LIBERO commit `f78abd68`、robosuite `1.4.1` 源码核对，
但微分逆解在动力学上不等价于 robosuite 的 torque-level OSC。官方 checkpoint
worker 也尚未接入独立时钟推进的 Panda 或 LIBERO actuator loop。只有完成这些
集成、时间同步和新的冻结实验，才能提出端到端主张。

因此，不应表述为：`pi0.5` 已部署到 Panda、Panda guard 已证明 VLA 安全，或仿真结果已经达到硬实时。

独立推理/控制调度目前已经作为本地运行时组件实现并测试，但尚未接入官方
checkpoint 的 LIBERO evaluator 或 Panda actuator loop。Python 线程也不提供
操作系统调度和最坏时延保证。

## 下一阶段的合理整合

真正有意义的下一步不是换一个仿真器，而是把已实现的组件适配器接入完整
evaluator，并评测：

1. 将独立调度运行时接入端到端 evaluator，并形成任务级过期响应丢弃证据；
2. 有 deadline 的受限投影，包括连续碰撞和动力学约束；
3. 第二个开放 action-chunk policy 在相同冻结协议下的表现；
4. 面向 LeRobot 或真实硬件的 adapter，并单独报告安全和时序证据。

在此之前，准确的公开表述是：**七自由度受限执行基座 + `pi0.5 VLA`
运行时评测路径 + 显式的组件级笛卡尔动作适配器，共用可审计运行时基础设施。**
