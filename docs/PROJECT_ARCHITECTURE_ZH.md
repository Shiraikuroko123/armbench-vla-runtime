# ArmBench 架构与主张边界

状态：Current。更新日期：2026-08-05。

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

## 当前证据

| 组件 | 已验证结果 | 主张边界 |
| --- | --- | --- |
| Measured-age 调度器 | `pi0.5`-LIBERO Spatial，120 组匹配试验：88/120 到 116/120，+23.33 个百分点，exact McNemar `p=1.94e-6` | 一个冻结的官方 checkpoint 与仿真矩阵 |
| 跨任务集验证 | Object、Goal、LIBERO-10，300 rollouts / 150 pairs：83/150 到 141/150 | 同一模型族和仿真套件内的确定性延迟证据 |
| RTC-style sampler extension | 300 组匹配 triplet：baseline 96/100，hard projection 97/100，RTC 97/100 | 没有任务成功率优势；seam 是探索性指标 |
| Panda 运行时 | 本地 MuJoCo 中的协议、guard 和故障 trace | 不是官方 `pi0.5` 的效果证据，也不是物理安全证明 |

详细结果见[结果说明](RESULTS.md)，冻结协议和审计记录见[文档索引](README.md)。

## 当前已经集成了什么

当前集成的是共同的运行时接口与证据模型，而不是经过验证的“`pi0.5` 的 LIBERO 动作直接转为 Panda 关节命令”控制链。后者需要显式动作 adapter、坐标系和夹爪语义、逆运动学或受限投影、时间同步，以及新的端到端实验。

因此，不应表述为：`pi0.5` 已部署到 Panda、Panda guard 已证明 VLA 安全，或仿真结果已经达到硬实时。

## 下一阶段的合理整合

真正有意义的下一步不是换一个仿真器，而是通过明确的 adapter 连接两条路径，并评测：

1. 独立调度的推理和控制，以及过期响应丢弃；
2. 有 deadline 的受限投影，包括连续碰撞和动力学约束；
3. 第二个开放 action-chunk policy 在相同冻结协议下的表现；
4. 面向 LeRobot 或真实硬件的 adapter，并单独报告安全和时序证据。

在此之前，准确的公开表述是：**七自由度受限执行基座 + `pi0.5 VLA` 运行时评测路径，共用可审计运行时基础设施。**
