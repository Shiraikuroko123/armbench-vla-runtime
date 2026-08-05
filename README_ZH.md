[English](README.md) | [文档索引](docs/README.md)

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

完整协议、验证器、统计和研究限制见[结果说明](docs/RESULTS.md)。

## 不应声称的内容

- 没有训练或微调 `pi0.5`。
- 官方 checkpoint 结果仅覆盖 LIBERO 仿真。
- 时序实验使用 blocking inference 加 simulator catch-up，不是操作系统级硬实时控制。
- Panda guard 不是碰撞安全认证，也不能证明 `pi0.5` 已控制 Panda。
- 没有集成 Isaac Lab、ROS2、真实 Franka Panda 或安全 PLC。

## 验收已保存结果

以下命令只核验保留的实验数据并重建离线 dashboard，不重跑模型推理，也不需要 GPU：

```powershell
.\scripts\measured_age_confirmatory_acceptance.cmd
.\scripts\rtc_primary_acceptance.cmd
```

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
