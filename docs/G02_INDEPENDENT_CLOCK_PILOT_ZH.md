# G02：pi0.5-LIBERO 独立时钟 pilot

G02 是 ArmBench 第一次把官方 `pi05_libero` checkpoint、LIBERO 仿真器和
运行时调度器放在同一条实验链路中，并真正使用相互独立的时钟。父进程以
20 Hz 推进仿真；独立子进程通过已经完成 attestation 的 OpenPI 服务执行可能
阻塞的推理。单槽 latest-only mailbox 不让控制时钟等待推理，同时记录请求替换、
响应年龄、deadline 决策以及 hold/execute 动作。

## 冻结矩阵

| 项目 | 设置 |
| --- | --- |
| 套件 | LIBERO Spatial |
| 任务 | 0-9 |
| episode | 每个任务 0-3 |
| rollout | 40 |
| seed | 7 |
| 控制周期 | 50 ms（20 Hz） |
| deadline | 200 ms |
| action horizon | 10 x 7 |
| checkpoint | 官方 `pi05_libero`；内容 SHA-256：`9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5` |

## 结果

40/40 个分配的 rollout 均完成。LIBERO 报告 38/40 成功（95.0%），失败的
两条均为 task 4 的 `max_ticks`，没有从分母中删除。运行时共记录 4,623 个
控制 tick，其中 4,521 个发生在推理子进程执行期间；4,031 个 tick 执行动作
后缀，592 个 tick 进入 hold。没有 deadline 超时，也没有 provider 失败。

这回答的是运行时工程问题：在仿真继续推进的同时，真实 checkpoint 是否可以被
独立时钟驱动，并且每个请求、动作、回退和失败都可审计。它不是官方 LIBERO
排行榜成绩、方法优越性结论、操作系统级硬实时保证、硬件安全证明或真机结果。
两条失败视频被完整保留。

## 证据与验收

- [40-rollout 核心 artifact](../evidence/pi05_libero_independent_clock_core_40_001/README.md)
- [单条可视化成功 artifact](../evidence/pi05_libero_independent_clock_visual_success_001/README.md)
- [证据目录](EVIDENCE_CATALOG.md)

在没有 GPU 的情况下验证核心结果：

```powershell
& '.\.venv\Scripts\python.exe' -m integrations.openpi.validate_libero_independent_clock `
  'evidence\pi05_libero_independent_clock_core_40_001\evaluation' --json
```

validator 不会重新调用 checkpoint，而是重算 manifest、源码快照、模型身份、
初始状态摘要、action chunk hash、过期前缀、请求/tick 顺序、聚合统计和视频覆盖。

## 复现边界

重新运行 G02 需要 Linux、NVIDIA CUDA、LIBERO、OpenPI 服务以及公开 checkpoint。
没有 GPU 时仍可验证保存的 artifact、播放视频，并运行
[独立时钟 CPU smoke](INDEPENDENT_CLOCK_RUNTIME_ZH.md)。
