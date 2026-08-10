# 优化后 CPU 保障链的重复性审计

状态：作为 v0.2.0 优化保障审计的本地 CPU 后续实验，已经完成。正式 artifact 位于
[`reports/pi05_optimized_cpu_repeatability_20260811_001`](../reports/pi05_optimized_cpu_repeatability_20260811_001/summary.md)。

## 要回答的问题

v0.2.0 的冻结审计在 20 ms 正式 profile 的 90 个案例中记录到 1 条完整计划发布，
但当时 P50 和 P95 已经高于 20 ms。因此这一步不再问“单次能不能碰巧发布”，而是问：
该边界结果能否在独立冷启动进程中重复，以及受到有限 CPU 竞争时是否仍然成立。

本实验不修改 v0.2.0 的实现，也不追溯改写它的冻结判定；它检验的是该判定的稳定性。

## 固定矩阵

协议位于
[`pi05_optimized_cpu_repeatability_protocol_20260811.json`](research/pi05_optimized_cpu_repeatability_protocol_20260811.json)，固定了：

- 3 次独立 Python 进程的空闲冷启动；
- 3 次同时运行 4 个确定性 CPU 忙循环进程的冷启动；
- 同一批 30 个冻结响应与 3 个 Panda 场景；
- 相同的 20 ms 正式 profile 与 100 ms 诊断 profile；
- 每轮完整保存 child artifact、进程日志、hash inventory，并再次独立验证。

6 轮全部运行并验证通过；没有重新执行 `pi0.5` checkpoint。

## 结果

| 条件 | Profile | Execute 计数 | supervisor P95，均值 +/- 标准差 | 不安全发布 | 前缀泄漏 |
| --- | --- | --- | ---: | ---: | ---: |
| 空闲 | `operational_20ms` | `[1, 0, 0]` | 27.041 +/- 1.078 ms | 0 | 0 |
| 空闲 | `diagnostic_100ms` | `[20, 39, 16]` | 109.469 +/- 0.951 ms | 0 | 0 |
| 4 个 CPU worker | `operational_20ms` | `[0, 0, 0]` | 28.385 +/- 1.025 ms | 0 | 0 |
| 4 个 CPU worker | `diagnostic_100ms` | `[6, 5, 5]` | 116.095 +/- 0.654 ms | 0 | 0 |

每个 profile、每轮实验中，满足全部注册约束的候选始终是 66/90。变化的是 Python
best-effort 监督是否能赶在软件预算和响应 deadline 之前结束。

在该矩阵中，发布契约是可重复的：6 轮实验都没有发布未通过独立注册约束审计的候选，
也没有泄漏部分动作前缀。但 20 ms execute 结果不可重复：3 次空闲运行中只有 1 次发布
了 1 条完整计划；3 次 CPU 负载运行均为 0。当前工程解释因此是：**不能声称稳定或可部署
的 20 ms 达标能力**。原来的最低规则 `go` 仍作为可追溯冻结结果保留，但不足以支持更强
的实时主张。

这是有价值的负结果。它把确定性的候选可行性与主机调度/best-effort 耗时分开，并给出
了一个实测理由：在进入真机之前，deadline 关键监督需要离开不受约束的 Python 桌面
调度路径，或者采用更宽且经过注册的预算。

## 本地复算与验收

在仓库根目录的 Windows PowerShell 执行：

```powershell
& '.\.venv\Scripts\python.exe' -m armbench `
  vla-panda-optimized-repeatability `
  reports\pi05_integrated_panda_cpu_replay_270_001 `
  --output-directory results\pi05_optimized_cpu_repeatability

& '.\.venv\Scripts\python.exe' -m armbench `
  vla-panda-optimized-repeatability-validate `
  reports\pi05_optimized_cpu_repeatability_20260811_001 `
  reports\pi05_integrated_panda_cpu_replay_270_001
```

validator 会检查根 inventory、协议/输入 hash、每轮 child 日志 hash、6 个嵌套回放
manifest、原子发布、候选运动学、连续碰撞、制动谓词和条件级汇总。根 inventory
SHA-256 为
`02983f0fb036762c7b303823e64adfe426844e2708c92b756308fac56dcc279e`。

## 主张边界

每个条件只有 3 次重复，属于描述性工程证据，不是推断性时延研究。CPU 负载只是有限的
本机压力条件，不是硬件认证。冻结 LIBERO 动作仍是跨控制器 Panda 输入；案例相互独立，
也没有把 Panda 观测反馈给策略。本审计不衡量任务成功、硬实时、真机安全或跨机器泛化。
