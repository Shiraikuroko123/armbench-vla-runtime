# 优化后的 pi0.5 到 Panda CPU 保障回放

状态：R02 本地 CPU 优化审计已完成。正式 artifact 位于
[`reports/pi05_optimized_cpu_replay_180_001`](../reports/pi05_optimized_cpu_replay_180_001/summary.md)。
它是已保留的[`20 ms no-go 基线`](PI05_INTEGRATED_PANDA_CPU_REPLAY_ZH.md)的后续版本，
不会覆盖或改写旧结果。

## 这一步回答什么

旧基线已经证明完整 Panda 保障链能够 fail closed，但在 20 ms 软件预算内没有发布
任何一条经过完整检查的计划。本次审计只改变 CPU 实现，检验在不放宽运动学、连续
碰撞、动力学制动、deadline 和原子发布条件的前提下，能否让至少一条完整计划进入执行。

优化后的链路包括：

- 复用关节位置、速度和加速度投影的 OSQP workspace；
- 在精确 MuJoCo 距离查询前执行保守 broad phase，并向量化区间上界计算；
- 复用逆动力学与制动计算 workspace；
- 只缓存已经证明安全的 configuration，且每个正式案例开始前清空；
- 由统一 supervisor 完整发布计划或发布零运动 hold，不泄漏局部通过的动作前缀。

## 固定的工程审计

协议保存在
[`pi05_optimized_cpu_replay_protocol_20260810.json`](research/pi05_optimized_cpu_replay_protocol_20260810.json)。
它在优化 profiling 后固定，因此属于工程审计，不是预注册的推断性研究。

runner 复用旧基线中的 30 个已校验冻结响应和 3 个 Panda 场景，得到 90 个案例；
每个案例分别进入两个 profile：

| Profile | 用途 | supervisor 预算 | 响应 deadline |
| --- | --- | ---: | ---: |
| `operational_20ms` | 主要工程 go/no-go | 20 ms | 200 ms |
| `diagnostic_100ms` | 独立的耗时诊断 | 100 ms | 200 ms |

同一场景/profile 内 worker 常驻，但启动耗时不计入请求时延；每个正式案例都会清空
safe cache。100 ms 诊断结果不能覆盖 20 ms 的正式判定。

## 正式结果

| Profile | 执行 / 总数 | 满足全部约束的候选 | 发布不安全计划 | 预算超限 | deadline 超限 | supervisor P50 / P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `operational_20ms` | 1/90 | 66/90 | 0 | 69 | 3 | 21.493 / 23.888 ms |
| `diagnostic_100ms` | 58/90 | 66/90 | 0 | 3 | 8 | 49.560 / 86.559 ms |

两个 profile 的动作前缀泄漏都为 0。broad phase 在 20 ms profile 中先于精确检查
排除了 95.55% 的 pair test，在诊断 profile 中排除了 94.99%。

冻结的正式判定是 **`go`**：至少一条完整计划执行、发布计划的独立注册约束审计为
零违规、artifact 验证通过且没有前缀泄漏。但这只是达到最低判定规则，不代表 20 ms
稳定达标：89/90 个正式案例仍然 hold，P50 和 P95 都高于 20 ms，最大 supervisor
耗时为 24.550 ms。因此本轮证明候选可行性显著改善，同时也把“时延是否稳定”保留为
下一项 CPU 问题。

作为工程版本对照，旧基线完整保障为 0/90 execute、5/90 个安全候选、P95 42.974 ms；
优化后为 1/90 execute、66/90 个安全候选、P95 23.888 ms。这是同一冻结输入矩阵上的
版本化实现演进，不是统计意义上的方法优越性实验。

## 本地复算与验收

在仓库根目录的 Windows PowerShell 执行：

```powershell
& '.\.venv\Scripts\python.exe' -m armbench `
  vla-panda-optimized-replay `
  reports\pi05_integrated_panda_cpu_replay_270_001 `
  --output-directory results\pi05_optimized_cpu_replay

& '.\.venv\Scripts\python.exe' -m armbench `
  vla-panda-optimized-replay-validate `
  reports\pi05_optimized_cpu_replay_180_001 `
  reports\pi05_integrated_panda_cpu_replay_270_001
```

validator 会核对递归 manifest、协议和输入绑定、实现与场景哈希、180 行案例身份、
候选/发布轨迹积分、独立运动学/碰撞/制动谓词、全有或全无发布、汇总和 Markdown。
测试还会重新签名语义篡改后的 artifact，并确认它仍然被拒绝。

## 主张边界

本次不会重新运行官方 `pi0.5` checkpoint。冻结响应来自 LIBERO，再跨控制器适配到
Panda，并非 Panda 原生策略输出。每个案例互相独立，也没有把 Panda 观察反馈给策略，
因此不衡量任务成功率。Python 墙钟耗时不是操作系统硬实时保证，MuJoCo 检查也不是
实体机器人安全认证。
