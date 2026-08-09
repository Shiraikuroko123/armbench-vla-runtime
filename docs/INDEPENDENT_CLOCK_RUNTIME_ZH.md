# 独立时钟运行时验收

这是项目中 VLA 运行时主张对应的 CPU 验收。父进程独立推进环境控制 tick，
子进程执行可能阻塞的策略推理；请求邮箱只保留一个最新待处理观测。每个请求的
替换、响应年龄、deadline 决策、进程 ID 和 hold/execute 状态都会写入结果，便于
复核而不是只看最终成功率。

在没有 GPU 的本地环境运行：

```bash
cd project
python -m armbench vla-independent-clock-smoke `
  --policy-latency-ms 160 `
  --control-period-ms 10 `
  --action-period-ms 66.6667 `
  --deadline-ms 200 `
  --max-ticks 20 > independent_clock_smoke.json
```

输出必须包含 `"passed": true`。`parent_process_id` 与
`worker_process_id` 不同，且 `superseded`、`response_age_ms`、`deadline_ms`、
`hold/execute` 等字段可直接作为验收证据。

这只是调度和来源追踪验收，不是 pi0.5 任务成功率、真机安全或操作系统硬实时
保证。真实 OpenPI→Panda 链路见 `LIVE_PI05_PANDA_BRIDGE_ZH.md`，需要单独的
checkpoint attestation。

## G02 官方 checkpoint pilot

CPU smoke 之后，G02 使用同一套 latest-only mailbox 和 deadline 契约运行了
官方 `pi05_libero` checkpoint。LIBERO 由父进程以 20 Hz 推进，阻塞式 OpenPI
调用位于独立子进程。冻结矩阵覆盖 LIBERO Spatial 的 10 个任务、每个任务 4 个
episode，共 40 次 rollout。

最终 40/40 完成、38/40 成功。40/40 个 episode 都包含推理进行期间的控制
tick；4,521/4,623 个 tick 发生在推理期间，4,031 个执行动作、592 个 hold，
没有 deadline 超时或 provider 失败。两条 task 4 的 `max_ticks` 失败完整保留。

该结果证明真实 checkpoint 的独立时钟执行与审计链路成立，但仍是仿真 pilot，
不是官方排行榜成绩、硬实时保证、硬件安全证明或方法优越性结论。完整协议、
artifact 和验收命令见 [G02 中文报告](G02_INDEPENDENT_CLOCK_PILOT_ZH.md)。
