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
