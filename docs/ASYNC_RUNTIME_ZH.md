# 非阻塞运行时验收

状态：当前组件与官方 checkpoint 验证

仓库保留了早期采用 blocking inference 和 simulator catch-up 的
`pi0.5`-LIBERO 实验。当前维护版本同时包含推理与仿真独立推进的路径：策略
worker 可以阻塞，而 20 Hz 仿真继续运行。该路径已经运行经过认证的官方
`pi05_libero` checkpoint；scripted CPU harness 继续作为快速的本地契约测试。

## 运行结构

```text
观测采集
   |
   v
latest-only 请求邮箱 ---> 阻塞策略 worker
                              |
控制 tick 循环 <--- 结果邮箱--+
   |
   v
按观测年龄选取动作后缀 -> 执行或 fail-closed hold
```

`LatestPolicyWorker` 允许一个正在推理的请求，并最多保留一个待处理观测。
更新观测只替换尚未执行的请求；Python 无法取消已经开始的推理。结果邮箱有
容量上限，控制侧通过非阻塞 `drain` 读取。

`AsyncChunkDispatcher` 在每个控制 tick 根据观测采集时间重新计算年龄，拒绝
策略失败、乱序响应、deadline miss 和已耗尽 horizon 的响应。较新的策略失败
会清除当前动作块并进入 hold，不会继续执行更旧的动作序列。

## 本地验收

```powershell
& '.\.venv\Scripts\python.exe' -m armbench vla-async-smoke
```

JSON 会记录 worker/control 线程 ID、推理耗时、推理阻塞期间的控制 tick、最大
tick 间隔、后缀偏移、hold 次数和最终调度决定。还可以显式验收超时路径：

```powershell
& '.\.venv\Scripts\python.exe' -m armbench vla-async-smoke `
  --policy-latency-ms 240 --deadline-ms 200
```

第二条命令仍应通过运行时验收，同时报告响应被拒绝，并以
`deadline_exceeded` 原因 hold。

## 证据边界

该工具只用延迟 scripted policy 验证线程、邮箱、乱序和 deadline 状态机，不
运行 OpenPI、`pi0.5`、LIBERO 或 MuJoCo，也不产生任务成功率证据。它没有设置
实时线程优先级，不能提供最坏调度时延保证。

官方 checkpoint 证据单独记录在
[720-rollout deadline 研究](../reports/pi05_deadline_multisuite_report_720_20260810_001/summary.md)
与
[240-rollout held-out 动作选择研究](../reports/pi05_selection_heldout_report_240_20260810_001/summary.md)。
这些结果只证明一个 checkpoint、两个 LIBERO suite 中的独立时钟仿真行为；不证明
硬实时调度、通用 deadline 阈值、实体安全或跨模型泛化。下一步是把受约束投影和
制动修复接入同一个任务级 evaluator。
