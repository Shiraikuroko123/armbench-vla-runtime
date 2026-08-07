# 非阻塞运行时验收

状态：当前组件级验证

已完成的 `pi0.5`-LIBERO 实验采用 blocking inference，再对等待期间进行
simulator catch-up。当前维护版本增加了不依赖 GPU 的独立开发验收，用于逐步
消除这个架构限制。

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

下一步应将 worker/dispatcher 接到官方策略客户端与持续推进的 simulator 之间，
并在 actuator dispatch 前把选中的动作后缀送入现有运动学 guard。
