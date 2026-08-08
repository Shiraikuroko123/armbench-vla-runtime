# Panda 异步动作保障：CPU 收口与验收

状态：Current / Operational
运行成本：本机 CPU，¥0
正式证据：`reports/cpu_runtime_completion_001/`

## 它解决什么问题

VLA 或其他策略输出的是一段动作，而不是机械臂可以无条件执行的命令。策略调用可能
阻塞、超时、断连、返回错误形状或旧序列；即使返回值合法，机器人也可能已经移动，
动作中的后续边还可能发生连续碰撞，当前速度也可能无法在限位和力矩约束下停止。

本模块把这些问题放进同一条可重放的 CPU 运行时边界：

```text
VLAObservation
    -> provider / policy worker             独立线程
    -> ActionChunk (H x 8)
    -> IntegratedPandaSupervisor worker     独立线程
         -> OSQP 运动学投影
         -> 连续静态/自碰撞证书
         -> 动力学可达制动证书
    -> AtomicPandaPlanGate                  控制线程提交点
         -> execute 整段动作
         -> 或发布零个策略动作并 hold / brake / stop
```

这里的核心不是“让脚本策略看起来像 VLA”，而是固定模型输出之后的工程契约。真实
OpenPI provider、冻结 provider 和测试 provider 必须先转换成同一个 `ActionChunk`
语义，之后才能共用保障链。

## 这次本地收口增加了什么

`src/armbench/vla/integrated_panda_async.py` 增加了两个运行时组件：

- `LatestIntegratedPandaWorker`：在独立线程完成整段 QP、连续碰撞和制动检查；控制
  线程只轮询结果，不直接执行高开销监督。
- `AtomicPandaPlanGate`：动作激活前重新检查 deadline、机器人状态漂移、请求顺序和
  reset generation。只有完整计划仍有效时才发布；任何拒绝均返回空动作数组。

`src/armbench/vla/cpu_runtime_completion.py` 把 provider 和保障线程接成 17 个固定案例：

| 类别 | 注册条件 | 验收含义 |
| --- | --- | --- |
| 延迟 | 0/40/80/160 ms | 策略阻塞时控制线程仍持续 tick |
| provider | mock、冻结响应、OpenPI 接口 fixture | 三类来源进入同一动作契约 |
| 非法响应 | shape、NaN、断连、超时、sequence mismatch | provider 失败不进入监督或执行 |
| 时序与状态 | 过期观测、状态错位、reset replay | 旧结果和错位结果 fail closed |
| 监督边界 | 超预算、近限位不可恢复停止 | 明确区分 hold 与 stop |
| 几何机制控制 | 端点安全但中间自碰撞 | 连续边检查拒绝离散采样可能漏掉的碰撞 |

正式结果为 17/17 个预期状态匹配：6 个完整计划、10 个 hold、1 个不可恢复停止，
拒绝路径暴露的部分策略动作前缀为 0。

## 一键验收

在仓库根目录运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  '.\scripts\accept_integrated_panda.ps1'
```

脚本会自行寻找工作区 `.venv`，因此从 `C:\WINDOWS\system32` 启动时也不依赖
错误的相对 Python 路径。它依次重算 27 案例同步故障矩阵、17 案例异步发布矩阵和
2 条 MuJoCo 任务。

只验收本报告时运行：

```powershell
& 'D:\arm-planning-control-project\.venv\Scripts\python.exe' -m armbench `
  vla-cpu-runtime-validate `
  'D:\arm-planning-control-project\project\reports\cpu_runtime_completion_001'
```

validator 不只是检查文件是否存在。它会验证精确文件集合和 SHA-256，核对源码哈希，
交叉比较 CSV/JSON/Markdown，重算 provider 聚合，并重新运行 17 个注册场景。修改结果
后重新生成 manifest 也不能绕过语义重放。

## 如何调试

推荐先从一个专项测试开始：

```powershell
$Python = 'D:\arm-planning-control-project\.venv\Scripts\python.exe'
Set-Location 'D:\arm-planning-control-project\project'
& $Python -m pytest tests\test_integrated_panda_async.py -q
& $Python -m pytest tests\test_cpu_runtime_completion.py -q
```

调试器中的第一组断点：

1. `cpu_runtime_completion.py::_run_case`：观察 policy 和 assurance 两个阶段的输入。
2. `async_worker.py::LatestPolicyWorker._run`：检查 provider 异常如何进入
   `PolicyOutcome`。
3. `integrated_panda_async.py::LatestIntegratedPandaWorker._run`：检查 supervisor
   是否在非控制线程执行。
4. `integrated_panda_guard.py::IntegratedPandaSupervisor.supervise`：查看失败发生在
   deadline、状态对齐、QP、连续碰撞、制动还是预算阶段。
5. `integrated_panda_async.py::AtomicPandaPlanGate.commit`：查看最终为什么执行、hold
   或拒绝重放。

关键数据形状和状态机见[中文代码导读](CODE_WALKTHROUGH_ZH.md)。

## 可视化与数字证据的关系

17 案例矩阵主要验证并发和发布语义，正式验收是可重算表格，不适合用一段动画代替。
机械臂是否按通过保障的轨迹运动，由已保存的 MuJoCo 任务 trace 可视化：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  '.\scripts\accept_integrated_panda.ps1' `
  -Visualize -Case narrow_gate_payload_delay_goal
```

viewer 展示 `actual_positions`，用于直观检查负载和反馈延迟下的物理执行；碰撞、制动、
哈希和原子发布仍以 validator 为验收权威。

## 主张边界

这份 CPU 证据没有运行学习式 checkpoint。OpenPI 接口 fixture 只证明调用契约能够接入，
不证明 `pi0.5` 任务成功；任务轨迹来自 scripted RRT-Connect。线程调度是 Windows/Python
best effort，不是硬实时；MuJoCo 证书不是实体机器人安全认证。要补齐这些边界，后续仍需
GPU checkpoint、独立时钟任务评测，以及 Ubuntu/ROS2/MoveIt 或真实执行器实验。
