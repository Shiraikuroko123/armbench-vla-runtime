# 本地 CPU 一键验收

`scripts/accept_cpu.py` 是仓库的总验收入口。它只重跑已经保存的 artifact
validator，不修改 `reports/` 中的正式证据；本地日志写入被 Git 忽略的
`output/cpu_acceptance/`。

## 快速验收

在仓库根目录运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  '.\scripts\accept_cpu.ps1'
```

PowerShell 包装器会同时检查仓库内和外层工作区的 `.venv`，所以从
`C:\WINDOWS\system32` 启动时也可以向 `-File` 传入它的绝对路径，不需要手写 Python
相对路径。

Ubuntu：

```bash
./.venv/bin/python scripts/accept_cpu.py
```

输出会逐项显示 `passed / skipped / failed`，并生成：

- `output/cpu_acceptance/summary.md`：适合人工查看；
- `output/cpu_acceptance/summary.json`：适合脚本或面试现场复核。

默认检查包括环境、MuJoCo 场景、QP 投影、连续静态/自碰撞、动力学制动、
provider 语义契约、LeRobot 风格 watchdog、冻结 `pi0.5` 响应回放、制动修复、
27 案例同步 supervisor、2 条 Panda 任务、27 案例异步闭环、17 案例异步边界和
G02 的 40-rollout 独立时钟核心 artifact、单条可视化 artifact 与证据目录。
这些步骤均不需要 GPU、服务器或实体机械臂；G02 validator 重算已保存的请求、
action chunk、tick、初始状态和 provenance，不会重新执行 checkpoint。

## 官方 LeRobot 环境

官方 `lerobot==0.4.4` 使用独立 NumPy 2 环境。脚本会自动探测
`.venv-lerobot-0.4.4`、`.venv-lerobot` 和 `.venv-lerobot-cpython`；也可以显式指定：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  '.\scripts\accept_cpu.ps1' `
  -OfficialPython 'D:\arm-planning-control-project\.venv-lerobot-cpython\Scripts\python.exe'
```

没有隔离环境时，官方 round-trip 标记为 `skipped`，其余 CPU 验收仍会运行。需要把
它作为硬性门槛时使用 `--require-official`。

## 完整测试

在提交前追加 `--full-tests`，会在 artifact 验收后运行完整 `pytest`：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  '.\scripts\accept_cpu.ps1' -FullTests
```

验收通过只说明保存的输入、实现哈希、清单和 validator 结果一致。G02 提供独立
时钟的真实 checkpoint 仿真 pilot，但仍不等于 `pi0.5` 端到端控制 Panda、官方
排行榜成绩、真机、硬实时或安全认证结果。
