# Panda 原子窗口 CPU 重复性审计

状态：原子认证窗口审计的后续阶段已经完成。正式 artifact 位于
[`reports/pi05_windowed_cpu_repeatability_180_002`](../reports/pi05_windowed_cpu_repeatability_180_002/summary.md)。

## 要回答的问题

配对窗口审计显示：在注册的软件预算下，认证一个动作窗口 `H=1` 比认证完整的
十步动作块 `H=10` 更容易完成。本阶段检验这一发布边界能否在独立 Python 进程和
受控主机竞争下重复出现。实验不重新运行 `pi0.5` checkpoint，也不生成任务成功率。

## 固定矩阵

协议见
[`pi05_windowed_cpu_repeatability_protocol_20260811.json`](research/pi05_windowed_cpu_repeatability_protocol_20260811.json)，固定：

- 3 次空闲、3 次 4-worker CPU 负载冷启动；
- 同一批 30 个冻结 `pi0.5` 响应与 3 个 Panda 场景；
- 成对比较 `H=10` 完整动作块和 `H=1` 原子认证窗口；
- 20 ms 监督预算、200 ms 响应 deadline、5 ms QP 步预算；
- 每轮完整保存 child 日志、嵌套 manifest、SHA-256 绑定，并由独立根 validator
  重算。

`180_002` 的 6 轮全部通过验证，未重新执行 checkpoint。

## 结果

| 条件 | Profile | Execute 计数 | supervisor P95，均值 +/- 标准差 | 安全候选 | 不安全窗口 | 部分窗口 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 空闲 | `full_chunk_h10` | `[0, 0, 0]` | 27.175 +/- 1.347 ms | 66/90 | 0 | 0 |
| 空闲 | `certified_window_h1` | `[90, 90, 90]` | 6.585 +/- 0.980 ms | 90/90 | 0 | 0 |
| 4 个 CPU worker | `full_chunk_h10` | `[0, 0, 0]` | 31.355 +/- 2.096 ms | 66/90 | 0 | 0 |
| 4 个 CPU worker | `certified_window_h1` | `[89, 84, 86]` | 19.980 +/- 5.300 ms | 90/90 | 0 | 0 |

这支持一个范围明确的工程结论：把认证发布窗口缩短为一个动作后，在本机上比完整
动作块更容易重复完成，同时仍保持窗口级 all-or-none 边界。它不等于硬实时、真机
安全或部署性能证明。`H=1` 改变了源动作块的发布契约：十步响应会作为多个独立认证
窗口逐步暴露。

## 本地验收

在仓库根目录的 Windows PowerShell 执行：

```powershell
& '.\.venv\Scripts\python.exe' -m armbench `
  vla-panda-windowed-repeatability-validate `
  reports\pi05_windowed_cpu_repeatability_180_002 `
  reports\pi05_integrated_panda_cpu_replay_270_001
```

validator 会重算根 inventory、协议与输入绑定、6 个 child replay manifest、候选
运动学、连续碰撞与制动谓词、发布长度以及条件级统计。已验证根 inventory 的
SHA-256 为 `7cc99c4420b706a30fcb3beee774b937d0a3eb619fa17f704f59419c89323306`。

如需重新生成实验，请使用新的输出目录，不要覆盖正式 artifact：

```powershell
& '.\.venv\Scripts\python.exe' -m armbench `
  vla-panda-windowed-repeatability `
  reports\pi05_integrated_panda_cpu_replay_270_001 `
  --output-directory results\pi05_windowed_cpu_repeatability
```

## 主张边界

每个条件只有 3 次重复，属于描述性工程证据，不是推断性时延研究。CPU 负载只是
有限的本机压力条件，不是硬件认证。冻结响应仍是跨控制器 Panda 输入，Panda 观测
没有回传给策略。本审计不衡量任务成功、硬实时调度、真机安全或跨机器泛化。
