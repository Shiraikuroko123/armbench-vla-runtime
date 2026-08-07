# 异步 Panda 闭环运行时

状态：当前工程与验收说明。

这个阶段把此前分开的模块接成了一条 CPU-only 闭环：

```text
MuJoCo Panda 状态快照
        |
        +--> latest-only 相机 worker --> 224x224 外部/腕部图像
                                           |
                                           v
                                      阻塞策略 worker
                                           |
                                           v
                                    带时间戳的 action chunk
                                           |
                                           v
                                  观测年龄后缀调度器
                                           |
                  +------------------------+------------------------+
                  |                        |                        |
              unguarded               旧 greedy guard          制动不变量修复
                  |                        |                        |
                  +------------------------+------------------------+
                                           |
                                  力矩控制 MuJoCo Panda
```

本基准使用的策略名为 `scripted_non_learned_async_reference`。它会真实地
阻塞指定墙钟时间，并可注入 jitter、响应丢失和错误关节速度。它不是
`pi0.5`，也不会下载或执行任何模型 checkpoint。

## 相比离线修复增加了什么

冻结响应实验只检查 action chunk，没有执行 Panda 反馈闭环。当前运行时还会：

- 通过按墙钟调度的 best-effort 周期控制循环推进 MuJoCo；
- 采集实时 Panda 状态和两路策略相机；
- 用 latest-only worker 将渲染和阻塞推理移出控制循环；
- 每个控制 tick 检查 dispatcher，而不是只在 action 边界检查；
- 跳过时间槽已经过去的 action index；
- repair 完成后再次检查 deadline，再决定是否激活动作；
- 通过 PD、偏置补偿和力矩限制执行 Panda 动力学；
- 在响应超时或失败时，从实测状态重新构造满足加速度限制并经过碰撞检查的
  完整停止序列；
- 记录控制 jitter、响应年龄、跟踪误差、干预、接触状态、command switch
  和 worker 身份。

deadline 前最后一个 action 可能只执行剩余的一部分墙钟时间。ArmBench 仍检查
它的完整关节空间边，因此对这一段的位置运动是保守检查；deadline 到达后，
再从实测状态按控制周期重建并检查终端制动。

## 本地运行

不需要 GPU、OpenPI 服务器或实体机器人。完成本地安装后，先运行一个 9 案例
短矩阵：

```powershell
& '.\.venv\Scripts\python.exe' -m armbench vla-panda-async-run `
  --output-directory results\async_panda_quick `
  --scenario single_block --quick --deadline-ms 400

& '.\.venv\Scripts\python.exe' -m armbench `
  vla-panda-async-validate results\async_panda_quick
```

`--quick` 会对三种运行时模式分别执行 0、80、240 ms 条件，并只运行有界的
参考轨迹前缀。去掉 `--quick` 后，默认矩阵还包括五档固定延迟、jitter、丢响应、
0.5 kg 负载和持续动作故障。

仓库原配置的响应 deadline 为 200 ms。如果本机 CPU 双相机采集本身已占据较大
预算，出现高 hold 率是合理结果。`--deadline-ms` 是明确的实验变量，不是暗中
放宽限制；解析后的数值会写入 artifact。

参考规划仍使用配置中的 20 mm 障碍膨胀。运行时检查默认继承同一余量，为闭环
跟踪误差和制动距离留出空间；只有做真实几何消融时才显式传入
`--runtime-clearance-mm 0`。两种 clearance 及运行时取值来源都会写入 provenance，
MuJoCo contact 仍作为单独的物理结果指标。

当前 MuJoCo checker 还会计算逐关节 workspace 运动上界，并细分每条边，使相邻
采样点之间的上界不超过声明障碍 clearance 的一半。机器人模型使用同一 clearance
膨胀后的障碍，因此在模型几何假设下，通过的边对静态障碍是保守检查；它不是连续
自碰证明、动力学扫掠体证明或真机安全保证。仓库保留的 v2 artifact 生成于该
checker 优化之前，故其 provenance 仍按分辨率有界检查解释。

## 查看机械臂运动

每个案例都会生成可由现有 viewer 回放的轨迹：

```powershell
& '.\.venv\Scripts\python.exe' -m armbench mujoco-view `
  --scenario single_block `
  --trace results\async_panda_quick\traces\case_000__fixed_000ms__unguarded.npz `
  --array actual_positions --play --loop
```

运行基准时添加 `--videos` 可生成 MP4。视频在测量结束后根据轨迹离线渲染，
因此录像和编码不会污染控制时延。

## 三种模式

| 模式 | 行为 |
| --- | --- |
| `unguarded` | 执行按观测年龄对齐的原始后缀，超时后直接 hold。 |
| `legacy_greedy` | 使用已有的逐步边界、slew 和碰撞回溯，超时后直接 hold。 |
| `braking_invariant` | 修复有界 chunk，检查终端停止路径，并在超时或失败时按控制频率重新制动。 |

这三种模式是工程控制组，不是学习策略 baseline。不同案例顺序执行时，墙钟调度
会有机器相关波动，因此当前 artifact 不构成“某方法任务成功率统计显著更高”的结论。

## Artifact 与独立验收

每次运行包含：

- `per_case.csv`：每个条件/模式一行；
- `events.jsonl`：观测、策略、调度、修复和 command 事件；
- `traces/*.npz`：墙钟时序、期望/实际关节位置、命令、响应年龄、接触和
  各阶段延迟；
- `summary.json` 与 `summary.md`：聚合结果和主张边界；
- `provenance.json`：完整矩阵、运行参数、依赖与模型身份、实现哈希和限制；
- `manifest.json`：文件清单、字节数、SHA-256 和清单总哈希。

validator 不只是检查文件哈希。它会从每个 NPZ 重算跟踪误差、最终误差、控制
jitter、延迟分位数、接触次数和 safe-success；再从 `events.jsonl` 独立重算
过期后缀、响应接受/拒绝、hold、制动、干预、加速度违规和制动修复选择预算
超限次数；软件预算不会被写成硬 deadline。因此，即使篡改 CSV 后重新生成
manifest，错误数字仍不能通过验收。

## 调试顺序

1. 先运行 `python -m armbench doctor`，排除 Panda 模型和 Python 环境问题。
2. 只运行一种模式、一个固定延迟和 `--max-reference-steps 10`。
3. 在 `per_case.csv` 中先看观测延迟、策略延迟、deadline rejection、hold 率和
   最大控制 tick lateness。
4. 按 `case_id` 筛选 `events.jsonl`，依次检查 `observation_outcome`、
   `policy_outcome`、`plan_prepared` 和 `command_switch`。
5. 回放 `actual_positions`，并核对同一 NPZ 中的接触和命令数组。
6. 任何数字写入简历或报告前，都要运行 `vla-panda-async-validate`。

## 已保留的 v2 矩阵

仓库保留的
[`async_panda_closed_loop_400ms_20mm_v2_001`](../reports/async_panda_closed_loop_400ms_20mm_v2_001/summary.md)
由 commit `6f21d2b6bdc925604f9241f92a1c310e84bf9e7e` 生成。27 个案例由三种
运行模式与以下条件交叉组成：0/40/80/160/240 ms 固定策略延迟、80 +/- 25 ms
jitter、10% 响应丢失、0.5 kg 负载和持续 2.5 rad/s 的 0 号关节动作故障。
响应 deadline 为 400 ms，规划与运行时 clearance 均为 20 mm，控制/动作周期
分别为 10/66.7 ms。

| 模式 | 满足物理谓词 | 到达目标 | 突停违规 | 修复预算超限 | 修复 P95 / 最大值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `unguarded` | 8/9 | 6/9 | 211 | 0 | 0 / 0 ms |
| `legacy_greedy` | 9/9 | 1/9 | 266 | 0 | 5.36 / 20.07 ms |
| `braking_invariant` | 9/9 | 1/9 | 0 | 0 | 5.99 / 11.47 ms |

物理谓词表示实测 trace 中障碍/自碰接触步数和关节限位违规步数均为 0。
unguarded 的持续故障案例因关节限位违规失败，并非障碍接触。三种模式都按观测
年龄调度后缀：artifact 共记录 8,193 条过期索引之后的命令，index 0 命令为 0。
160 ms 条件的跨模式平均 hold 率为 0.892，240 ms 为 1.000，明确显示本地 CPU
吞吐边界。每个条件只有一次墙钟运行，因此这是工程验收证据，不是统计方法比较。

## 主张边界

这是 Panda 闭环物理执行与异步运行时证据，不是学习策略任务结果，不是官方
`pi0.5` 到 Panda 的端到端部署，不是操作系统硬实时保证，不是连续碰撞认证，
也不是实体机器人安全验证。下一研究阶段是在不修改运行时和 validator 的前提下，
用同一 `ActionChunkPolicy` 契约替换 scripted policy，并执行预注册、多随机种子的
正式矩阵。
