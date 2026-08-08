# Panda 集成动作保障链

状态：Current。更新日期：2026-08-08。

## 工程问题

动作块策略即使返回了形状正确、数值有限的张量，也不代表它此刻可以直接下发。
响应可能已经过期、对应旧的机器人状态、违反关节运动学约束、在两个离散路点之间
穿过障碍，或者让机械臂在已注册的模型约束内无法停下。

ArmBench 的集成 Panda supervisor 是一个运行在 CPU 上的 policy-to-controller
参考实现。只有完整动作块通过全部注册检查后才会被接受；一旦任一检查失败，已经
通过的前缀也不会作为可执行动作泄漏出去。

## 保障链

```text
实测 q/qdot + 观测 q + 响应年龄 + Hx8 动作块
                         |
                         v
                 deadline 与状态对齐门禁
                         |
                         v
      OSQP 投影：关节位置、速度、加速度和声明的线性约束
                         |
                         v
            每条边的连续静态障碍/自碰撞证书
                         |
                         v
              每个动作边界的动力学可行停止证书
                         |
                         v
      accepted | verified_brake | hold | unrecoverable_stop
```

OSQP 负责运动学 box 和显式线性约束。碰撞不在 QP 中伪装成凸约束，而是在投影后
由独立的 fail-closed 连续边 checker 验证。停止可行性使用 MuJoCo 逆动力学和已
注册的执行器余量进行采样检查。

四种原子结果的含义是：

- `accepted`：完整投影动作块可执行，并且每个动作边界都有停止证书；
- `verified_brake`：拒绝策略动作，但实测运动状态存在已验证的制动轨迹；
- `hold`：拒绝策略动作，静止状态存在已验证的零运动回退；
- `unrecoverable_stop`：模型内连回退都无法通过证书，不输出策略动作，必须交给
  更高层处理。

## 已保存证据

### 注册故障矩阵

[`integrated_panda_fault_matrix_001`](../reports/integrated_panda_fault_matrix_001/summary.md)
包含三个场景、两种负载，以及 nominal、速度尖峰、过期响应、状态不一致、边中间
自碰撞控制和近关节限位停止状态，共 27 个确定性案例。

- 27/27 个注册预期结果均可重算；
- 12 个完整计划被接受；
- 6 个过期响应进入 `verified_brake`；
- 7 个案例 fail-closed 到 `hold`；
- 2 个近限位状态被报告为 `unrecoverable_stop`；
- 共检查 124 条连续边；
- 被拒绝案例均未输出部分策略动作。

记录主机上的 supervision P95 为 588.49 ms，最大值为 2.972 s。这是离线 CPU
参考耗时，不是实时性能结论。

### MuJoCo 闭环任务执行

[`integrated_panda_task_001`](../reports/integrated_panda_task_001/summary.md)
会重新构建 RRT-Connect 参考轨迹，先对完整动作块执行保障，再将接受的轨迹送入
力矩控制 MuJoCo 物理执行。

| 案例 | 条件 | 最终目标误差 | 跟踪 RMSE | 注册结果 |
| --- | --- | ---: | ---: | --- |
| `single_block_goal` | 0 kg、0 ms 延迟 | 0.00343 rad | 0.00607 rad | 到达目标，满足物理谓词 |
| `narrow_gate_payload_delay_goal` | 0.5 kg、80 ms 反馈延迟 | 0.01962 rad | 0.03057 rad | 到达目标，满足物理谓词 |

两个案例合计 351/351 条运动边通过证书，351/351 个动作边界保留动力学可行停止。
物理执行记录到 0 个障碍接触 step、0 个自碰撞 step、0 个关节限位违规 step 和
0 次力矩饱和。

任务 checker 只有一条显式 allowed-collision 规则：固定张开的夹爪不检查
`left_finger`/`right_finger` 这个 body pair。它在编译模型中展开为 36 个 geom
pair，并不是任意关闭 36 对机械臂自碰撞；其余注册的 Panda 自碰撞对全部保留。

两条完整轨迹的离线 supervision 分别耗时 5.27 s 和 10.20 s，而且都在物理执行
前完成。因此这组证据补齐了本地“规划-保障-执行”接线，但不能证明在线 deadline。

## 本地验收与可视化

在仓库根目录执行：

```powershell
& '.\.venv\Scripts\python.exe' -m armbench vla-integrated-fault-validate `
  reports\integrated_panda_fault_matrix_001

& '.\.venv\Scripts\python.exe' -m armbench vla-integrated-task-validate `
  reports\integrated_panda_task_001

& '.\.venv\Scripts\python.exe' -m armbench vla-cpu-runtime-validate `
  reports\cpu_runtime_completion_001
```

第一个 validator 会重建注册输入并重跑 27 个 supervisor 决策；第二个会重新运行
规划、动作保障、MuJoCo 物理、轨迹数组与汇总指标；第三个会重放 17 个 provider、
异步保障和原子发布场景。它们都不只是读取 JSON。

仓库还提供与启动目录无关的一键脚本。在仓库根目录执行；若从其他目录启动，则向
`-File` 传入脚本的绝对路径，不再依赖容易出错的 `..\.venv`：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  '.\scripts\accept_integrated_panda.ps1'

powershell -NoProfile -ExecutionPolicy Bypass -File `
  '.\scripts\accept_integrated_panda.ps1' `
  -Visualize -Case narrow_gate_payload_delay_goal
```

可视化播放的是保存的 `actual_positions` 实测轨迹。viewer 用于直观检查，正式验收
仍以 manifest 保护、能够重跑物理的 validator 为准。

## 能公开说什么

准确表述是：**实现了将动作块经过 OSQP 运动学投影、连续静态/自碰撞证书和逐边界
动力学可行停止检查后，再送入七自由度 Panda 力矩控制仿真的原子保障链；注册故障
矩阵和两条闭环任务均可在本地重算。**

动作源仍是 scripted RRT-Connect 参考，不是学习式 VLA；任务是关节路点到达，不是
抓取或物体操作；结果不是真机、硬实时或安全认证。验收这些现有结果不需要 GPU 或
服务器。

真正需要 GPU 的下一步，是用至少两个学习式 VLA 的冻结输出、再用在线输出替换
scripted 动作源，冻结各自精确动作语义，并在异步任务循环中评测同一 supervisor。
若要主张在线 deadline，还必须进一步降低或并行化连续碰撞与停止证书的计算耗时。
