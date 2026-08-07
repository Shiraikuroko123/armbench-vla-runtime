# 延迟有界的终端制动不变量动作修复

状态：当前离线集成证据。更新日期：2026-08-07。

## 为什么要增加这一阶段

第一阶段 Panda 回放暴露了一个明确的运行时失败模式。旧的 greedy guard
逐步检查动作，并可能选择一条通过碰撞检查的停止或保持路径；但如果上一条命令
仍有较大速度，立即切换到保持命令可能超过关节加速度上限。冻结回放中有
6/270 个案例出现了这个冲突。

本阶段测试一个不训练模型的运行时干预：为整个动作块选择一个可行速度比例，并
验证该轨迹可以在不违反同一组约束的情况下减速到零。目标是消除执行边界上的
“可行域为空”冲突，不是提升 VLA 策略能力，也不主张任务成功率提升。

## 从 `pi0.5` 到七自由度 Panda 的完整边界

输入是此前由 Physical Intelligence 官方 `pi0.5` LIBERO checkpoint 生成、并经过
哈希核验的冻结响应。每个响应是 `10 x 7` 的 LIBERO 风格笛卡尔动作块；本地
适配器先把它转换成 Panda 运行时的 `10 x 8` 关节速度/夹爪契约，再进入两种
guard 的成对比较。

```text
官方 checkpoint 的冻结 pi0.5 响应
              |
              v
       LIBERO 语义与动作裁剪
              |
              v
 Panda Jacobian 微分逆运动学（H x 7 -> H x 8）
              |
       +------+------+
       |             |
       v             v
  旧逐步 guard   轨迹级制动修复
       |             |
       +------+------+
              v
 MuJoCo Panda 构型、边碰撞和终端制动检查
              |
              v
      成对 CSV、JSON、NPZ 与 manifest
```

三个 Panda 场景（`free_space`、`single_block`、`narrow_gate`）每个案例都独立
重置。回放不会重新执行 checkpoint，不把 Panda 观测反馈给策略，也不会产生新的
策略推理。

## 修复算法

`BrakingTrajectoryGuard` 先建立比例为 0 的 hold 候选，然后按降序最多检查五个
比例：`1.0, 0.75, 0.5, 0.25, 0.0`。每个候选依次完成：

1. 将关节速度裁剪到 Panda 与运行时上限；
2. 在每个控制周期施加关节加速度 slew 上限；
3. 在关节空间积分并检查构型限位；
4. 对插值后的 MuJoCo 边进行碰撞检查；
5. 追加有界减速轨迹，并要求最终速度降为零。

选择第一个可行的非零比例。响应超时、状态不一致、修复选择超时或连停止路径
都不可用时，运行时 fail-closed 到 hold。20 ms 是测得的软件选择预算，不是
操作系统级硬实时保证。修复器不优化任务进度：必要时会用更小的速度换取可验证的
停止路径。

## CPU 复现

在仓库根目录的 Windows PowerShell 中执行：

```powershell
& '.\.venv\Scripts\python.exe' -m armbench vla-panda-braking-repair `
  evidence\pi05_rtc_overlap_primary_v3_seed_20260807_001\evaluation `
  --output-directory results\pi05_panda_braking_repair_90_001 `
  --chunks 90 --selection-seed 20260807

& '.\.venv\Scripts\python.exe' -m armbench `
  vla-panda-braking-repair-validate `
  results\pi05_panda_braking_repair_90_001 `
  --source-directory `
  evidence\pi05_rtc_overlap_primary_v3_seed_20260807_001\evaluation
```

仓库已保存的报告是
[报告摘要](../reports/pi05_panda_braking_repair_90_001/summary.md)。固定版本
MuJoCo Panda 模型准备好后，这条命令只需要 CPU，不下载 checkpoint，也不需要
OpenPI 服务。

## 可视化验收

NPZ 中保存了原始、旧 guard、修复后以及终端制动的关节轨迹。可以用交互式 MuJoCo
查看第一阶段冲突案例（例如 `trajectory_index=101`）：

```powershell
& '.\.venv\Scripts\python.exe' -m armbench mujoco-view `
  --scenario narrow_gate `
  --trace reports\pi05_panda_braking_repair_90_001\trajectories.npz `
  --array legacy_positions --episode 101 --play

& '.\.venv\Scripts\python.exe' -m armbench mujoco-view `
  --scenario narrow_gate `
  --trace reports\pi05_panda_braking_repair_90_001\trajectories.npz `
  --array repair_positions --episode 101 --play
```

先在 `per_case.csv` 中按 `trajectory_index` 查场景、方法和源响应哈希，再打开
对应数组。可视化用于调试和面试演示，最终结论仍以 validator 为准。

## 成对结果

| 指标 | 旧 greedy guard | 终端制动不变量修复 |
| --- | ---: | ---: |
| 满足全部已注册约束 | 264/270 | 270/270 |
| 0.02 rad 边采样下路径有效 | 270/270 | 270/270 |
| 避碰/加速度冲突 | 6 | 0 |
| 修复回归 | - | 0 |
| 软件选择延迟 P95 | 13.550 ms | 12.777 ms |
| 软件选择延迟最大值 | - | 19.979 ms |
| 超过选择预算 | - | 0/270 |
| 终端制动路径有效 | - | 270/270 |

选中的比例分布为 `1.0:233`、`0.75:3`、`0.5:9`、`0.25:21`、`0.0:4`。四个
零比例案例包含已注册的 deadline 或 fallback 条件，按 hold 行为单独记录，没有
伪装成正常策略执行。

## 证据与主张边界

报告把每一行绑定到源响应哈希，并保存实现哈希、运行环境、轨迹数组和文件清单。
validator 会重新核验源 archive，从 CSV 重算汇总，并检查轨迹身份与形状。

该结果支持的工程结论是：在这套冻结 MuJoCo 诊断矩阵中，有限候选搜索消除了
6 个已注册的碰撞/加速度冲突，且没有观测到选择预算超限。它不支持以下主张：

- `pi0.5` 任务成功率提升；
- `pi0.5` 到 Panda 的在线反馈闭环；
- 连续碰撞检测或物理安全认证；
- 最坏情况硬实时调度；
- 超出当前 checkpoint archive 和三个 Panda 场景的泛化。

因此，对外最准确的名称是：**构建在七自由度 Panda 执行基座上的 VLA 运行时
guard 与评测组件**。接入实时 VLA evaluator、第二个策略家族以及硬件时序/安全
证据，仍是后续独立里程碑。
