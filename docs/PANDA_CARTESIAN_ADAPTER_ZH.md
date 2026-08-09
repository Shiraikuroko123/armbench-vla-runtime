# LIBERO 到 Panda 的笛卡尔动作适配器

状态：当前组件实现；已由 scripted CPU 验收、冻结响应离线回放和一次真实 checkpoint
集成门分别验证

## 作用

该适配器补上 ArmBench 两条验证路径之间的一处软件边界：把标记为
`libero.ee_delta_pose_gripper.v1` 的有限 `H x 7` 末端动作块，转换为本地
Panda guard 使用的 `H x 8` 契约，即 7 个关节速度和 1 个归一化夹爪命令。

下面的默认 CPU 验收命令使用脚本生成的笛卡尔动作，不运行模型推理；该命令本身
不能证明学习策略已经部署。G01 则在另一条证据路径中，把官方 OpenPI
`pi05_libero` checkpoint 的实时响应接入了异步 Panda runtime，并实际经过本适配器。
这证明 checkpoint-to-runtime 接线成立，不证明官方任务能力或真机部署。

## 转换流程

每一步动作依次执行：

1. 按声明范围裁剪 6 维运动输入；
2. 按 LIBERO 的 `0.05 s` 控制周期，使用经源码核对的 `0.05 m` 平移和
   `0.5 rad` 旋转缩放；
3. 默认按 base/world frame 解释动作；显式配置 tool frame 时再转换；
4. 计算 Menagerie Panda hand body 的 6 x 7 几何 Jacobian；
5. 使用阻尼最小二乘求差分逆运动学；
6. 按关节速度上限和关节限位余量统一缩放；
7. 把 LIBERO 的 `-1=张开、+1=闭合` 反向映射为本地 Panda 的
   `0=闭合、1=张开`；
8. 在输出动作块中保留来源、观测序号、推理耗时和接收时间。

保存的官方响应可能超过名义输入范围。适配器与 robosuite 一致，会记录这些
步骤并在缩放前裁剪。

转换结果继续进入已有 `ActionChunkGuard`，由 guard 执行跨步加速度限制、
关节状态检查、分辨率有界的碰撞边检查、deadline/state-mismatch latch、
backtracking 和 hold 回退。

## 本地验收

完成普通本地安装后运行：

```powershell
& '.\.venv\Scripts\python.exe' -m armbench vla-panda-adapter-smoke
```

命令会生成确定性的 10 步末端动作，经 MuJoCo Panda Jacobian 和 guard 后，
检查输出有限、手部沿目标方向移动且受保护路径有效。报告固定标注为
`scripted_cartesian_adapter_component_only`。

## 不能据此声称

- 默认动作语义已核对 OpenPI 固定的 LIBERO 子模块 commit `f78abd68` 与
  robosuite `1.4.1`：环境频率为 20 Hz，`OSC_POSE` 输入裁剪到 `[-1,1]`，
  平移缩放为 `+/-0.05 m`、旋转缩放为 `+/-0.5 rad`，增量位于
  base/world frame；Panda gripper 源码规定 `-1=张开、+1=闭合`。
- 阻尼最小二乘加确定性缩放不等于 QP 安全滤波器。
- 本地关节速度微分逆解不等价于 robosuite 的 torque-level OSC 动力学。
- 本地运动学控制点是 Menagerie 的 `hand` body 原点，不是 robosuite 的
  `grip_site`；运行报告会显式给出该差异。
- 当前碰撞检查是关节空间插值采样，不是连续碰撞认证。
- CPU smoke 不使用 OpenPI、`pi0.5`、LIBERO 任务执行、真机、ROS2 或安全 PLC。

## 集成状态

保存的官方 LIBERO policy 响应已经接入该 adapter，并完成严格校验的离线回放，详见
[冻结 pi0.5 响应的 Panda 离线回放](PI05_PANDA_ARCHIVE_REPLAY_ZH.md)。

G01 随后把 Panda 双相机实时观测送入官方 checkpoint，将 35 个被接收的响应经本组件
转换，再交给 measured-age dispatcher、制动修复和 MuJoCo 力矩闭环执行。保留的
[真实 checkpoint 集成 artifact](../evidence/g01_live_panda_smoke_final_001/summary.md)
记录了 `target_reached=false`，因此它是 integration gate，不是复现 LIBERO 分数或
Panda 任务成功率。

剩余任务级里程碑是具有明确成功判据、统计报告和多 seed 的任务对齐冻结协议。真机部署
还需要具体驱动、标定时序、watchdog、急停集成和实体故障注入证据。
