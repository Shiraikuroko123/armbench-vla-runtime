# LIBERO 到 Panda 的笛卡尔动作适配器

状态：当前组件级实现

## 作用

该适配器补上 ArmBench 两条验证路径之间的一处软件边界：把标记为
`libero.ee_delta_pose_gripper.v1` 的有限 `H x 7` 末端动作块，转换为本地
Panda guard 使用的 `H x 8` 契约，即 7 个关节速度和 1 个归一化夹爪命令。

这不等于官方 `pi0.5` checkpoint 已经控制本地 Panda。当前验收命令使用
脚本生成的笛卡尔动作，不运行模型推理。

## 转换流程

每一步动作依次执行：

1. 按声明范围裁剪 6 维运动输入；
2. 应用显式平移和旋转缩放；
3. 必要时把 tool-frame 命令转换到 MuJoCo world frame；
4. 计算 Menagerie Panda hand body 的 6 x 7 几何 Jacobian；
5. 使用阻尼最小二乘求差分逆运动学；
6. 按关节速度上限和关节限位余量统一缩放；
7. 把 `[-1, 1]` 的夹爪命令映射为 `[0, 1]`；
8. 在输出动作块中保留来源、观测序号、推理耗时和接收时间。

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

- 默认缩放只是保守的组件测试参数，正式 LIBERO 实验前必须从上游 OSC
  controller 冻结单位、坐标系、裁剪和夹爪语义。
- 阻尼最小二乘加确定性缩放不等于 QP 安全滤波器。
- 当前碰撞检查是关节空间插值采样，不是连续碰撞认证。
- smoke 不使用 OpenPI、`pi0.5`、LIBERO 任务执行、真机、ROS2 或安全 PLC。

## 下一里程碑

在独立控制时钟持续推进仿真的同时，让官方 LIBERO policy 在现有 worker 中
推理；在同一冻结配对协议下比较无 guard、观测年龄对齐、观测年龄对齐加
可行性约束三种模式。正式解释任务结果以前，必须用实际 controller 配置
替换组件测试缩放并保存其来源。
