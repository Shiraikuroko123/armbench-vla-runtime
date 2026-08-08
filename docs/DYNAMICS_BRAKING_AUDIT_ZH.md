# Panda 动力学可达制动

此前的 braking invariant 检查关节位置、速度、加速度和碰撞约束，但它没有回答：
负载或关节阻尼变化后，MuJoCo Panda 模型是否还能产生停止轨迹所需的扭矩。本阶段新增
fail-closed 的逆动力学边界。

对于注册的初始关节速度，ArmBench 生成同步恒减速度停止轨迹。每个采样状态都要满足
Panda 的关节位置、速度和加速度限制；相邻位置使用连续 MuJoCo 几何距离证书检查；随后
使用 `mujoco.mj_inverse` 复算广义力，并与编译后执行器力范围的 80% 比较。执行器映射
缺失或不唯一、动力学结果非有限、碰撞、越限或扭矩过大都会在执行前拒绝整条停止轨迹。

仓库保留的审计包含 45 个确定性条件：

- payload：0、0.5、1 kg；
- 机械臂关节粘性阻尼倍率：0.5、1、2；
- 初速度：静止、低速/高速正向和低速/高速反向。

45 条注册停止轨迹全部通过。最长停止时间为 0.20 s，最大关节空间停止距离为
0.159374 rad，最大执行器限值占用比例为 0.424543。artifact 将逐案例结果写入 CSV，
使用递归 SHA-256 manifest 绑定全部文件，并在验证时重新运行所有 MuJoCo 逆动力学与
连续碰撞决策。

```powershell
& '.\.venv\Scripts\python.exe' -m armbench `
  mujoco-dynamics-braking-validate `
  reports\dynamics_braking_audit_001
```

这只是基于模型和离散时间采样的可行性证据，不证明闭环跟踪、jerk 上界、操作系统硬实时、
MuJoCo 与真实 Panda 的动力学一致性，也不构成经过认证的急停功能。
