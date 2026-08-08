# Panda 连续自碰撞审计

状态：当前 CPU-only 几何审计。

本地 Panda checker 现在可以在关节空间线性边上，对自碰撞几何对执行保守的距离上界
证书。该审计把证书结果与同一份编译版 Menagerie 几何上的更密 MuJoCo 采样 oracle
逐边对照。矩阵包含固定的“端点安全、边中间自碰撞”机制控制，以及固定 seed 的局部和
全局随机边；端点是否安全与整条边是否安全分别记录。

## 运行

```powershell
& '.\.venv\Scripts\python.exe' -m armbench mujoco-self-collision-audit `
  --output-directory reports\mujoco_self_collision_audit_001 `
  --samples-per-stratum 24 --dense-resolution-rad 0.002

& '.\.venv\Scripts\python.exe' -m armbench mujoco-self-collision-validate `
  reports\mujoco_self_collision_audit_001
```

输出目录可以独立验收。`per_edge.csv` 保存边端点、连续证书状态与碰撞 witness、
dense oracle 判定以及两种耗时；`manifest.json` 保护所有生成文件，摘要还记录 Panda
场景和实现代码哈希。

## 已保存结果

保留的 72 条边报告位于
[`reports/mujoco_self_collision_audit_001`](../reports/mujoco_self_collision_audit_001/summary.json)。
其中 70 条边的两个端点均无碰撞，false-safe 为 0，保守拒绝为 21。也就是说，在这
份采样 oracle 下，连续 checker 没有放过 oracle 判定为碰撞的边，但会拒绝一部分 oracle
认为安全的边。

这只是对关节空间线性插值和固定 MuJoCo 几何的实现审计。dense oracle 是采样证据，
不是解析证明；结果不等于实体机器人安全、硬实时、急停或真实负载动力学验证。
