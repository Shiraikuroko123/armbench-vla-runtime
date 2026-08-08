# MuJoCo clearance-backed swept 碰撞审计

状态：当前 CPU-only 碰撞验证方法与审计。

Panda runtime 使用配置的静态障碍余量构造 guard 场景。swept checker 根据编译后的
Menagerie 几何，为每个机械臂关节计算保守的工作空间位移半径；随后不断细分边，直到
每个子边的最大位移上界不超过静态 clearance 的一半，同时保留 joint-resolution
下界。

这是针对该 clearance 所表示静态障碍的保守证书。本 artifact 不包含独立的连续自碰撞
矩阵，见[Panda 连续自碰撞审计](MUJOCO_SELF_COLLISION_AUDIT_ZH.md)。下面的 dense
对照是更密的采样 oracle，不是解析证明。

## 运行

```powershell
& '.\.venv\Scripts\python.exe' -m armbench mujoco-swept-audit `
  --output-directory reports\mujoco_swept_audit_001 `
  --samples-per-scenario 24 --clearance-mm 20 `
  --sampled-resolution-rad 0.05 --dense-resolution-rad 0.002

& '.\.venv\Scripts\python.exe' -m armbench mujoco-swept-validate `
  reports\mujoco_swept_audit_001
```

审计覆盖 `free_space`、`single_block` 和 `narrow_gate`，每个场景包含直接起点到目标
边以及带固定 seed 的随机关节空间边。它记录每条边的判定、工作空间上界、采样数、
耗时，以及保守 checker 是否接受了 dense oracle 拒绝的边。

## 已保存结果

[`reports/mujoco_swept_audit_001`](../reports/mujoco_swept_audit_001/summary.json)
共 72 条边，false-safe 为 0，保守拒绝为 0。这是实现审计，不是连续碰撞安全、
硬实时或实体机器人证据。
