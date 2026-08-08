# 官方 LeRobotDataset round-trip

本阶段固定官方 `lerobot==0.4.4` 与 v3.0 `LeRobotDataset` API，在隔离的
NumPy 2 环境中导出并重新加载一个 3 帧 Panda episode。验证器逐帧比较两路图像、
float32 状态、动作、task 字符串、时间戳、帧索引和 episode 索引。

LeRobot 的 `rerun-sdk` 依赖 NumPy 2，而仓库的 OpenPI 路径固定 NumPy 1.26.4，
因此不能把两个运行时混在同一个环境。请使用独立环境：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  scripts\setup_official_lerobot.ps1

& '.\.venv-lerobot-0.4.4\Scripts\python.exe' -m armbench `
  vla-lerobot-official-smoke `
  --output-directory reports\official_lerobot_roundtrip_001

& '.\.venv-lerobot-0.4.4\Scripts\python.exe' -m armbench `
  vla-lerobot-official-validate `
  reports\official_lerobot_roundtrip_001
```

安装脚本会写入 `.pth`，让隔离环境读取当前 checkout 的 `src`，并以
`pip check` 验证依赖。ABI 敏感的 NumPy、PyTorch、TorchVision、Rerun SDK、
Datasets、PyArrow 和 PyAV 版本记录在
`scripts/official_lerobot_windows_py310_constraints.txt`。

数据集声明 `robot_type=panda_armbench_runtime`。状态是 7 个 Panda 关节位置加
归一化夹爪位置，动作是 7 个 Panda 关节速度加归一化夹爪位置；它们不是 SO-101
关节位置动作。artifact 还绑定官方 loader 版本、动作语义 SHA-256 以及当前
ArmBench 适配器/watchdog 实现哈希。

这项结果只证明官方数据集序列化与加载边界。没有连接 Panda 或 SO-101，没有运行
学习策略 checkpoint，没有验证驱动器、急停或硬实时控制。
