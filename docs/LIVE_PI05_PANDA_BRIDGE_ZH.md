# pi0.5-LIBERO 到 Panda 的在线桥接验收

这条路径把远程 OpenPI 的 `pi05_libero` checkpoint 接到本地 MuJoCo Panda
运行时。它解决的是动作接口和运行时边界问题，不把 LIBERO 的 Hx7
笛卡尔动作直接当成 Panda 的 Hx8 关节动作：

```text
OpenPI WebSocket
  -> server attestation + pi05_libero 原生 Hx7
  -> LIBERO 语义校验
  -> PandaCartesianActionAdapter（差分 IK、关节/速度约束）
  -> latest-only policy worker
  -> deadline-aware guard / braking repair
  -> MuJoCo Panda 力矩控制与 trace
```

## 运行前提

- Ubuntu 主机上的 OpenPI server 必须返回 ArmBench attestation，且
  `policy_config=pi05_libero`、action horizon 为 10；
- host-side ArmBench venv 已安装 `.[test,vla]`；
- `MUJOCO_GL=egl` 和 `PYOPENGL_PLATFORM=egl` 用于无显示器的云主机；
- 这不是实体机器人实验，也不是操作系统硬实时保证。

## 命令

在 `/workspace/armbench/project` 执行：

```bash
export PYTHONPATH=/workspace/armbench/project/src
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

/workspace/armbench-venv/bin/python scripts/run_live_pi05_panda_smoke.py \
  --host 127.0.0.1 \
  --port 8000 \
  --scenario free_space \
  --mode braking_invariant \
  --steps 12 \
  --extra-steps 6 \
  --output-directory /workspace/armbench-results/g01_live_panda_smoke_001 \
  --video
```

输出目录包含：

| 文件 | 验收内容 |
| --- | --- |
| `summary.json` | server metadata、checkpoint identity、Panda 指标、限制条件 |
| `events.json` | 每次响应、动作切换、hold 和 deadline 事件 |
| `trace.npz` | wall-clock、MuJoCo 时间、关节状态和动作轨迹 |
| `panda_trace.mp4` | MuJoCo 外部相机回放（可选） |
| `manifest.json` | 输出文件 SHA-256 清单 |

## 如何判定

先看 `summary.json`：

1. `policy_provenance.identity.response_origin` 必须为
   `live_checkpoint_inference`；
2. `episode.scripted_policy` 必须为 `false`，
   `episode.policy_checkpoint_executed` 必须为 `true`；
3. `episode.policy_source` 应包含 `openpi_pi05_libero_live` 和
   `response_sha256`；
4. `episode.physical_safe`、`deadline_rejections`、P95 时延和 hold 率要与
   `events.json`、`trace.npz` 一致；
5. `manifest.json` 校验通过后，才把该目录复制到本地或 GitHub Release。

一次 smoke 只能证明“真实 attested checkpoint 已经进入 Panda 仿真运行时”。
它不能单独证明 LIBERO 任务成功、真实机器人安全或跨模型泛化；这些需要
独立时钟 pilot、多任务多 seed 和明确的基线矩阵。
