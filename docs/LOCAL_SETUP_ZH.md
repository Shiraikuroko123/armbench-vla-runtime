# 本地 CPU 安装与支持

状态：运维文档

本地路径支持 MuJoCo Panda 基准、scripted VLA 运行时检查、artifact 验证和
离线 dashboard。它不运行 `pi0.5` checkpoint，也不需要 GPU。

## 环境要求

- 64 位 CPython 3.10
- Git
- Windows 10/11 与 PowerShell 5.1+，或仍受维护的 Linux 发行版
- 使用 MuJoCo 交互式 viewer 时需要桌面会话

除仓库已有证据外，Python 环境、模型和输出大约还需要 1-2 GB 空间。

## 自动安装

在仓库根目录运行。

Windows：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_local.ps1
```

Linux：

```bash
./scripts/setup_local.sh
```

脚本会创建 `.venv`、安装 `.[test]`，并将 MuJoCo Menagerie 的固定提交
`71f066ad0be9cd271f7ed58c030243ef157af9f4` 稀疏检出到
`.cache/mujoco_menagerie`。只有需要 OpenPI 客户端适配器时才增加
`-WithVla` 或 `--with-vla`。真实在线推理仍然另行需要 checkpoint 和推理
服务器。

## 手动配置模型路径

自动安装之外，也可以显式指定 `scene.xml` 或 Menagerie 仓库根目录：

```powershell
$env:ARMBENCH_PANDA_SCENE = 'C:\models\mujoco_menagerie\franka_emika_panda\scene.xml'
$env:ARMBENCH_MENAGERIE_ROOT = 'C:\models\mujoco_menagerie'
```

没有设置变量时，ArmBench 先检查仓库内的 `.cache`，再兼容原有工作区
同级的 `upstream/mujoco_menagerie`。以下命令会报告实际解析到的路径：

```powershell
& '.\.venv\Scripts\python.exe' -m armbench doctor --json
```

## 本地验收

```powershell
& '.\.venv\Scripts\python.exe' -m armbench doctor
& '.\.venv\Scripts\python.exe' -m armbench mujoco-validate
& '.\.venv\Scripts\python.exe' -m armbench mujoco-self-collision-validate `
  reports\mujoco_self_collision_audit_001
.\scripts\vla_demo.cmd -CheckOnly
```

桌面环境中可以打开交互式 viewer：

```powershell
& '.\.venv\Scripts\python.exe' -m armbench mujoco-view `
  --scenario narrow_gate --clearance-mm 20 --payload 0.5
```

已保存的 VLA 证据不需要重新推理：

```powershell
.\scripts\measured_age_confirmatory_acceptance.cmd
.\scripts\rtc_primary_acceptance.cmd -NoOpen
```

| 能力 | GPU | 完成安装后需要联网 | OpenPI 服务器 |
| --- | --- | --- | --- |
| Panda 规划、控制和 viewer | 不需要 | 不需要 | 不需要 |
| Scripted 运行时和故障检查 | 不需要 | 不需要 | 不需要 |
| Artifact 验证和 dashboard | 不需要 | 不需要 | 不需要 |
| `pi0.5` 在线推理 | 需要 | 通常需要 | 需要 |

`doctor` 输出 `BLOCKED` 时，应先处理标为 `FAIL` 的必需项。缺少
`openpi_client` 或 FFmpeg 默认只是提示，只有在线 VLA 传输或视频编码命令
才会要求这些组件。
