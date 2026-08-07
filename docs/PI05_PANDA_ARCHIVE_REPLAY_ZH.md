# 冻结 pi0.5 响应在 Panda 保障链路上的离线回放

状态：当前离线集成证据

## 这一步解决什么问题

此前的笛卡尔适配器 smoke 只使用脚本动作，能够证明接口可以运行，但不能证明它能处理真实
VLA 输出。本流程使用 Physical Intelligence 官方 `pi0.5` LIBERO checkpoint 先前生成并冻结
保存的动作块，先核验来源和哈希，再把确定性分层样本送入本地 MuJoCo Panda 适配器和运行时
guard。

它属于跨控制器离线诊断，不是闭环部署：本次没有重新执行 checkpoint，没有让 Panda 新观测
反馈给策略，也没有计算任务成功率。

```text
冻结的官方 pi0.5 响应（10 x 7）
              |
              v
LIBERO 语义、裁剪、Panda 微分逆运动学
              |
              v
关节速度、加速度、deadline、碰撞 guard
              |
              v
逐案例 CSV、汇总 JSON、文件哈希清单
```

## 运行前如何验证证据

命令在执行任何 Panda 案例前，会完成以下核验：

1. 源清单中的完整文件集合、字节数、逐文件 SHA-256 和聚合哈希；
2. 官方 `pi05_libero` 配置、checkpoint URI、checkpoint 内容哈希以及干净的 OpenPI 服务证明；
3. LIBERO 动作空间和动作适配源文件哈希；
4. NPZ 的全部字段、dtype 和 shape；
5. bootstrap 到后续查询的引用链与 overlap 调度方程；
6. 全部 7,934 个响应动作哈希，而不只是抽中的 90 个。

流程只读访问 `evidence/`。派生报告写入新的目录，不修改原始冻结实验。

## CPU 复现命令

在 Windows 仓库根目录执行：

```powershell
& '.\.venv\Scripts\python.exe' -m armbench vla-panda-archive-replay `
  evidence\pi05_rtc_overlap_primary_v3_seed_20260807_001\evaluation `
  --output-directory results\pi05_panda_archive_replay `
  --chunks 90 --selection-seed 20260807

& '.\.venv\Scripts\python.exe' -m armbench `
  vla-panda-archive-replay-validate `
  results\pi05_panda_archive_replay `
  --source-directory `
  evidence\pi05_rtc_overlap_primary_v3_seed_20260807_001\evaluation
```

Linux 使用 `./.venv/bin/python` 和正斜杠路径。该流程需要固定版本的 MuJoCo Menagerie Panda
模型，但不需要 GPU、OpenPI 服务或下载 checkpoint。输出目录必须尚不存在。

90 个动作块在 30 个 `(task_id, method)` 分层中等额抽取：10 个 LIBERO-10 任务、3 种运行时
方法、每层 3 个动作块。每个动作块在 3 个 Panda 场景中都从场景起点独立开始，并新建 guard，
因此 deadline latch 和前一案例的速度状态不会串扰。

## 已保存的 90 动作块结果

仓库中的派生报告位于
[`reports/pi05_panda_archive_replay_90_001`](../reports/pi05_panda_archive_replay_90_001/summary.md)，
共包含 90 个动作块和 `free_space`、`single_block`、`narrow_gate` 下的 270 个独立案例。

| 观测项 | 结果 |
| --- | ---: |
| 已重新计算的源响应哈希 | 7,934 / 7,934 |
| 含输入裁剪的动作块 | 89 / 90 |
| 原始 Panda 前瞻路径无效 | 36 / 270 |
| guard 发生干预 | 226 / 270 |
| 在 200 ms deadline 下超时 | 3 / 270（同一动作块在 3 个场景） |
| 通过 0.02 rad 边采样检查的最终路径 | 270 / 270 |
| 同时满足 guard 全部约束 | 264 / 270 |
| 避碰与加速度约束冲突 | 6 / 270 |

最后 6 个案例是必须保留的负结果：guard 选出的停止或保持路径通过了离散碰撞检查，但立即改变
速度会超过配置的关节加速度上限。因此报告没有把它们称为安全案例。这个现象暴露了“可行域为空”
的问题，可作为下一阶段 deadline 有界轨迹修复方法的明确研究对象。

报告同时保存本机 OS、Python、NumPy、MuJoCo、Panda 场景和实现源码指纹。适配器与 guard 的
耗时是本机测量值，不是硬实时最坏情况保证。

## 输出文件

- `provenance.json`：源 checkpoint 证明、主张标志、抽样、运行环境与实现哈希；
- `per_chunk.csv`：每个动作块与 Panda 场景组合一行；
- `summary.json`：可由 CSV 重新计算的汇总；
- `summary.md`：供人工快速验收的摘要；
- `manifest.json`：文件集合、字节数与 SHA-256。

## 对外表述边界

- `source_policy_checkpoint_attested=true` 只表示冻结响应可以追溯到已证明的官方 checkpoint；
- `policy_checkpoint_executed_in_replay=false` 表示此次 CPU 回放没有运行 `pi0.5` 推理；
- `task_success_evaluated=false` 表示结果不是 LIBERO 或 Panda 任务成功率；
- `panda_closed_loop_executed=false` 表示 Panda 没有为下一次策略调用产生反馈观测；
- 微分逆运动学不等价于 LIBERO 的力矩级 OSC，离散边采样也不是连续碰撞或物理安全认证。
