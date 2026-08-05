[English](RTC_OVERLAP_PRIMARY_V3_ACCEPTANCE.md) | 简体中文

# RTC overlap corrected-v3：证据验收

## 证据沿革

RTC 实现经过分阶段集成与评测。每个阶段保留独立的有效性状态，后续实验不会
追溯性地证明早期比较有效：

| 阶段 | 目的 | 当前状态 |
| --- | --- | --- |
| 固定观测 G0 | 采样器一致性、guidance 方向、延迟、显存 | 有效的实现证据；不是任务有效性 |
| 40-rollout hard-projection pilot | 独立的双方法机制探索 | 保留为探索性证据；不与 RTC v3 合并 |
| 60-rollout RTC v2 尝试 | 首次三方法闭环集成 | 因 query-0 环境状态残留而作废 |
| 两次各 50-triplet 的 v2 held-out 尝试 | 原计划的主比较 | 因相同配对问题而作废 |
| corrected-v3 smoke | 新 environment 与四哈希配对 gate | 已通过；不计入正式结果 |
| corrected-v3 held-out primary | 100 个匹配 triplet、300 个 rollout | 已完成，是当前正式结果 |

v2 因因果比较无效而被排除，并非因为数值与 v3 不同。三种方法复用同一个
LIBERO environment，导致任务 3、8、9 的两幅 policy image
发生变化。sampling key 和 sampling noise 仍然相同，但 query-0 的输入与
动作已经不同。

[配对审计记录](research/RTC_OVERLAP_PAIRING_AUDIT_20260805.md)保存了问题发现、
复现和修复过程。

corrected-v3 为每一个 rollout 单独创建并关闭 environment。在 finalization
之前，它要求同一个 triplet 的三种方法具有完全一致的 query-0：

- policy-input hash；
- response-action hash；
- sampling-key hash；
- sampling-noise hash。

任意一项不匹配都会使 artifact 验收失败。v3 是对修正协议的完整重跑，不会
合并或选择性修补 v2 数据。

## 已验证结果

| 方法 | 成功率 | motion seam 均值 | gripper seam 均值 |
| --- | ---: | ---: | ---: |
| Unconditioned overlap | 96/100 | 0.106729 | 0.053754 |
| Hard projection | 97/100 | 0.083089 | 0.055490 |
| RTC guidance | 97/100 | 0.087204 | 0.043190 |

两种 conditioned 方法的任务成功率差异都为 +1 个百分点，raw/Holm exact
McNemar p=1.0。实验不支持任务成功率优越性。

motion seam 差异只能作为探索性过程证据：

- hard projection：-0.023640，任务块 95% CI
  [-0.028187, -0.019207]；
- RTC guidance：-0.019524，任务块 95% CI
  [-0.023128, -0.016142]。

不能把上述 seam 结果写成碰撞安全、任务成功率或真实部署提升。

## 验收命令

在任意 Windows 目录运行：

~~~powershell
D:\arm-planning-control-project\project\scripts\rtc_primary_acceptance.cmd
~~~

如果只验证而不打开浏览器：

~~~powershell
D:\arm-planning-control-project\project\scripts\rtc_primary_acceptance.cmd -NoOpen
~~~

该命令采用失败关闭规则。只有满足以下全部条件才会生成有效结果：

- 两份 v3 原始 artifact 分别通过验证；
- combined analysis 能从原始 artifact 确定性重建；
- analysis manifest 有效；
- 100 个 triplet 全部绑定到重建后的分析；
- 引用的 10 个失败视频全部存在且哈希正确。

有效输出应包含：

~~~text
valid=true
rollouts=300
triplets=100
failure_videos_verified=10
tasks=10
~~~

生成的离线 dashboard 位于
[reports/pi05_rtc_overlap_primary_v3_300_001/index.html](../reports/pi05_rtc_overlap_primary_v3_300_001/index.html)。

## 审计说明

实施顺序为固定观测 sampler gate、闭环三方法比较和配对不变量审计。审计
发现 v2 复用 environment，使部分任务的 query-0 图像与动作不再匹配。
原 artifact 因此仅作为审计记录保留，其方法效果结论被排除。

修正后的 v3 为每个 rollout 创建独立 environment，使用四类哈希强制配对，
并在新的 held-out seeds 上完整重跑 300 个 rollout。结果没有证明成功率
提升，只提供探索性的 motion seam 降低证据。这属于实验设计修复，不是对
冲突数值的选择性处理。

## 适用范围

这是同一个 pi0.5 checkpoint 在 10 个固定 LIBERO-10 仿真任务上的证据。
它不能证明：

- 推理线程与控制线程真正独立运行；
- 操作系统 hard deadline；
- 连续碰撞安全；
- 跨 VLA policy 的泛化；
- 真实机器人有效性；
- 训练或微调了 pi0.5。
