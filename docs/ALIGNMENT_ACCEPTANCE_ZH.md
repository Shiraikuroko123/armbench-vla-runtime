# 确定性时序对齐：证据验收

## 用途

该流程验证已经保存的 pi0.5-LIBERO 确定性时序对齐实验，并生成 baseline
与 aligned 条件的配对视频 dashboard。它不会重新运行 checkpoint 推理或
300 个 rollout，因此不需要 GPU。

## 运行命令

在 Windows 仓库根目录运行：

~~~powershell
.\scripts\alignment_acceptance.cmd
~~~

从任意目录可以使用绝对路径：

~~~powershell
D:\arm-planning-control-project\project\scripts\alignment_acceptance.cmd
~~~

只验证而不打开浏览器：

~~~powershell
.\scripts\alignment_acceptance.cmd -NoOpen
~~~

Linux、macOS 或已经配置好的 Python 环境可以运行：

~~~bash
python -m integrations.openpi.alignment_acceptance --no-open
~~~

## 验收契约

命令按以下顺序执行：

~~~text
root manifest
  -> nested evaluation validation
  -> checkpoint and source attestation
  -> matrix, pairing, query, and statistic recomputation
  -> analysis-to-source hash verification
  -> 300 required video checks
  -> offline dashboard generation
~~~

任何步骤失败都会返回非零退出码，结果不得标记为已验证。

## 关键输出

有效报告应包含：

| 字段 | 期望值 |
| --- | --- |
| valid | true |
| root_and_nested_validation | valid / complete |
| protected_files_checked | 331 |
| rollouts | 300 |
| matched_pairs | 150 |
| videos_verified | 300 |

pi05_identity 还应记录：

- policy_config 为 pi05_libero；
- checkpoint URI 为
  gs://openpi-assets/checkpoints/pi05_libero；
- checkpoint content SHA-256 为
  9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5；
- OpenPI commit 为
  15a9616a00943ada6c20a0f158e3adb39df2ccac；
- policy_loaded 为 true。

这些字段将实验绑定到记录的本地 checkpoint 内容与 source checkout，
不构成对上游发布者身份的密码学认证。

## 可视化检查

1. 在 dashboard 中选择 200 ms 主条件。
2. 选择一个 baseline 失败、aligned 成功的注册配对。
3. 同时播放两段视频。
4. 核对 pair ID、task、initial state、success label、policy query 数量和
   视频路径。
5. 在 per_pair.csv 与 per_episode.csv 中验证同一记录。

视频用于检查执行过程，统计结论仍以通过 manifest 绑定的 CSV/JSON 为准。

## 当前结果

200 ms 主条件包含 50 个匹配 pair：

- asynchronous baseline：18/50；
- latency-aligned：50/50；
- 配对差异：+64 个百分点；
- pair bootstrap 95% CI：[+50,+76]；
- Holm 校正 exact McNemar p=1.40e-9。

该结果支持确定性注入 LIBERO 延迟下的训练无关后缀对齐。它不能证明真实
网络抖动、独立运行的控制/推理线程、硬实时保证、碰撞安全或真机有效性。

## 失败处理

正式 evidence 目录应视为只读。验证失败时：

1. 保存完整终端输出；
2. 确认使用了正确的仓库与虚拟环境；
3. 根据首个失败边界检查 manifest、analysis source 或视频；
4. 使用新 run ID 复现实验；
5. 不直接修改原始 CSV、JSON、manifest 或视频。
