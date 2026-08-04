# ArmBench 五分钟面试验收

这个入口用于展示已经完成并保存在仓库中的正式 pi0.5-LIBERO 实验。它不会重新训练模型，也不会重新运行 300 次 rollout；因此不需要 GPU、云服务器或额外费用。

## 一条命令

在 Windows 中，从任意目录运行：

```powershell
D:\arm-planning-control-project\project\scripts\interview_acceptance.cmd
```

如果仓库移动过位置，使用仓库内的相对路径：

```powershell
.\scripts\interview_acceptance.cmd
```

入口会自行定位仓库和 `..\.venv\Scripts\python.exe`，重新验证正式证据，生成 dashboard，并在默认浏览器中打开。只验证而不打开浏览器：

```powershell
.\scripts\interview_acceptance.cmd -NoOpen
```

Linux/macOS 或已经进入仓库时可直接运行：

```bash
python -m integrations.openpi.interview_acceptance
```

## 五分钟顺序

第一分钟，读终端 JSON 中的 `problem` 和 `method`。项目解决的是异步 VLA 推理造成的 action chunk 前缀过期；方法不训练 pi0.5，而是跳过与延迟对应的过期动作，执行后续五步动作。

第二分钟，检查 `valid=true`、`root_and_nested_validation="valid / complete"`、`protected_files_checked=331`、`rollouts=300`、`matched_pairs=150` 和 `videos_verified=300`。任意受保护文件被修改后，入口应当失败关闭，不生成可展示的有效结论。

第三分钟，检查 `pi05_identity`：

- `policy_config=pi05_libero`
- checkpoint URI 为官方 `gs://openpi-assets/checkpoints/pi05_libero`
- checkpoint content SHA-256 为 `9cd1b00d...d4487e1d5`
- OpenPI commit 为 `15a9616...df2ccac`
- `policy_loaded=true`

这些字段来自 GPU server 的 checkpoint attestation，并且由 root manifest、evaluation manifest和独立 validator交叉验证。它们证明本次保存的闭环实验加载了所声明的 checkpoint 内容，不等于对上游发布者身份做密码学认证。

第四分钟，在 dashboard 保持 `200 ms` 条件，选择默认的 baseline-failure / aligned-success pair，点击同时播放。左侧是 `async_unguarded` 从动作索引 0 开始执行并失败；右侧是 `latency_aligned` 跳过四个过期动作后成功。页面中的视频路径、成功标签和查询次数都绑定到通过 manifest 验证的 CSV 与 MP4。

正式视频为 H.264、`224x224`、10 fps，这是官方 pi0.5-LIBERO 视觉协议的输入尺寸，适合动作和结果验收，但不应描述成高清演示素材。仓库中的代表性 baseline 与 method 视频分别包含 221 和 85 个可解码且互不相同的帧，因此不是静止截图伪装的视频。

第五分钟，解释结果和边界：200 ms 主条件下成功数从 `18/50` 提升到 `50/50`，配对差异为 `+64` 个百分点，bootstrap 95% 区间 `[+50,+76]`。这是确定性注入延迟下的 LIBERO 仿真证据，不是真机、模型训练、硬实时保证、碰撞安全认证或 measured network-jitter 结论。

## 面试官实际验证了什么

入口不是只打开一个预先写好的网页。它会先执行现有严格验收链：

```text
root manifest
  -> nested evaluation validator
  -> checkpoint/source attestation
  -> matrix、pair、query、统计量重算
  -> analysis manifest 和原始 CSV hash 绑定
  -> 300 个必需视频存在性与 manifest hash
  -> 生成并打开离线 dashboard
```

终端返回非零退出码或 `valid=false` 时，不应展示或引用结果。正式证据目录是只读材料；调试时创建新的 run ID，不要编辑其中的 CSV、JSON、manifest 或视频。

## 现场追问

“这是你训练的 pi0.5 吗？”

不是。项目部署并评测冻结的官方 pi0.5-LIBERO checkpoint，原创部分是运行时的时间对齐机制、实验协议、失败关闭、证据保存和独立验证。

“视频能证明模型是真的吗？”

视频只能证明保存的视觉执行结果。模型身份来自 server attestation、checkpoint 文件清单及内容 SHA-256、固定 OpenPI commit、运行日志和两层 manifest；这些证据必须一起成立。

“为什么不用 RL？”

当前问题是可直接定义的运行时时间同步错误，确定性后缀对齐是更直接且可解释的干预。它适合 VLA 部署、机器人系统和评测岗位；申请 VLA 训练或 RL 算法岗时，仍需要单独的学习式项目或扩展。
