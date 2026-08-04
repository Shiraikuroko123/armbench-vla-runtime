# pi0.5-LIBERO 异步延迟与状态拒绝实验 Runbook

## 0. 当前边界

本文件定义一次尚未执行的真实学习策略实验。固定研究对象为官方 OpenPI 仓库提交
`15a9616a00943ada6c20a0f158e3adb39df2ccac`、配置 `pi05_libero`、检查点
`gs://openpi-assets/checkpoints/pi05_libero` 和官方 LIBERO 协议。

截至 2026-08-04，仓库中没有 attested pi0.5 checkpoint GPU rollout；实际 checkpoint
实验结果状态为 `false`。下面所有预算、矩阵和简历数字位置都是预注册计划，不是已得结果。

在保存真实 GPU 服务 attestation、完整 rollout 产物、finalize/validate 结果和统计结果以前，
项目只能称为：

> OpenPI-compatible pi0.5-LIBERO runtime evaluator（兼容 OpenPI 的 pi0.5-LIBERO
> 运行时评测器）。

此时不能称为“部署了 pi0.5”“完成了 pi0.5-LIBERO 实验”或“提升了 VLA 成功率”，也不能
在简历、README 或面试材料中填写任何尚未实测的成功率、时延、拒绝率和查询开销数字。
仓库现有的 scripted policy、loopback 和 DROID 请求回放只能证明协议、故障处理与可复现
基础设施，不是学习策略效果证据。

## 1. 研究问题与预注册主张

研究问题：在官方 pi0.5-LIBERO 闭环中，异步推理期间继续执行上一动作所产生的状态漂移，
会怎样影响任务成功率和策略查询成本；一个不训练、不微调模型的状态失配拒绝器能否改变这
种影响？

运行时并不把测得的 GPU 推理耗时直接换算为仿真步数。每次查询先保存查询时观测，然后
进行阻塞式 WebSocket 推理，再让 LIBERO 环境用上一条已执行动作前进固定的
`latency_steps`，最后消费对应旧观测的动作块。这样把可重复的“注入延迟步数”和机器相关
的“实测推理墙钟时间”分开。

预注册主分析为：

- 数据范围：`libero_spatial` 全部 10 个任务，每个任务初始状态索引 `0:5`。
- 主比较：`replan_steps=5`、`latency_steps=4` 下，`state_guard` 相对
  `async_unguarded` 的配对任务成功率差值。
- 干预控制：在相同主条件下增加 `fixed_refresh`，区分 state-aware 拒绝效果与单纯增加
  hold/requery、策略查询和新观测的效果。
- 机制证据：实际执行的陈旧动作步数、接受的陈旧动作块数、状态拒绝数，以及末端位置、
  姿态和夹爪失配。
- 代价证据：策略查询数相对开销、每次推理 P50/P95、episode 步数与墙钟时间。
- 延迟曲线：固定 `replan_steps=5`，比较 `latency_steps=0,2,4`。
- 动作跨度消融：固定 `latency_steps=2`，比较 `replan_steps=1,5,10`。

要检验而不是预设为真的假设：

1. `async_unguarded` 的任务成功率会随注入延迟增加而下降。
2. `state_guard` 会减少超过阈值的陈旧动作块执行，但可能因拒绝和重新查询降低任务成功率。
3. guard 的收益或损失取决于动作执行跨度，并伴随可量化的查询开销。
4. 若 state-aware 选择本身有作用，在干预/查询预算接近时，`state_guard` 应与状态无关的
   `fixed_refresh` 呈现可复现差异；若两者相近，则效果更可能来自通用刷新而非失配阈值。

无论方向是否符合假设，都保留并报告结果。若置信区间跨零、guard 伤害成功率或没有触发
拒绝，这些都是有效负结果，不能换任务、改阈值后只展示更好的一组。

## 2. 固定的官方协议

| 项目 | 固定值 |
| --- | --- |
| OpenPI commit | `15a9616a00943ada6c20a0f158e3adb39df2ccac` |
| policy config | `pi05_libero` |
| checkpoint | `gs://openpi-assets/checkpoints/pi05_libero` |
| benchmark | `libero_spatial`，主矩阵覆盖 10 个任务 |
| 官方随机种子 | `7` |
| 官方默认 replan | `5` 步 |
| policy action horizon | `10` 步；评测器拒绝 `replan_steps > 10` |
| 控制频率 | `20 Hz`，每个注入 delay step 对应 `50 ms` |
| 输入图像 | base/wrist 两路；各旋转 180 度，再 pad-resize 为 `224x224 uint8` |
| 状态 | 末端位置 3 + quaternion 转 axis-angle 3 + gripper 2，共 8 维 |
| 动作 | LIBERO 7 维末端动作；不能复用 DROID 的 8 维关节速度 guard |
| 成功定义 | LIBERO 环境返回的 `done` |
| 稳定等待 | episode 开始先执行 10 步官方 dummy action |
| 最大任务步数 | Spatial 220、Object 280、Goal 300、LIBERO-10 520、LIBERO-90 400 |

OpenPI 上游在该提交的 `examples/libero/README.md` 中报告该检查点在 Spatial、Object、
Goal 和 LIBERO-10 上分别为 98.8、98.2、98.0 和 92.4，平均 96.85。这些是上游参考值，
不是本项目的结果，也不能复制到简历的个人实验栏。

LIBERO 任务需要合法的抓取接触，因此不把原始 contact count 称为“安全违规”。本实验的
`state_guard` 是运行时启发式拒绝器，不是形式化证明、硬实时系统或安全认证。

## 3. 基线、变量与公平性

### 3.1 三个运行模式

`async_unguarded` 是主基线。查询期间环境继续执行上一动作，返回的旧观测动作块不做状态
检查并直接执行。

`state_guard` 只增加一个机制：分别比较查询时和响应时的末端位置误差（米）、quaternion
角距离（弧度）与夹爪 L-infinity 误差。任一超过阈值就拒绝动作块，执行 Cartesian hold
并重新查询。默认阈值固定为：

- position：`0.01 m`
- orientation：`0.10 rad`
- gripper：`0.05`
- 连续重新查询上限：`2`

`fixed_refresh` 是干预控制组。它不根据位置、姿态或夹爪失配决定是否拒绝，而是每收到
`N` 个候选动作块就强制拒绝一次，记录 `rejected_fixed_refresh/scheduled_refresh`，然后与
`state_guard` 共用完全相同的 Cartesian hold、bounded requery 和 `max_requeries` 路径。
紧接强制拒绝后的重新查询允许通过，避免对同一刷新点无限拒绝；状态失配仍被记录，但不
触发该模式的拒绝。

这个控制组回答的是：“guard 的变化来自状态相关选择，还是任何额外刷新/查询都会发生？”
它不能自动实现逐 episode 相同的干预数或 query 数，因此必须报告残余预算差；更不能重置
或共享服务端 JAX RNG。

主矩阵开始后不得根据测试结果调这些阈值。阈值敏感性必须另建输出目录，并标为消融，
不能回填替换主结果。

### 3.2 自变量和控制量

| 类型 | 变量 |
| --- | --- |
| 主自变量 | mode：`async_unguarded,state_guard,fixed_refresh`；核心延迟矩阵先跑前两者 |
| 延迟 | `latency_steps=0,2,4` |
| 动作跨度 | `replan_steps=1,5,10` |
| 固定量 | task、初始状态、prompt、checkpoint、commit、seed、图像预处理、步数上限 |
| 主结果 | 官方任务成功率 |
| 机制结果 | `stale_action_steps`、stale chunks、rejections/interventions、三类状态失配 |
| 系统结果 | query count、推理 P50/P95、环境步数、墙钟时间、故障类型 |

每个 `pair_id` 的两个或三个模式使用同一 task、同一官方 initial state、同一 horizon 和
同一 delay，并校验 `initial_state_sha256` 相等。各模式相邻运行，顺序按 group 交替，减少服务器热身和
时间漂移偏差。官方 `client.reset()` 是空操作，pi0.5 服务端的 JAX RNG 会随每次 query
推进，因此两个模式没有共享或重置的 policy noise。“输入状态严格配对”不等于“策略随机
数严格配对”；这一限制必须留在 `resolved_protocol.json` 和报告中。投稿级因果主张还需要
多次独立 server 启动/seed 的确认性复跑，或实现 keyed-noise 控制。

`seed=7` 是官方环境种子，`episode_indices=0:5` 表示每个任务的 5 个固定初始状态，不能写成
“5 个随机种子”。统计单位是 episode/pair，不是动作步或 policy query。

这是一项 matched-initial-state sequential study：同一 initial state 上的两个 mode 顺序相邻
且交替，但实际 policy query 按顺序推进同一个服务端随机流。因此“matched”“paired”仅指
任务、环境初态和已列出的协议字段匹配，不能表述为相同随机噪声下的严格成对试验。

### 3.3 `N` 的校准与冻结

`fixed_refresh_interval=N` 没有合法的隐式默认值。选择 `fixed_refresh` 却不显式传入正整数
N 时 evaluator 必须 fail closed。N 只能在与确认性矩阵不重叠的 pilot initial states 上校准；
本 runbook 预留 `45:47` 给 pilot，正式主矩阵使用 `0:5`。

校准只看 `state_guard` 的 interventions、rejections 和 policy queries，用来选择能让固定刷新
预算尽量接近 guard 的 N；不能看 task success 后挑最有利的 N。候选 N 集合、距离准则和
tie-break 必须在读取 pilot 成功率前写入 calibration note。选定后把 N、pilot run ID 和规则
记录进证据，确认性三模式 run 全程冻结；任何改 N 的实验都使用新 ID 并标为探索性分析。

若 pilot 中 guard 从不拒绝，就没有可校准的干预预算。此时不能虚构 N 来支持机制主张，
应报告“控制不可识别”，扩大不重叠 pilot 或停止该对照。

### 3.4 服务端 attestation

正式 run 默认拒绝普通、未证明身份的 OpenPI server。服务端必须通过
`serve_policy_attested.py` 在加载 policy 后生成 `armbench.openpi_server_attestation.v1`，
并把公开摘要放入 WebSocket metadata。evaluator 在第一个 episode 前核对：

- `policy_loaded=true`、`policy_config=pi05_libero` 和声明的 checkpoint URI；
- 固定 OpenPI commit；`git status --porcelain=v1 --untracked-files=all` 为空，即没有任何
  已跟踪或未跟踪变更；所有递归 submodule 均已初始化且位于父仓库记录的 commit；
- action horizon 为 10；
- checkpoint 文件数、总字节数与全目录内容 SHA-256；
- attested server 入口源码 SHA-256。

缺少 attestation、字段不匹配或 digest 格式错误时必须在 rollout 前 fail closed。
`--allow-unattested-server` 只允许诊断连通性；任何带该参数的 run 都不能进入 L1/L2、性能表
或简历结果。attestation 绑定的是本次加载的本地 cache 内容、launcher 与 commit，并不等同
于上游发布者的数字签名，也不证明模型质量、统计结论或物理安全。

## 4. ¥50 smoke test 与停损规则

前 50 元只解决一个问题：官方 checkpoint 是否能在租用的 Linux/NVIDIA 环境中按本协议
完成真实闭环。不要在本地 Windows 无 NVIDIA 机器上安装整套 CUDA，也不要先跑大矩阵。

smoke 固定为 task 0、initial state 0 和 1、官方 nominal 条件
`async_unguarded / replan=5 / latency=0`，共 2 个 rollout，并保存全部视频。

继续投入必须同时满足：

1. `nvidia-smi` 正常，显存足以加载检查点，OpenPI checkout 的 HEAD 与固定 commit 完全一致。
2. server attestation 与启动日志明确记录 `pi05_libero`、目标 checkpoint、内容 digest 和固定
   commit；attestation 与日志进入同一个 run 证据目录。
3. 两个 rollout 都正常完成，无 transport、shape、nonfinite、渲染或依赖故障。
4. 至少 1/2 episode 得到 LIBERO `done=true`；0/2 与上游 nominal 水平严重不符，应先停机排查。
5. `evaluation/integrity.json` 为 valid，根目录 finalize 与 validate 均成功；两层 manifest、
   环境、协议、CSV、summary 和视频均可打开。
6. 根据账单单价和实测墙钟时间估算 40-run calibration pilot、300-run 核心矩阵、150-run
   干预控制和 200-run horizon 消融成本。

出现以下任一情况立即停机并下载已有日志：花费达到 ¥50 仍未加载模型；commit/checkpoint
身份不清；连续 3 次基础设施故障；0/2 成功；视频全黑或相机/动作维度不符；按 pilot
吞吐量推算总费用超过个人上限。失败 smoke 也要保留，不能删除后声称“首次即跑通”。

任务成功失败与基础设施失败必须分开。单个正常结束但任务失败的 episode 是模型结果；
连接中断、OOM、无效动作或渲染错误是基础设施故障，不能拿来解释模型优劣。

## 5. ¥300-800 扩展矩阵

预算不是承诺值。云平台计费、checkpoint 下载和推理速度差异很大，实际 go/no-go 以账单和
pilot 的 `wall_time_s`、`policy_queries` 为准。

### 阶段 A：40-run 不重叠校准 pilot

- task IDs：全部 10 个 Spatial 任务，避免从两个任务估计全 suite 的刷新频率
- initial states：`45:47`，不与正式矩阵的 `0:5` 重叠
- modes：2
- replan：`5`
- delay：只用预注册主条件 `4`
- 总计：`10 x 2 x 2 x 1 x 1 = 40` rollouts，20 个 pair

目的不是形成结论，而是测量 guard 拒绝率、horizon 5 查询成本、每个条件的运行时间，并按
第 3.3 节的预注册规则选择 N。pilot 的任务成功率不得用于选 N 或写 headline。若基础设施
故障不为零、guard 没有可校准的拒绝，或按保守上界估算核心矩阵将超过剩余预算，不进入
阶段 B。

#### 阶段 A2：pilot 后的时间对齐探索

`pi05_libero_pilot_002` 完成后，固定状态阈值 guard 的 282 次拒绝全部包含位置失配，且
11/20 个 guard 回合终止于连续重查询上限。由于延迟期间环境按协议继续执行上一动作，
query-time 与 response-time 的位移混合了预期受控运动和未建模漂移。该结果保留为负结果，
不能通过事后放宽阈值改写。

新增的 `latency_aligned` 是 pilot 后提出的独立探索模式，不是原预注册 guard 的替代结果。
若实际执行了 `d` 个延迟步，它跳过返回动作块的前 `d` 个动作，再执行长度为
`replan_steps` 的后缀。运行前必须满足
`latency_steps + replan_steps <= 10`；不足时 fail closed，不得截断 horizon 或静默回退。

第一次云端探索固定使用未参与 pilot 与确认性矩阵的 initial states `47:49`：

- task IDs：全部 10 个 Spatial 任务；
- modes：`async_unguarded,latency_aligned`；
- `replan_steps=5`、`latency_steps=4`；
- 40 rollouts、20 个 matched condition groups；
- 全部任务成功率、失败、视频和非效能故障均保留；
- 结果只用于 go/no-go 和方法诊断，不使用确认性 p 值措辞。

只有该探索无基础设施故障、动作长度契约全部通过且没有新的系统性终止类型，才能在新的
冻结说明中把 `latency_aligned` 加入阶段 B。无论成功率是否提高，探索结果都必须归档。

### 阶段 B：300-run 核心矩阵

- Spatial 全部 10 个任务
- 每任务 initial states `0:5`
- 两个 mode
- `replan_steps=5`
- `latency_steps=0,2,4`，在 20 Hz 控制周期下对应注入 `0/100/200 ms`
- 总计 300 rollouts、150 个 pair；每个 suite-level condition 有 50 个 episode

它支持预注册主比较和延迟曲线。每任务每条件只有 5 个 episode，因此 task-level 数字只作
描述，不作为稳定的逐任务显著性结论。

### 阶段 C：150-run 三模式干预控制

在 N 已由阶段 A 冻结后，针对预注册主条件运行：

- Spatial 全部 10 个任务与 initial states `0:5`
- `async_unguarded,state_guard,fixed_refresh` 三个 mode
- `replan_steps=5`、`latency_steps=4`
- 总计 150 rollouts、50 个 matched condition groups

它用于生成 `intervention_control_comparisons`，优先级高于 horizon 扩展。阶段 B+C 合计
450 rollouts；只有该控制显示干预/query 预算足够接近，才能把 guard 与 fixed refresh 的差异
解释为“与 state-aware 选择一致的证据”。

### 阶段 D：200-run horizon 消融

仅在阶段 B 完整、无基础设施故障且预算仍足够时执行：

- 同样的 10 个任务与 initial states `0:5`
- 两个 mode
- 只补 `replan_steps=1,10`
- 固定 `latency_steps=2`
- 新增 200 rollouts；与阶段 B 已有的 `replan_steps=5, latency_steps=2` 合并解释

阶段 B+C+D 共 650 个 rollout。预算接近 ¥300 时优先保证阶段 B 完整；剩余预算优先阶段 C，
再做阶段 D。不要为了凑图留下半个核心矩阵；接近 ¥800 仍不能完成时，保留全部负结果和
进度文件，按实际完成度报告。

成本预测使用 pilot 实测值，而不是宣传价。至少计算：

```text
保守 GPU 小时 = 计划 rollout 数 x pilot 每 rollout P95 墙钟秒 / 3600
预计费用 = 保守 GPU 小时 x 云平台实际单价 + 存储/流量
```

`replan_steps=1` 会显著增加查询数，阶段 D 前还应单独完成至少一个 h=1 小跑，再更新预测。
为重跑和 checkpoint 下载至少预留总预算的 20%，不要让抢占中断产生无法配对的半矩阵。

## 6. GPU 执行命令模板

以下命令在 Linux/NVIDIA 云主机执行。路径是占位符，必须替换为绝对路径；结果目录必须在
可持久化磁盘。不要手工拼接 `docker compose up/logs/stop`：入口
`libero_compose_run.py` 会把 preflight、官方 compose + ArmBench overlay、attested server、
evaluator、停止记录和两层 manifest 作为一次事务执行。

```bash
export OPENPI_ROOT=/workspace/openpi
export ARMBENCH_ROOT=/workspace/armbench/project
export ARMBENCH_RESULTS_ROOT=/workspace/armbench-results
export OPENPI_DATA_HOME=/workspace/openpi-cache

mkdir -p "$ARMBENCH_RESULTS_ROOT" "$OPENPI_DATA_HOME"
cd "$OPENPI_ROOT"
test "$(git rev-parse HEAD)" = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
git submodule update --init --recursive
cd "$ARMBENCH_ROOT"
```

入口首先检查 Linux、固定 commit、无已跟踪或未跟踪变更的 OpenPI worktree、全部递归
submodule（包括 LIBERO）、结果盘写权限、Docker/Compose、宿主机 NVIDIA GPU，以及 CUDA
容器能否看到 GPU。任一失败就不
启动 compose，但仍产生一个标为 incomplete 的失败证据目录。正式运行不得传
`--skip-container-gpu-probe`、`--allow-unattested-server`、`--allow-commit-mismatch` 或
`--continue-after-runtime-failure`。

固定 commit、checkpoint、server 地址、输出目录和 attestation 等证据关键参数由 outer
runner 与 Compose overlay 注入，不能放进 `--libero-args` 覆盖；runner 会拒绝完整名称及
缩写形式的受保护选项。下面模板因此只传实验矩阵参数。

### 6.1 Smoke

```bash
python3 -m integrations.openpi.libero_compose_run run \
  --openpi-root "$OPENPI_ROOT" \
  --armbench-root "$ARMBENCH_ROOT" \
  --results-root "$ARMBENCH_RESULTS_ROOT" \
  --run-id pi05_libero_smoke_001 \
  --libero-args "--task-suite libero_spatial --task-ids 0 --episode-indices 0:2 --modes async_unguarded --replan-steps 5 --latency-steps 0 --seed 7 --video-mode all"
```

该命令应规划并完成 2 个 rollout。它会等待 attested server 真正就绪，不使用固定 sleep；
server 若缺少/返回不匹配的 checkpoint attestation，evaluation 在第一个 rollout 前失败。

### 6.2 40-run Calibration Pilot

```bash
python3 -m integrations.openpi.libero_compose_run run \
  --openpi-root "$OPENPI_ROOT" \
  --armbench-root "$ARMBENCH_ROOT" \
  --results-root "$ARMBENCH_RESULTS_ROOT" \
  --run-id pi05_libero_pilot_001 \
  --libero-args "--task-suite libero_spatial --task-ids all --episode-indices 45:47 --modes async_unguarded,state_guard --replan-steps 5 --latency-steps 4 --position-threshold-m 0.01 --orientation-threshold-rad 0.10 --gripper-threshold 0.05 --max-requeries 2 --seed 7 --bootstrap-resamples 10000 --video-mode all"
```

### 6.2a 40-run Post-pilot Temporal-alignment Exploration

```bash
python3 -m integrations.openpi.libero_compose_run run \
  --openpi-root "$OPENPI_ROOT" \
  --armbench-root "$ARMBENCH_ROOT" \
  --results-root "$ARMBENCH_RESULTS_ROOT" \
  --run-id pi05_libero_alignment_pilot_001 \
  --no-build \
  --libero-args "--task-suite libero_spatial --task-ids all --episode-indices 47:49 --modes async_unguarded,latency_aligned --replan-steps 5 --latency-steps 4 --seed 7 --bootstrap-resamples 10000 --video-mode all"
```

该 run 是在原 pilot 结果已知后提出的方法探索。即使结果为正，也不能写成原计划内的
confirmatory finding；它只决定是否为 `latency_aligned` 冻结新的确认性矩阵。

### 6.3 300-run 核心矩阵

```bash
python3 -m integrations.openpi.libero_compose_run run \
  --openpi-root "$OPENPI_ROOT" \
  --armbench-root "$ARMBENCH_ROOT" \
  --results-root "$ARMBENCH_RESULTS_ROOT" \
  --run-id pi05_libero_core_001 \
  --libero-args "--task-suite libero_spatial --task-ids all --episode-indices 0:5 --modes async_unguarded,state_guard --replan-steps 5 --latency-steps 0,2,4 --position-threshold-m 0.01 --orientation-threshold-rad 0.10 --gripper-threshold 0.05 --max-requeries 2 --seed 7 --bootstrap-resamples 10000 --video-mode all"
```

### 6.4 三模式 Plan 与 Run

下面用 `N=4` 演示语法，它不是预设的正式值。先在本地 Windows 免费确认三模式矩阵；必须
从项目目录调用绝对解释器路径，不能在 `C:\WINDOWS\system32` 使用相对路径：

```powershell
$ArmbenchPython = 'D:\arm-planning-control-project\.venv\Scripts\python.exe'
Set-Location 'D:\arm-planning-control-project\project'

& $ArmbenchPython -m integrations.openpi.libero_runtime_eval plan `
  --task-suite libero_spatial --task-ids all --episode-indices 0:5 `
  --modes async_unguarded,state_guard,fixed_refresh `
  --replan-steps 5 --latency-steps 4 `
  --fixed-refresh-interval 4
```

输出必须是 `150 rollouts`、`50 matched_condition_groups`、三个 modes 和
`fixed_refresh_interval=4`。正式云端 run 把示例 4 替换为不重叠 pilot 已冻结的 N：

```bash
export FIXED_REFRESH_INTERVAL=4  # 仅语法示例；正式运行替换为 calibration note 中冻结的 N

python3 -m integrations.openpi.libero_compose_run run \
  --openpi-root "$OPENPI_ROOT" \
  --armbench-root "$ARMBENCH_ROOT" \
  --results-root "$ARMBENCH_RESULTS_ROOT" \
  --run-id "pi05_libero_intervention_n${FIXED_REFRESH_INTERVAL}_001" \
  --fixed-refresh-interval "$FIXED_REFRESH_INTERVAL" \
  --libero-args "--task-suite libero_spatial --task-ids all --episode-indices 0:5 --modes async_unguarded,state_guard,fixed_refresh --replan-steps 5 --latency-steps 4 --position-threshold-m 0.01 --orientation-threshold-rad 0.10 --gripper-threshold 0.05 --max-requeries 2 --seed 7 --bootstrap-resamples 10000 --video-mode all"
```

outer runner 会把 `--fixed-refresh-interval` 追加到 evaluator 参数并写入协议与逐 episode/query
记录。选了 `fixed_refresh` 却漏传 N 会 fail closed。该三模式组缓解的是 intervention/query
budget confounding；各 mode 仍顺序调用同一个随机推进的 policy server，不是 RNG 严格匹配。

### 6.5 200-run Horizon 消融

```bash
python3 -m integrations.openpi.libero_compose_run run \
  --openpi-root "$OPENPI_ROOT" \
  --armbench-root "$ARMBENCH_ROOT" \
  --results-root "$ARMBENCH_RESULTS_ROOT" \
  --run-id pi05_libero_horizon_001 \
  --libero-args "--task-suite libero_spatial --task-ids all --episode-indices 0:5 --modes async_unguarded,state_guard --replan-steps 1,10 --latency-steps 2 --position-threshold-m 0.01 --orientation-threshold-rad 0.10 --gripper-threshold 0.05 --max-requeries 2 --seed 7 --bootstrap-resamples 10000 --video-mode failures"
```

每个 `run-id` 必须全新，入口拒绝非空目录。`run` 无论成功、失败或中断都会尝试停止 compose
并 finalize；退出码非零时先保留目录，不要删行、补 CSV 或复用同一个 ID。

### 6.6 Finalize 与 Validate

`run` 已自动 finalize。下面的显式命令用于最终验收，或在 compose 已确认停止后加入账单等
补充证据再重新封存。finalize 之后不得继续修改目录；若修改，必须再次 finalize。

```bash
export RUN_DIRECTORY="$ARMBENCH_RESULTS_ROOT/pi05_libero_smoke_001"

python3 -m integrations.openpi.libero_compose_run finalize "$RUN_DIRECTORY"
python3 -m integrations.openpi.libero_compose_run validate "$RUN_DIRECTORY"
```

通过标准同时包括：finalize 输出 `complete=true`、`errors=[]`、
`manifest_validation.valid=true`；validate 输出 `valid=true`、`errors=[]`。还要打开根目录
`finalization.json`、`artifact_validation.json` 与 `manifest.json`，确认 finalization 和
manifest 的 complete 均为 true，artifact validation 的 valid 为 true。validate 除校验哈希、
字节数、缺失文件和未受 manifest 保护的额外文件外，还会独立解析原始 CSV、重建预注册矩阵、
核对 episode/query 决策语义，并从原始记录重算 aggregate、配对统计、三模式控制比较、条件
contrasts、progress、integrity 和确定性 summary。它仍不是防恶意篡改的数字签名。

销毁云实例前下载整个 `$RUN_DIRECTORY`、实际账单和平台计费记录，并在下载副本上再次执行
validate。不要只下载 `evaluation/summary.md` 或视频。

## 7. 统计报告规则

正式报告以 ITT（intention-to-treat 风格）为主，PP（per-protocol）为敏感性分析。这里借用
ITT/PP 的审计语义，不把机器人 episode 称为临床试验：

- 完整 run 的 ITT 分母是所有已规划且结构匹配的 rollout/pair；运行时或协议故障按失败计入，
  不能因为“不是模型问题”从 headline 中删掉。
- PP 只排除明确的非效能类别：`environment_runtime`、`observation_contract` 和
  `policy_transport_or_server`。`policy_timeout` 与 `policy_contract` 仍是部署策略结果，不因
  使用 PP 自动消失。
- condition 表同时给出 `rollouts/eligible_rollouts`、ITT/PP success；pair 表同时给出
  `paired_episodes/per_protocol_pairs`、ITT/PP 差值和排除计数。
- 缺 episode、pair 字段不匹配或未 finalize 的 run 不是有效 ITT 数据集，不能用较小的实际
  分母冒充完整主矩阵。

evaluator 自动生成以下统计量：

- 每个 condition 的 ITT/PP 成功率与各自 95% Wilson 区间；
- `state_guard - async_unguarded` 的 ITT 配对成功率差值及 PP 敏感性差值；
- 对 pair 差值进行 10,000 次确定性 bootstrap 的 95% 区间；
- discordant pair 计数：`guard_wins`、`unguarded_wins`、`ties`；
- 两侧 exact McNemar p 值；
- 对同一 run artifact 中全部 `selected_tasks` 条件进行 Holm 多重比较校正后的 p 值；
- delay-vs-zero 与 horizon-vs-five 的 matched condition contrasts；
- 查询相对开销、平均拒绝数、`stale_action_steps` 和三层推理时延。

`stale_chunks_executed` 只说明多少个有注入延迟的动作块被接受；`stale_action_steps` 记录这些
块中真正从队列取出并执行的 VLA 动作步，因此是主要陈旧动作暴露指标。查询等待期间继续
执行上一动作的步数另记为 `latency_action_steps`。guard 可能接受阈值内的旧动作块，所以
不能预先声称 `state_guard` 会令 stale 指标归零。

`intervention_control_comparisons.json/csv` 已由 evaluator 生成，并由独立 validator 从
`per_episode.csv` 重算。两模式 run 仍会生成空的同名 artifact；只要 matrix 含
`fixed_refresh`，它就必须非空并进入两层 manifest。其验收含义预注册为：

- 每个纳入 group 同时有 `async_unguarded/state_guard/fixed_refresh`，task、initial-state
  hash、seed、horizon、delay 和冻结的 N 一致；列出缺失、字段不匹配与 PP 排除数量。
- 主要机制对比是 `state_guard - fixed_refresh`；`fixed_refresh - async_unguarded` 用于估计
  单纯定时刷新效应，不取代原始 `state_guard - async_unguarded` 主比较。
- 同时报告 ITT/PP success 差、区间，以及 interventions、policy queries、
  `stale_action_steps` 的均值差和残余 budget mismatch，不能只展示成功率。
- 只有干预与查询预算足够接近时，guard-vs-refresh 差异才是 state-aware selection 的支持性
  证据；预算不接近时结果仍受混淆，必须降级解释。

该 comparison artifact 已实现，但只有真实三模式 run 完整且通过独立 validator 后，才能
作为受边界约束的机制证据；它不会自动证明混淆已被完全隔离，也不修复顺序运行的 JAX RNG
差异。

报告顺序必须是效应量、95% CI、样本量，再给 p 值。只有比较在实验前已注册、pair 数足够、
方向与效应量一致，并且 `mcnemar_holm_p < 0.05` 时，才能写“具有统计显著差异”。否则写
“观测到 X 个百分点差异，95% CI 为 [...]”，不得把 raw p 值或某个 task 的偶然结果包装
为显著。

主比较锁定为 h=5/delay=4。其他 `selected_tasks` 条件仍进入 Holm 校正，但解释为次要比较；
只有 `suite_coverage_complete=true` 才能把 scope 简写成“完整 Spatial suite”。
核心矩阵与 horizon 消融分属两个 artifact；若在同一报告中把两个 artifact 的条件组成一个
推断族，必须对合并后的 raw p 值重新做一次 Holm 校正，不能直接拼接两份校正后 p 值。
task-level N=5 只报告原始成功数和区间。动作步和 query 是 episode 内相关样本，不能把数千
动作步当成数千独立样本来缩窄置信区间。

基础设施故障绝不静默删除。ITT 必须保留并计为失败，PP 只能按上述固定类别排除；若要重跑，
使用新目录并同时发布原始与重跑产物。正式 headline matrix 要求完整且 0 个非效能运行时
故障，使 ITT 与 PP 主结论一致；否则结果只算 pilot，并明确给出 planned/completed、ITT、
PP、故障类型和排除数量。

## 8. 视频、失败与证据保留

每个正式 run 是一个自包含目录，至少应有：

```text
<run-id>/
  preflight.json
  resolved_compose_config.json
  checkpoint_attestation.json
  openpi_server.log
  compose_up.log
  compose_up.json
  compose_stop.json
  finalization.json
  artifact_validation.json
  manifest.json
  evaluation/
    resolved_protocol.json
    environment.json
    progress.json
    integrity.json
    manifest.json
    per_episode.csv
    per_query.csv
    aggregate.json
    aggregate.csv
    paired_comparisons.json
    paired_comparisons.csv
    condition_contrasts.json
    condition_contrasts.csv
    intervention_control_comparisons.json  # 三模式确认性 run 必须非空
    intervention_control_comparisons.csv   # 三模式确认性 run 必须非空
    summary.md
    run.log
    provenance/armbench_source/
    videos/
```

`checkpoint_attestation.json` 保存完整 checkpoint 文件清单、逐文件 hash、总大小和内容摘要；
`environment.json` 保存公开 server metadata、实际命令、GPU/包版本、OpenPI/ArmBench git 状态
与运行时源码 hash；`provenance/armbench_source/` 保存本次实际运行的关键源码字节。失败时
还可能出现 `evaluation/run_error.json`，它必须保留。

`compose_up.json` 和 `compose_stop.json` 记录实际 argv、返回码与耗时。只有确认 compose stop
成功后，finalize 才把不再增长的 server log 和整个 `evaluation/` 子 artifact 一起纳入根
manifest；这避免先 hash 日志、服务随后又继续写入的竞态。根 manifest 与 evaluation manifest
构成两层目录校验，缺文件、多出未保护文件、字节数或 SHA-256 不一致都会使 validate 失败。
根 `artifact_validation.json` 还保存对 evaluation 原始行、协议语义和派生统计的独立重算
报告；validate 会再次重算并要求结果与该记录完全一致。

smoke、pilot、核心矩阵和三模式干预控制使用 `--video-mode all`，让成功、失败、guard 拒绝与
scheduled refresh 均可追溯；
horizon 扩展可使用 `--video-mode failures` 控制存储。正式运行前把全视频磁盘开销计入
预算。此外为作品展示锁定三个案例：一个成功、一个正常任务失败、一个发生
`rejected_state_mismatch` 的 episode。选择规则用“满足条件的最小 task_id，再取最小
episode_index”，避免只挑视觉效果最好的案例；若核心矩阵没有某类案例，如实写“未观察到”。

视频只用于解释机制，数值结论必须来自 CSV/JSON。失败视频、异常日志和负结果与成功案例
同等保留。`manifest.json` 校验的是目录一致性；实际加载内容由 full attestation、公开 server
metadata、固定 commit、包含全部未跟踪文件的 worktree 状态、递归 submodule 状态、resolved
compose config 和 server log 共同约束。即使全部一致，它们仍不是上游签名或安全认证。

## 9. 完成度分级

| 等级 | 必须证据 | 能否作为高竞争力 VLA 项目 |
| --- | --- | --- |
| L0：evaluator | 本地单元测试、fake policy/env、协议与 artifact | 只能证明 VLA 基础设施能力 |
| L1：real-checkpoint smoke | attested 官方 checkpoint、2 个 nominal rollout、完整且通过 validate 的证据 | 可证明接通真实 pi0.5，不能作性能结论 |
| L2：配对核心实验 | 300 rollout 完整矩阵、0 非效能故障、ITT/PP 配对统计、失败视频 | 可以作为有竞争力的 VLA 简历重点，但不作 state-specific 机制归因 |
| L3：机制控制 | L2 + 不重叠 N 校准 + 150 rollout 三模式控制 + 通过验收的 intervention artifact | 更接近研究型项目，可写受边界约束的机制证据 |
| L4：投稿证据 | 官方 50 initial states/task、多个 suite、独立复跑、完整 compute 报告 | 才接近论文实验规模，¥300-800 通常不够 |

代码量、测试数和漂亮可视化不能把 L0 自动变成 L2。对 VLA 岗位真正增值的是：真实开放
checkpoint、标准 benchmark、清晰对照、配对统计、失败审计和可复现产物同时存在。

## 10. 简历可写与不可写

达到 L0 但尚无 GPU 结果时，只能写：

> 实现 OpenPI-compatible pi0.5-LIBERO 异步运行时评测器，复现官方双相机/语言/状态输入
> 协议，并支持延迟注入、状态失配拒绝、配对实验、置信区间与可校验失败产物。

这段只能描述已经测试过的 evaluator 能力，不能附加“成功率 X%”“P95 为 X ms”“减少
X% 陈旧动作”等结果数字。模板中的方括号不是可估算字段；必须等 attested 真实 checkpoint
run 完整 finalize、validate 通过后再填写。

达到 L2 后，才可将真实结果填入下面模板：

> 基于固定 OpenPI commit 与官方 pi0.5-LIBERO checkpoint，在 LIBERO Spatial 的
> `[N]` 个成对 rollout 中评估 `0/2/4` 步异步延迟；状态失配拒绝将 h=5、delay=4 下
> ITT 成功率由 `[X%]` 改变为 `[Y%]`（配对差 `[D]`，95% CI `[L,U]`；PP 敏感性结果
> `[PP]`），代价为 `[Q%]` 额外策略查询，陈旧动作暴露改变 `[S]` steps/episode；保留逐
> query 状态失配、P95 推理时延、attested checkpoint digest 和全部失败视频。

数字必须直接来自 `paired_comparisons.csv`、`aggregate.csv` 和 `per_query.csv`。结果若为负，
就写“量化了代价/识别了失效区间”，不能改写为“提升”。

只有达到 L3，且 `intervention_control_comparisons` 显示 state guard 与 fixed refresh 的实际
intervention/query budget 足够接近，才可使用“state-aware 选择优于/不同于定时刷新”措辞。
否则只能写 guard 相对 unguarded 的观测差异，不能把额外查询、新观测或 hold 的效果归因给
失配阈值。

任何阶段都不可写：

- 训练、微调或提出了 pi0.5；
- 超过官方 pi0.5 的 96.85 平均分；
- 未保存真实 server 证据的“pi0.5 部署”；
- 使用 `--allow-unattested-server` 的诊断 run 作为 checkpoint 性能证据；
- 形式化安全、零碰撞、硬实时或 certified safety；
- Isaac Lab、真机、ROS2 或 `libfranka` 验证；
- 把 5 个 initial states 写成 5 个 random seeds；
- 只凭 raw p 值或未校正比较声称统计显著；
- 将 scripted/loopback 结果与学习 checkpoint rollout 混在同一性能表中。
- 在真实 checkpoint run 完成并通过 finalize/validate 之前填写任何结果数字。

## 11. 最终验收标准

L2 验收必须逐项通过：

1. `git rev-parse HEAD`、`evaluation/environment.json`、server attestation 和 server log 都
   指向固定 OpenPI commit。
2. config/checkpoint URI、checkpoint content SHA-256、server source SHA-256、GPU 型号、
   driver、Python/包版本、完整命令和实际费用可追溯。
3. `preflight.json` 为 ready；`evaluation/progress.json` 显示 300/300 complete；
   `evaluation/integrity.json` valid；根 `manifest.json` complete 并覆盖全部最终文件。
4. 所有 mode pair 的 task、initial state hash、horizon 和 delay 一致，运行顺序交替。
5. 正式 headline matrix 的非效能运行时故障数为 0；任何早期失败 artifact 仍保留，ITT/PP
   分母与排除数一致。
6. 主比较为预注册的 h=5/delay=4，没有看结果后换 primary cell。
7. ITT/PP 成功率、Wilson CI、配对差、bootstrap CI、discordant counts、McNemar/Holm、
   查询开销、`stale_action_steps` 和 P95 推理时延均有可复算表格。
8. 位置、姿态、夹爪失配按各自单位报告，不合成没有标定意义的单一 max score。
9. 至少能演示一个成功、一个任务失败和一个拒绝案例；视频能对应到 episode/query 行。
10. README/简历/面试表述遵守本文件边界，不声称训练、真机、Isaac Lab 或安全认证。
11. 在一台新的 Linux/NVIDIA 环境中，按照保存的命令可以重建容器并重现至少 smoke。
12. 能口头解释 pi0.5 是策略、LIBERO 是 benchmark/simulator、ArmBench 是运行时 evaluator，
    为什么注入 delay steps 与实测 inference milliseconds 是两种不同变量，以及为什么
    matched-initial-state sequential study 并不表示共享了服务端 JAX 随机噪声。

L3 还必须满足：pilot initial states 与正式矩阵不重叠；calibration note 在查看确认性成功率
前冻结 N；三模式每个 matched group 完整；`intervention_control_comparisons` 进入 manifest
并通过 validator；报告 guard-vs-refresh 的 ITT/PP 效应、区间和 intervention/query/stale
budget mismatch。这个控制缓解干预混淆，但不升级为 RNG 严格配对。

只有 L2 全部通过，才把它定位为“高竞争力 VLA 简历项目”。L1 之前继续称
OpenPI-compatible evaluator；L1 只能证明接通；L3/L4 才逐步接近独立研究成果和投稿证据。
