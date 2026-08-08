# ArmBench 中文代码导读

状态：Current / Operational
适用版本：仓库 `main` 的当前实现
阅读目标：让维护者可以从命令行入口一路追到 provider、worker、调度器、Panda
保障链、MuJoCo 执行和 artifact validator，并在正确的边界设置断点。

这不是一份新的算法说明，也不替代[架构与主张边界](PROJECT_ARCHITECTURE_ZH.md)、
[调试指南](DEBUGGING.md)或各个冻结实验协议。代码导读只记录当前实现的导航和
可验证的数据契约。仓库路径均相对于 `project/` 根目录。

## 先记住一个事实：当前有三条运行路径

ArmBench 的类名可以拼成一条很长的链，但当前实现并没有把每个组件都接成同一个
在线系统。调试时先判断自己要复现哪一条路径。

```text
命令行
  |
  +-- A. provider/异步运行时路径 ------------------------------+
  |      MuJoCo observation -> VLAObservation                    |
  |             -> policy/provider -> worker -> PolicyOutcome     |
  |             -> AsyncChunkDispatcher -> async Panda loop      |
  |             -> legacy ActionChunkGuard/BrakingTrajectoryGuard|
  |             -> MuJoCo torque execution                       |
  |
  +-- B. 正式集成 Panda CPU 证据路径 ----------------------------+
  |      registered scripted ActionChunk                           |
  |          -> IntegratedPandaSupervisor.supervise                |
  |          -> OSQP -> continuous collision -> dynamics brake    |
  |          -> (task run only) execute_trajectory -> trace        |
  |                                                               |
  +-- C. provider-neutral 异步集成发布路径 ------------------------+
         policy/provider -> LatestPolicyWorker
             -> LatestIntegratedPandaWorker
             -> IntegratedPandaSupervisor.supervise
             -> AtomicPandaPlanGate.commit
             -> complete plan OR zero policy actions
             -> manifest/source-hash/replay validator
```

路径 A 的入口是 `vla-panda-async-run`；它用于验证独立观测 worker、阻塞策略 worker、
观测年龄对齐和墙钟控制循环，内置策略是 scripted、非学习的。路径 A 的运行时目前
使用 `ActionChunkGuard` 与 `BrakingTrajectoryGuard`，**不会自动调用**
`IntegratedPandaSupervisor`。

路径 B 的入口是 `vla-integrated-fault-matrix` 和 `vla-integrated-task-run`。它是
同步 CPU 参考链：先把完整 action chunk 做完 QP、连续碰撞和每个动作边界的制动
检查，再决定是否允许完整前缀进入任务执行。它的动作源是 scripted RRT-Connect
参考，不是学习式 VLA，也没有硬实时调度保证。

路径 C 的入口是 `vla-cpu-runtime-run`。它补上了 provider 与正式 supervisor 之间的
异步接线：策略推理和完整动作保障分别运行在独立 worker，控制侧只通过原子 gate
发布完整计划。当前 provider 是 mock、冻结响应或 OpenPI-compatible 接口 fixture，
因此证明的是运行时契约，不是真实 checkpoint 任务效果。

三条路径共享 `VLAObservation`、`ActionChunk`、provider 语义校验、MuJoCo Panda
模型和证据序列化工具，但结果不能混合统计。A 是带力矩物理执行的旧 guard 闭环，
B 是同步 supervisor 与任务证据，C 是新的异步 supervisor 发布边界；真实 provider
接入 C 后仍需生成新的 GPU/任务 artifact，不能只修改名称。

## 1. 从 CLI 开始

### 1.1 入口和命令分发

| 位置 | 入口 | 作用 | 首个断点 |
| --- | --- | --- | --- |
| [`src/armbench/__main__.py`](../src/armbench/__main__.py) | `main()` | `python -m armbench` 的薄入口 | 查看 `arguments` 是否按预期传入 |
| [`src/armbench/cli.py`](../src/armbench/cli.py) | `build_parser()` | 注册所有子命令和参数类型 | 查看 `args.command`、路径和默认值 |
| 同上 | `main(arguments)` | 解析参数并调用具体 runner/validator | 在对应 `if args.command == ...` 分支停住 |
| [`scripts/accept_integrated_panda.ps1`](../scripts/accept_integrated_panda.ps1) | 脚本主体 | 从任意当前目录定位仓库、运行矩阵/任务并可启动 viewer | 查看解析出的 `$ProjectRoot`、`$Python` 和输出目录 |

从 `C:\WINDOWS\system32` 运行时不要使用 `..\\.venv\\Scripts\\python.exe`；相对
路径会从 `C:\WINDOWS` 解析。使用绝对解释器，或使用自定位 acceptance 脚本：

```powershell
$Project = 'D:\arm-planning-control-project\project'
$Python = 'D:\arm-planning-control-project\.venv\Scripts\python.exe'
& $Python -m armbench doctor
& $Python -m armbench vla-integrated-guard-smoke
powershell -NoProfile -ExecutionPolicy Bypass -File `
  "$Project\scripts\accept_integrated_panda.ps1" -Visualize
```

### 1.2 命令到函数的速查表

| 命令 | 实际调用 | 产物/范围 |
| --- | --- | --- |
| `vla-async-smoke` | `run_async_runtime_smoke` | worker 与控制 tick 组件 smoke |
| `vla-process-smoke` | `run_process_runtime_smoke` | spawn 子进程 worker smoke |
| `vla-qp-smoke` | `run_qp_projection_smoke` | OSQP 约束投影组件 |
| `vla-integrated-guard-smoke` | `run_integrated_panda_guard_smoke` | 单个同步集成决策 |
| `vla-integrated-fault-matrix` | `run_integrated_panda_fault_matrix` 后立即 validator | 27 个注册故障案例 |
| `vla-integrated-fault-validate` | `validate_integrated_panda_fault_matrix` | 只读重算现有矩阵 |
| `vla-integrated-task-run` | `run_integrated_panda_tasks` 后立即 validator | 规划、保障和 MuJoCo 力矩任务 |
| `vla-integrated-task-validate` | `validate_integrated_panda_tasks` | 重算任务规划、trace 和接触指标 |
| `vla-cpu-runtime-run` | `run_cpu_runtime_completion` 后立即 validator | C 路径的 17 个 provider/并发/原子发布案例 |
| `vla-cpu-runtime-validate` | `validate_cpu_runtime_completion` | 校验源码哈希并重跑 C 路径全部案例 |
| `vla-panda-async-run` | `execute_async_panda_benchmark` | A 路径的墙钟矩阵、事件、trace、可选视频 |
| `vla-panda-async-validate` | `validate_async_panda_artifact` | A 路径 artifact 内部一致性检查 |
| `mujoco-self-collision-validate` | `validate_self_collision_audit` | 独立连续自碰撞审计 |
| `mujoco-dynamics-braking-validate` | `validate_dynamics_braking_audit` | 独立动力学制动审计 |

命令分支本身通常不包含算法；若 JSON 输出不符合预期，先在 CLI 分支确认参数，
再进入被调用的模块，而不是先改底层数值代码。

## 2. 共享数据契约：先看形状，再看算法

### 2.1 观测和动作

定义在[`src/armbench/vla/types.py`](../src/armbench/vla/types.py)：

| 对象 | 形状/类型 | 语义 | 调试检查 |
| --- | --- | --- | --- |
| `VLAObservation.exterior_image` | `(224, 224, 3)`, `uint8` | 外部相机帧 | `shape`, `dtype`, `std`，确认不是全黑 |
| `VLAObservation.wrist_image` | `(224, 224, 3)`, `uint8` | 腕部相机帧 | 同上，确认视野没有错位 |
| `joint_position` | `(7,)`, finite float | Panda 关节位置，rad | 与 MuJoCo `qpos[arm_qpos_addresses]` 对齐 |
| `gripper_position` | `(1,)`, finite float | 本地夹爪标量 | 检查是否在调用方约定的 `[0,1]` |
| `state` | `(8,)` | `joint_position` 与夹爪拼接 | 确认顺序是 7+1 |
| `ActionChunk.actions` | `(H, 8)`, finite float | 前 7 列关节速度，末列本地夹爪动作 | 检查 `H > 0`、序列号和 latency |
| `RawActionChunk.actions` | `(H, D_provider)` | provider 原生动作，不能直接进 Panda guard | 先检查 `ActionSemantics` |

`VLAObservation.to_openpi_droid()` 生成 OpenPI DROID 键；
`to_openpi_libero()` 生成 LIBERO 键。图像键相同并不代表动作语义相同。

LIBERO 风格末端动作是 `(H,7)`：3 个平移、3 个旋转、1 个夹爪（LIBERO 约定
`-1=open, +1=closed`）。[`PandaCartesianActionAdapter.adapt`](../src/armbench/vla/cartesian_adapter.py)
通过 6x7 hand Jacobian 和阻尼最小二乘微分逆运动学，输出本地 `(H,8)`；它是
运动学组件适配，不等价于 robosuite 的 torque-level OSC。

### 2.2 时间和身份字段

- `sequence_id` 标识观测序列，不是动作索引；provider response 必须回显同一序列。
- `captured_at_s`、`received_at_s` 和 worker 的 `submitted/started/finished` 使用
  monotonic clock。出现负年龄时应停在时钟/测试替身，而不是用 wall clock 补偿。
- `ActionChunk.age_ms(observation)` 需要先通过序列匹配，再计算响应年龄。
- `canonical_action_sha256` 使用 little-endian float32、C-order；改变 dtype、顺序或
  一字节都会改变响应哈希。

## 3. Provider、worker 与 dispatcher

### 3.1 Provider 边界

| 文件 | 类/函数 | 输入 -> 输出 | 断点关注点 |
| --- | --- | --- | --- |
| [`vla/policy.py`](../src/armbench/vla/policy.py) | `BoundedOpenPIBackend.infer` | OpenPI MessagePack/WebSocket 请求 -> mapping | 连接、超时、服务端 metadata 和原始响应 |
| 同上 | `OpenPIPolicyClient.infer` | `VLAObservation` -> `(H,8) ActionChunk` | `actions` 是否存在且恰为 `(expected_horizon,8)` |
| 同上 | `ScriptedActionChunkPolicy.infer` | scripted chunk -> `ActionChunk` | 仅用于 CPU 测试，不要标成 learned policy |
| [`vla/openpi_provider.py`](../src/armbench/vla/openpi_provider.py) | `OpenPILiberoRawProvider.infer_raw` | LIBERO provider 响应 -> `RawActionChunk` | 原生动作维度和身份元数据 |
| [`vla/provider_contract.py`](../src/armbench/vla/provider_contract.py) | `require_semantic_compatibility` | 两份 `ActionSemantics` -> 通过/异常 | 坐标系、周期、尺度、旋转和夹爪字段必须逐项相等 |
| 同上 | `FrozenResponseProvider.infer_raw` | 已校验 archive + observation -> 下一条冻结响应 | 响应顺序、观测哈希、耗尽状态 |
| 同上 | `AdaptedActionChunkPolicy.infer` | `RawActionChunk` -> adapter 后 `ActionChunk` | 语义门禁和 sequence 检查发生在 adapter 前 |

provider 失败应在这里保留异常类型和消息；不要把形状错误静默变成零动作。
上层 worker 会把异常编码成 `PolicyOutcome(failure_type, failure_message)`，再由
dispatcher 或 runtime 进入 hold。

### 3.2 Worker：最新请求优先，不取消正在执行的调用

| 文件 | 类/函数 | 行为 |
| --- | --- | --- |
| [`vla/async_worker.py`](../src/armbench/vla/async_worker.py) | `LatestPolicyWorker.submit` | 只保留一个 pending observation；新请求替换旧 pending |
| 同上 | `LatestPolicyWorker._run` | 独立线程调用 `policy.infer`，生成 `PolicyOutcome` |
| 同上 | `drain` / `metrics` / `close` | 控制线程非阻塞取结果、观察丢弃计数、关闭线程 |
| [`vla/process_worker.py`](../src/armbench/vla/process_worker.py) | `ProcessPolicyWorker` | spawn 子进程隔离策略资源；队列大小为 1 |
| 同上 | `_process_main` | 子进程构造 policy、执行 infer、发布可序列化 outcome |
| [`vla/async_panda.py`](../src/armbench/vla/async_panda.py) | `LatestObservationWorker` | 从 MuJoCo 状态渲染两路图像并生成观测 |
| [`vla/integrated_panda_async.py`](../src/armbench/vla/integrated_panda_async.py) | `LatestIntegratedPandaWorker` | 在第二个独立线程运行完整 QP、连续碰撞和动力学制动监督 |
| 同上 | `AtomicPandaPlanGate.commit` | 激活时重查 deadline、状态、顺序和 reset epoch；只发布完整计划或空动作 |

检查 worker 卡住时先看 `metrics()`：
`submitted -> started -> completed` 是否推进，`pending` 是否一直为真，
`superseded_pending` 是否快速增长，以及 `worker_alive`/`closed` 是否合理。不要在
控制线程中直接调用 blocking `infer`，否则即使 dispatcher 正确，deadline 也会失真。

### 3.3 Dispatcher：把响应年龄转换为动作后缀

[`vla/async_dispatch.py`](../src/armbench/vla/async_dispatch.py) 的
`AsyncChunkDispatcher` 维护一个当前 active response：

1. `publish(outcome)` 拒绝旧 request、失败 outcome、时钟回退、超 deadline 或已耗尽
   horizon；通过时计算 `action_offset`。
2. `select(hold_action)` 在每个控制 tick 重算当前年龄和 `action_index`；无有效响应时
   返回 `status="hold"`，有可用后缀时返回 `status="execute"`。
3. `metrics()` 是定位积压的第一手数据：`accepted_responses`、`rejected_responses`、
   `executed_commands`、`hold_commands` 和 `hold_reason`。

状态转移可简化为：

```text
无 active --publish(success,fresh)--> active
active --select(age within horizon)--> execute(index=ceil(age/period))
active --age/deadline/horizon failure--> hold(reason)
任意 --publish(old/failure/invalid)--> hold(reason)
```

这里的 `select` 只选择 action chunk 后缀，不做 OSQP、连续碰撞或动力学制动。
这些检查在 A 路径由 `async_panda.py` 的 guard/repair 逻辑完成；在 B 路径由下面的
`IntegratedPandaSupervisor` 完成。

## 4. IntegratedPandaSupervisor：正式 CPU 保障链

### 4.1 入口和决策对象

[`vla/integrated_panda_guard.py`](../src/armbench/vla/integrated_panda_guard.py)：

- `IntegratedPandaGuardConfig`：控制周期、response/supervision budget、关节速度/加速度、
  状态不匹配阈值、QP budget 和 braking config。
- `IntegratedPandaSupervisor.__init__`：确认 robot 与 continuous checker 是同一个
  七自由度 Panda，并创建 `QPActionProjector`。
- `IntegratedPandaSupervisor.supervise(q, qvel, chunk, observed_q, response_age_ms)`：
  从一个冻结状态生成**原子** `IntegratedPandaDecision`。
- `IntegratedPandaDecision`：accepted 时必须有完整 `H x 8` 动作、`H+1 x 7` 预测位置、
  每边一个连续碰撞证书和一个制动证书；拒绝时 `executable_actions` 必须为空，
  不暴露部分前缀。

### 4.2 逐阶段控制流

```text
输入 q(7), qvel(7), observed_q(7), response_age, ActionChunk(Hx8)
  |
  +-- deadline / state_alignment --------------------> fallback brake
  |
  +-- QPActionProjector.project_chunk ----------------> fallback brake
  |      每步 OSQP box + acceleration + joint-limit constraints
  |      primary collision check 失败时尝试 zero-velocity fallback
  |
  +-- checker.edge_certificate(q_i, q_{i+1}) --------> fallback brake
  |      连续静态/注册自碰撞；indeterminate 也 fail-closed
  |
  +-- generate_dynamics_validated_brake(q_i, dq_i) ---> fallback brake
  |      每个动作边界检查停止轨迹、mj_inverse、力矩范围、插值边
  |
  +-- IntegratedPandaDecision(status="accepted")
```

决策状态和典型失败阶段：

| `status` | 含义 | 是否有可执行 policy action |
| --- | --- | --- |
| `accepted` | 完整 chunk 所有阶段通过 | 是，完整前缀 |
| `verified_brake` | policy chunk 被拒，但从当前状态生成并验证了停止轨迹 | 否 |
| `hold` | 当前已静止或只允许保持 | 否 |
| `unrecoverable_stop` | 停止轨迹也未通过，不能输出运动 | 否 |

常见 `failure_stage` 是 `deadline`、`state_alignment`、`qp_projection`、
`continuous_collision`、`dynamics_braking` 或 `supervision_budget`。在断点处先看
`failure_stage/failure_index/reason`，再看对应证书，而不是只看最终 status。

### 4.3 QP、连续碰撞和制动的代码地图

| 文件 | 关键入口 | 数据和判定 |
| --- | --- | --- |
| [`vla/qp_projection.py`](../src/armbench/vla/qp_projection.py) | `QPActionProjector.project_chunk` | 每个动作取前 7 列速度；输出 `QPProjectionResult`，`predicted_positions` 为 `(accepted_steps+1,7)` |
| 同上 | `_box_bounds` / `_solve` | 关节速度、加速度、限位边界；OSQP status、迭代数、step latency |
| [`mujoco_sim/continuous_collision.py`](../src/armbench/mujoco_sim/continuous_collision.py) | `ContinuousMuJoCoCollisionChecker.edge_certificate` | 对一个线性关节边返回 safe/collision/indeterminate certificate |
| 同上 | `_interval_status` | midpoint 距离 + 运动上界；达到深度/评估预算时返回 indeterminate，调用方 fail-closed |
| [`mujoco_sim/dynamics_braking.py`](../src/armbench/mujoco_sim/dynamics_braking.py) | `generate_dynamics_validated_brake` | 从 `(q,qvel)` 构造加速度受限停止轨迹，逐采样运行 `mj_inverse` 并检查 actuator effort 和边 |
| 同上 | `DynamicsBrakingResult` | `times (N,)`、`positions/velocities/accelerations (N,7)`、effort `(evaluated,7)`、failure sample |

注意：QP 内部可以暂时填充部分 `projected_actions`，但 supervisor 在非 `accepted`
决策中会把对外 `executable_actions` 置为空。这是“无部分前缀泄漏”的原子边界，
不要为了方便调试而直接执行 QP 中间数组。

## 5. 从保障到 MuJoCo 任务执行

[`vla/integrated_panda_task.py`](../src/armbench/vla/integrated_panda_task.py)
负责正式任务 artifact，顺序是：

1. `_case_specs` 从 `REGISTERED_PROFILES` 展开场景、负载、延迟、速度和 seed。
2. `_plan_reference` 使用 `RRTConnect`；全目标路径再用 `shortcut_path` 和
   `time_parameterize`，短 smoke 使用注册直线路径。
3. `_scripted_chunk` 将时间参数化的 `(N,7)` 位置差分成 scripted `(H,8)` 关节速度块。
4. `_evaluate_case` 创建 guard robot（障碍膨胀）、运行 supervisor；只有 accepted 才
   创建 torque-control execution robot。
5. `execute_trajectory` 位于[`mujoco_sim/execution.py`](../src/armbench/mujoco_sim/execution.py)：
   每个控制周期按 delayed 或 velocity-prediction feedback 计算 PD + bias torque，按
   MJCF force range 裁剪，并记录实际状态、接触、限位、饱和和误差。
6. `_trace_document` 保存以下 NPZ 字段：

   - `certified_times_s`: `(H+1,)`
   - `certified_positions`: `(H+1,7)`
   - `certified_velocities`: `(H+1,7)`
   - `times`: 实际控制采样时间 `(T,)`
   - `desired_positions`, `actual_positions`: `(T,7)`
   - `applied_torques`: `(T,7)`

任务 checker 明确排除固定张开夹爪的 `left_finger/right_finger` body pair；这不是
“忽略所有自碰撞”。其余注册 pair 仍经过连续检查，范围见任务 summary 的
`allowed_collision_matrix`。

## 6. Artifact validator 在验证什么

### 6.1 正式集成故障矩阵

`validate_integrated_panda_fault_matrix(directory)` 的顺序是：

1. 检查 `manifest.json` 的文件集合、大小、SHA-256 和 inventory hash；
2. 读取 `summary.json`，核对 schema、scope、MuJoCo/Numpy/OSQP 版本、Panda scene hash；
3. 核对 `implementation_sha256`；
4. 从 `cases.json` 重建注册 27 个输入，重新调用 supervisor；
5. 重新解析 CSV、比较确定性字段和聚合统计，确认 `summary.md` 可重建。

### 6.2 正式 MuJoCo 任务

`validate_integrated_panda_tasks(directory)` 在上面基础上还会：

- 重新规划、保障和运行闭环 MuJoCo torque execution；
- 对每个 trace 逐数组比较形状和数值；
- 重算目标误差、跟踪误差、接触、限位和安全任务成功；
- 检查允许碰撞矩阵及每个 case 映射。

### 6.3 异步 Panda artifact

`validate_async_panda_artifact(directory)` 主要从 `per_case.csv`、`events.jsonl`、
`provenance.json` 和 NPZ trace 重算事件计数、时延、控制 tick、接触和 summary，并
检查 manifest 文件哈希。它是路径 A 的 artifact 一致性验证；与路径 B 不同，当前
validator 不会把完整 integrated supervisor 重新跑一遍，也不应被写成 formal task
certificate。

### 6.4 异步集成发布 artifact

`validate_cpu_runtime_completion(directory)` 是路径 C 的语义重放 validator：

1. 要求 artifact 只有注册的 5 个数据文件，并逐字节核验 recursive manifest；
2. 核对 10 个关键实现文件的当前 SHA-256，注释或逻辑改动都会使旧报告失效；
3. 要求 `cases.json` 与源码注册的 17 个场景完全一致，不能通过重新签名改题；
4. 交叉核对 CSV、summary JSON、provider 聚合和 Markdown；
5. 重新运行 provider worker、integrated supervisor worker 和原子 gate；
6. 比较 status、reason、动作数量、fallback、线程边界与 reset/order 语义。

墙钟 latency 和轮询 tick 会被保存并聚合，但不要求逐次重跑完全相等；会影响执行器
语义的字段必须精确相等。

推荐的只读验收命令：

```powershell
& $Python -m armbench vla-integrated-fault-validate `
  reports\integrated_panda_fault_matrix_001
& $Python -m armbench vla-integrated-task-validate `
  reports\integrated_panda_task_001
& $Python -m armbench vla-cpu-runtime-validate `
  reports\cpu_runtime_completion_001
& $Python -m armbench vla-panda-async-validate `
  reports\async_panda_closed_loop_400ms_20mm_v3_001
```

数字通过后再用 `mujoco-view` 播放 trace。viewer 只证明保存的轨迹能被读取和显示；
它不是 validator 的替代品。

## 7. 断点和故障定位顺序

### 7.1 先确认运行环境

```powershell
& $Python -m armbench doctor --json
& $Python -m pytest -q
```

若 `doctor` 失败，先修 Python、MuJoCo Menagerie 缓存和工作目录；不要进入 QP 或
碰撞代码。若只想验证一个边界，可依次运行 `vla-async-smoke`、`vla-process-smoke`、
`vla-qp-smoke`、`vla-integrated-guard-smoke`。

### 7.2 观测/请求错误

断点：

- `vla/observation.py::MuJoCoDroidObservationBuilder.capture`
- `vla/types.py::VLAObservation.__post_init__`
- `vla/types.py::VLAObservation.to_openpi_droid`

检查图像是否 `(224,224,3) uint8`、两路 camera hash 是否变化、`q` 是否 `(7,)`、
`sequence_id` 是否单调。全黑图像、重复 frame 或非有限 state 都应在 provider 前被
定位。

### 7.3 provider/语义错误

断点：

- `policy.py::OpenPIPolicyClient.infer`
- `provider_contract.py::require_semantic_compatibility`
- `provider_contract.py::AdaptedActionChunkPolicy.infer`
- `cartesian_adapter.py::PandaCartesianActionAdapter.adapt`

先打印动作 shape、source、sequence 和 semantics hash；再看是“形状不对”还是“语义
字段不一致”。不要通过 reshape、静默裁剪或改 gripper 顺序来绕过门禁。

### 7.4 worker/dispatcher 不推进

断点：

- `LatestPolicyWorker.submit` -> `_run` -> `drain`
- `LatestIntegratedPandaWorker.submit` -> `_run` -> `drain`
- `AtomicPandaPlanGate.commit`
- `ProcessPolicyWorker._process_main`
- `AsyncChunkDispatcher.publish` -> `select`

对照 `submitted/started/completed/failed` 和 `request_id`。若 `superseded_pending`
增长，说明生产频率高于 provider；若 `completed` 增长但 `select` 全是 hold，检查
`hold_reason`（通常是 `policy_failure`、`deadline_exceeded`、`superseded_response` 或
`action_horizon_exhausted`）。

### 7.5 integrated supervisor 拒绝

断点：`integrated_panda_guard.py::IntegratedPandaSupervisor.supervise`。按
`failure_stage` 分支：

| 阶段 | 先看什么 | 常见根因 |
| --- | --- | --- |
| `deadline` | `response_age_ms`, `total_latency_ms` | 预算过小或完整 horizon 检查为秒级 |
| `state_alignment` | `state_mismatch_rad` | `observed_q` 与实际 q 偏差超过阈值 |
| `qp_projection` | `QPProjectionResult.failure_step/reason`, solver status | 速度/加速度/限位盒不相容、OSQP 超预算、碰撞 fallback 失败 |
| `continuous_collision` | certificate `status/reason/collision_pair`, `maximum_depth_reached` | 真实碰撞或保守 indeterminate；后者必须 fail-closed |
| `dynamics_braking` | `DynamicsBrakingResult.failure_reason/failure_sample_index/max_torque_ratio` | 停止时间、关节限位、连续边或逆动力学力矩不可行 |
| `supervision_budget` | 各 stage latency | CPU 完整检查超过 declared budget |

拒绝时应看到 `policy_actions_executable=False` 且动作数组长度为 0；如果出现非空
部分动作，优先怀疑错误地绕过了 `IntegratedPandaDecision`。

若 supervisor 已返回 accepted，但 gate 最终 hold，继续看
`reset_generation_mismatch`、`replayed_or_out_of_order_assurance_result`、
`response_deadline_exceeded_before_activation` 或
`state_changed_during_assurance`。这是激活时二次校验，不是重复的 supervisor 故障。

### 7.6 MuJoCo 执行异常

断点：`mujoco_sim/execution.py::execute_trajectory`。检查：

- `control_dt / model.opt.timestep` 是否整数；
- `delay_ms` 是否为控制周期整数倍；
- `requested` 与 `applied` 的力矩差是否导致 saturation；
- `obstacle_contacts`、`self_contacts`、`q` limit 计数；
- `actual_positions` 是否和 `desired_positions` 同长度、无 NaN。

视频卡顿只说明渲染/编码或 viewer 播放问题；先用 NPZ 数字 trace 和 validator 判断
执行是否正确。

## 8. 源码哈希保护：为什么不能随便补注释

正式 artifact 把实现源码的 SHA-256 写进 `summary.json`，validator 会重新计算并拒绝
不一致版本。因此，即使只是添加注释或 docstring，也会改变文件字节和 hash。当前
正式报告保护的文件如下。

### 8.1 27 案例集成故障矩阵

对应 `reports/integrated_panda_fault_matrix_001/summary.json`：

- `src/armbench/vla/integrated_panda_matrix.py`
- `src/armbench/vla/integrated_panda_guard.py`
- `src/armbench/vla/qp_projection.py`
- `src/armbench/mujoco_sim/continuous_collision.py`
- `src/armbench/mujoco_sim/dynamics_braking.py`

### 8.2 MuJoCo 任务 artifact

对应 `reports/integrated_panda_task_001/summary.json`，除上列文件外还保护：

- `src/armbench/vla/integrated_panda_task.py`
- `src/armbench/mujoco_sim/execution.py`
- `src/armbench/planners/rrt_connect.py`
- `src/armbench/postprocess/shortcut.py`
- `src/armbench/postprocess/time_parameterization.py`

### 8.3 17 案例异步集成发布 artifact

`reports/cpu_runtime_completion_001/provenance.json` 保护 provider、adapter、policy
worker、异步 supervisor worker、集成保障链、QP、连续碰撞和制动共 10 个实现文件。
这份报告是在新增注释和运行时逻辑完成后重新生成的；后续修改任一受保护文件都必须
使用新 run ID 生成新报告。

独立的 swept、自碰撞、动力学审计和 provider/LeRobot artifact 也可能在自己的
`summary.json` 或 manifest 中记录源码/文件哈希；修改前先搜索
`implementation_sha256`，不要假设只有上面两份报告受保护。

若必须修改受保护源码：

1. 先复制旧报告或保留旧 run ID，不覆盖原目录；
2. 修改后用新的 run ID 重新生成 artifact；
3. 让新 summary 记录新的 implementation hash；
4. 重新运行对应 validator 和全套测试；
5. 在结果文档中说明旧报告与新报告的版本边界。

本导读、普通文档和未被 artifact 注册的调试脚本不会改变已有报告，但也不能把
文档描述当成新的实验结果。

## 9. 最小可复现阅读顺序

第一次接手代码时，建议按以下顺序阅读并逐步执行：

1. `README_ZH.md` 和[架构文档](PROJECT_ARCHITECTURE_ZH.md)，确认两条路径和主张边界；
2. `types.py`，记住 `VLAObservation`/`ActionChunk` 的形状与时间字段；
3. `policy.py`、`async_worker.py`、`async_dispatch.py`，用 `vla-async-smoke` 验证异步
   边界；
4. `integrated_panda_guard.py`、`qp_projection.py`、`continuous_collision.py`、
   `dynamics_braking.py`，用 `vla-integrated-guard-smoke` 跟一条 accepted 和一条
   fallback；
5. `integrated_panda_task.py`、`execution.py`，用任务 validator 对照 NPZ trace；
6. 最后读 `summary.json`、`manifest.json` 和 validator，理解“结果如何被重算”，而不
   只看网页上的汇总数字。

如果断点行为与本导读冲突，以当前源码、对应 artifact 的 manifest/hash 和 validator
输出为准；网页和说明文档不能扩大实验主张。
