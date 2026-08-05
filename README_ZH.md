[English](README.md) | 简体中文

# ArmBench

面向动作块式视觉—语言—动作策略的可复现运行时评测系统。

ArmBench 研究两个部署问题：推理期间逐渐过期的动作块，以及已提交动作与
新采样动作之间的不连续性。仓库包含基于官方 pi0.5/LIBERO checkpoint 的
闭环研究，也提供本地 MuJoCo/Panda 运行时，用于协议验证、故障注入和逐动作
分析。

> 项目状态：研究原型。仓库提供官方 pi0.5 checkpoint 的已验证仿真证据，
> 但不包含 pi0.5 训练、操作系统硬实时保证、碰撞安全认证或真实机器人结果。

## 问题定义

动作块式策略根据一次观测预测多个未来控制量。如果推理消耗 d 个控制周期，
响应到达时，前 d 个动作对应的时间已经过去。仍从索引 0 开始执行，会使
机器人控制与生成动作时使用的观测发生时序错位。

ArmBench 分别评测两类互补方法：

1. 时序对齐：根据观测年龄选择动作块后缀。该方法位于策略外部，不修改
   checkpoint 权重。
2. Projected overlap 与 RTC-style guidance：进入 pi0.5 flow sampler，
   根据控制器已经承诺执行的动作约束新动作块。

两类方法采用不同的调度语义，因此作为独立方法和独立实验进行报告。

## 系统架构

~~~text
图像 + 语言指令 + 机器人状态
                 |
                 v
        pi0.5 / OpenPI 策略
                 |
                 v
              动作块
                 |
                 v
       时序对齐或 overlap guidance
                 |
                 v
 deadline、状态、运动学与碰撞检查
                 |
                 v
         LIBERO 或 MuJoCo 执行
                 |
                 v
轨迹 + 视频 + 统计分析 + 内容清单
~~~

仓库包含两条相互隔离的执行路径：

| 路径 | 用途 | 策略来源 | 执行环境 |
| --- | --- | --- | --- |
| 官方 checkpoint 评测 | 闭环方法评测 | 经过 attestation 的 pi05_libero checkpoint | Linux/NVIDIA 上的 LIBERO |
| 本地运行时验证 | 协议、保护器与故障响应 | OpenPI server 或明确标记的确定性测试夹具 | Windows/CPU 上的 MuJoCo Panda |

两条路径的动作语义和实验结果不会混合统计。

## 已实现组件

- 训练无关的固定延迟与 measured-age 动作块时序对齐。
- 固定 pi0.5 flow sampler 内的 hard projected overlap 和 RTC-style
  denoised-action VJP guidance。
- 带 checkpoint/source attestation、显式采样噪声和匹配闭环条件的
  OpenPI 评测流程。
- 严格的观测/动作契约、有界 WebSocket 推理、deadline latch、状态一致性
  检查和失败关闭监督器。
- Panda 关节、速度、加速度、夹爪与采样 mesh-edge 检查，以及动作
  backtracking。
- MuJoCo 双相机反馈、力矩控制执行，以及可复现的传感器、传输、状态和
  延迟故障注入。
- 只读验证器、内容寻址 manifest、配对统计分析和离线视频 dashboard。

## 已验证研究

下表中的每一行都是独立注册或探索性研究；不同协议的结果不合并统计。

| 研究 | 矩阵 | 主要结果 | 结论范围 |
| --- | ---: | --- | --- |
| 确定性时序对齐 | 300 rollouts / 150 pairs | 200 ms 下，异步执行 18/50，对齐执行 50/50；差异 +64 个百分点，bootstrap 95% CI [+50,+76]，Holm 校正 McNemar p=1.40e-9 | 支持确定性注入 LIBERO 延迟下的后缀对齐 |
| Measured-age 确认实验 | 240 rollouts / 120 pairs | 基线 88/120，对齐 116/120；差异 +23.33 个百分点，pair bootstrap 95% CI [+15.00,+31.67]，McNemar p=1.94e-6 | 支持配对响应抖动和策略噪声条件下的观测年龄对齐 |
| 跨 suite 验证 | 300 rollouts / 150 pairs | 描述性汇总为异步 83/150、对齐 141/150；Object、Goal、LIBERO-10 三项检验均通过 Holm 校正 | 将确定性延迟证据扩展到三个额外任务 suite |
| Corrected-v3 RTC overlap | 300 rollouts / 100 triplets | Unconditioned 96/100，hard projection 97/100，RTC 97/100；两项成功率比较的 Holm 校正 p=1.0 | 未证明任务成功率优势；motion seam 结果仅为探索性证据 |
| Hard-projection pilot | 40 rollouts / 20 pairs | Unconditioned 19/20，projected 18/20；McNemar p=1.0 | 仅支持集成与机制检查 |

完整 provenance、置信区间、运行时指标和各研究限制见
[结果记录](docs/RESULTS.md)。

### Corrected-v3 配对修复

最初的 RTC v2 比较在不同方法之间复用了同一个 LIBERO environment。
后续不变量审计发现，任务 3、8、9 在声明状态、prompt、sampling key 和
sampling noise 相同的情况下，query-zero 的策略图像和动作仍然不同。因此，
v2 artifact 仅作为审计记录保留，不进入方法效果估计。

Corrected-v3 为每个 rollout 创建独立 environment，并要求同一方法 triplet
内四类 query-zero 哈希完全一致：策略输入、响应动作、sampling key 和
sampling noise。300-rollout v3 矩阵使用 held-out seeds 完整重跑，没有
合并或修补 v2 结果。

详细信息见[配对审计](docs/research/RTC_OVERLAP_PAIRING_AUDIT_20260805.md)、
[修正协议](docs/research/RTC_OVERLAP_PRIMARY_300_V3_PROTOCOL.md)和
[中文验收指南](docs/RTC_OVERLAP_PRIMARY_V3_ACCEPTANCE_ZH.md)。

## 验收已保存的结果

以下流程验证保存的 artifact 并重新生成本地 dashboard，不会重新执行策略
推理，因此不需要 GPU。

在 Windows 仓库根目录运行：

~~~powershell
.\scripts\rtc_primary_acceptance.cmd
.\scripts\measured_age_confirmatory_acceptance.cmd
& '..\.venv\Scripts\python.exe' -m integrations.openpi.acceptance_dashboard --open
& '..\.venv\Scripts\python.exe' -m integrations.openpi.cross_suite_dashboard --open
~~~

两个命令脚本都支持 -NoOpen，用于只验证而不打开浏览器。有效的 RTC 验收
输出包含：

~~~text
valid=true
rollouts=300
triplets=100
failure_videos_verified=10
tasks=10
~~~

生成的页面保存在 reports/，可以完全离线查看。

## 本地安装

已验证的 Windows 环境使用 Python 3.10.8。虚拟环境和 MuJoCo Menagerie
位于仓库同级的工作区目录。

~~~powershell
$Workspace = 'D:\arm-planning-control-project'
Set-Location $Workspace

git clone https://github.com/Shiraikuroko123/armbench-vla-runtime.git project
git clone --filter=blob:none --no-checkout https://github.com/google-deepmind/mujoco_menagerie.git upstream\mujoco_menagerie
git -C upstream\mujoco_menagerie sparse-checkout init --cone
git -C upstream\mujoco_menagerie sparse-checkout set franka_emika_panda
git -C upstream\mujoco_menagerie checkout 71f066ad0be9cd271f7ed58c030243ef157af9f4

py -3.10 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --editable '.\project[test,vla]'
~~~

如果工作区已经存在，可以从任意 PowerShell 目录调用自定位脚本：

~~~powershell
& 'D:\arm-planning-control-project\project\scripts\vla_demo.cmd' -CheckOnly
& 'D:\arm-planning-control-project\project\scripts\vla_demo.cmd'
& 'D:\arm-planning-control-project\project\scripts\vla_demo.cmd' -Formal
~~~

| 命令 | 行为 |
| --- | --- |
| -CheckOnly | 验证依赖、场景、相机契约、协议、guard 和已保存 artifact |
| 默认 | 运行单场景本地 smoke，不生成视频 |
| -Formal | 运行双场景本地矩阵并生成三个 MP4 |

本地 reference policy 是确定性测试夹具。MuJoCo 路径只有在连接真实 OpenPI
server 并保存相应证据后，才能形成学习式策略结论。

## 执行新的官方 checkpoint 实验

重新运行 pi0.5 rollout 需要 Ubuntu、NVIDIA GPU、固定 OpenPI checkout 和
官方 checkpoint cache。容器工作流、preflight gate、预算控制与验证命令见
[OpenPI/LIBERO 操作手册](docs/OPENPI_LIBERO_OPERATIONS.md)。

查看已有结果 dashboard 不需要上述环境。

## 仓库结构

~~~text
src/armbench/                 本地规划、控制、MuJoCo 与 VLA 运行时
integrations/openpi/          官方 checkpoint evaluator 与分析工具
integrations/openpi/patches/  固定版本的 OpenPI sampler 扩展
tests/                        单元测试与集成测试
scripts/                      可复现运行与验收命令
docs/                         架构、方法、协议和操作文档
evidence/                     保存的实验 artifact
reports/                      生成的离线 dashboard
configs/                      benchmark 与场景配置
~~~

通过[文档索引](docs/README.md)可以区分当前说明、冻结协议、审计记录和
历史 runbook。

## 可复现性设计

正式 artifact 绑定以下信息：

- ArmBench 与 OpenPI commit；
- checkpoint URI 和内容 SHA-256；
- 完整解析后的协议与实验矩阵；
- 协议要求的显式策略采样噪声；
- episode、query 和 action 级记录；
- 分析输入与派生输出；
- 必需视频路径及其哈希。

验证器对文件缺失、内容变化、重复键、非规范字段和矩阵外数据执行失败关闭。
这能验证 artifact 完整性和内部一致性，但不等同于上游发布者身份认证或
物理安全证书。

## 环境支持

| 能力 | 已验证环境 |
| --- | --- |
| Artifact 验证与 dashboard | Windows/CPU |
| 本地 MuJoCo 运行时 | Windows，CPU + OpenGL；不要求 CUDA |
| OpenPI 客户端 | Windows，有界 WebSocket/MessagePack transport |
| 官方 pi0.5 推理 | 独立 Ubuntu/NVIDIA 主机；根据上游说明需要超过 8 GB VRAM |
| Isaac Gym / Isaac Lab | 未集成 |
| Franka Panda 真机 | 未集成；没有 ROS2、libfranka、标定、watchdog 或安全 PLC adapter |

支持其他机器人本体需要重新实现观测转换、动作语义、运动学限制以及仿真器或
硬件 adapter。

## 验证与调试

在仓库根目录运行完整测试：

~~~powershell
& '..\.venv\Scripts\python.exe' -m pytest -q
~~~

运行时问题请按照边界顺序使用
[故障排查指南](docs/DEBUGGING.md)。

## 适用范围

- 官方 checkpoint 结果仅覆盖仿真中的一个 pi0.5-LIBERO checkpoint。
- 确定性延迟与 measured-age 研究使用 blocking policy call 和响应后的
  simulator catch-up，不是独立调度的推理与控制线程。
- Motion seam 是过程指标，不是任务成功率或安全终点。
- MuJoCo 碰撞检查使用构型采样和边插值，不是解析连续碰撞检测。
- 运行时限制命令变化率，但不认证加速度、jerk 或真实物理跟踪。

## 许可证与上游软件

ArmBench 使用 MIT License。上游模型、数据集、资产和代码继续适用各自原始
许可证，详见[第三方声明](THIRD_PARTY_NOTICES.md)。
