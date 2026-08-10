# 冻结 pi0.5 响应的 Panda 完整保障 CPU 回放

状态：R02 本地 CPU 预检已完成。正式 artifact 位于
[`reports/pi05_integrated_panda_cpu_replay_270_001`](../reports/pi05_integrated_panda_cpu_replay_270_001/summary.md)。

## 这一步回答什么

这一步把之前分开的两条链真正接到同一请求上：一边是经过哈希校验、由 Physical
Intelligence 官方 `pi05_libero` checkpoint 生成的冻结 `H x 7` 响应；另一边是七自由度
Panda 的 OSQP、连续静态/自碰、动力学制动和原子发布保障链。

同一响应只适配一次，三个模式共享完全相同的 `H x 8` 候选和 Panda 初始状态：

| 模式 | 执行边界 |
| --- | --- |
| `direct_dispatch` | 不做 QP、碰撞拒绝或制动修复，完整发布适配后的动作块。 |
| `qp_projection` | 用 OSQP 投影关节位置、速度和加速度约束；QP 或时序失败时 hold。 |
| `full_assurance` | 在独立线程完成 QP、连续碰撞和逐动作边界逆动力学制动，再经 `AtomicPandaPlanGate` 激活。 |

```text
已校验的冻结 pi0.5 Hx7 响应
             |
             v
LIBERO 语义门禁 + Panda 微分逆运动学 Hx8
             |
     +-------+--------+
     |       |        |
     v       v        v
   直接发布  OSQP    OSQP + 连续碰撞 + 动力学制动
     |       |        |
     +-------+--------+
             v
       完整发布或零动作 hold
             |
             v
 CSV + 候选/发布轨迹 NPZ + manifest + validator
```

本次不会下载或执行 checkpoint。runner 在选样前重新验证源目录递归 manifest、
checkpoint attestation、transition chain 和 7,934/7,934 个响应动作哈希。

## 冻结协议

协议在实现和查看结果前已提交：
[`pi05_integrated_panda_cpu_protocol_20260810.json`](research/pi05_integrated_panda_cpu_protocol_20260810.json)。
固定内容为 30 个按 task/method 分层的真实响应、3 个 Panda 场景、3 个模式、20 ms
软件预算和 200 ms 响应 deadline，共 270 行。

## 正式结果

| 模式 | 执行 / 总数 | 满足全部约束的候选 | 发布不安全计划 | 20 ms 超限 | P95 模式耗时 | 平均保留动作量 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 直接发布 | 90/90 | 0/90 | 90 | 0 | 0.002 ms | 1.000 |
| QP 投影 | 1/90 | 5/90 | 1 | 89 | 28.848 ms | 0.009 |
| 完整保障 | 0/90 | 5/90 | 0 | 90 | 42.974 ms | 0.000 |

冻结的 go/no-go 判定为 `no_go`：当前 Python CPU 实现无法在 20 ms 内发布一条经过
完整保障的响应。它按设计 fail closed，270 行中没有任何被拒绝动作前缀泄露到发布
轨迹。这一结果把下一步问题拆成两个互相独立的瓶颈：

1. **耗时瓶颈**：QP 在 89/90 个案例中超过 20 ms；完整保障为 90/90。
2. **可行性瓶颈**：即使忽略在线耗时，QP 后也只有 5/90 个候选同时通过运动学、
   连续碰撞和动力学制动谓词。

## 为什么不与上一阶段冲突

上一阶段的 `BrakingTrajectoryGuard` 是有限速度比例搜索，使用较轻量的离线边检查和
终端制动条件，P95 为 12.777 ms。当前阶段继续增加了：

- 每一步 OSQP 位置、速度和加速度投影；
- 注册的连续静态碰撞和连续自碰证书；
- 每一个动作边界的 MuJoCo 逆动力学可达停止；
- 独立保障线程与激活时 response-age/state 原子复查。

因此它们是从“局部轨迹修复”到“完整在线发布条件”的层层加严，不是同一实验得到
互相矛盾的结论。

## 本地验收

在仓库根目录的 Windows PowerShell 执行：

```powershell
& '.\.venv\Scripts\python.exe' -m armbench `
  vla-panda-integrated-replay-validate `
  reports\pi05_integrated_panda_cpu_replay_270_001 `
  --source-directory `
  evidence\pi05_rtc_overlap_primary_v3_seed_20260807_001\evaluation
```

validator 会重新计算源响应绑定、270 行成对矩阵、轨迹积分、全有或全无发布、连续碰撞、
逆动力学制动、汇总和 Markdown。测试还会修改 CSV/NPZ 并重新生成 manifest，确认重新
签名后的语义篡改仍然被拒绝。

## 下一轮 CPU 优化

下一轮不需要 GPU。优先级固定为：复用并预构建 OSQP、复用 MuJoCo `MjData` 和执行器
力矩上限、增加保守 broad phase 与证书缓存，以及在昂贵精确检查前加入能提高候选
可行性的 whole-chunk repair。优化完成后另建协议和 artifact；当前 no-go 结果不删除，
也不通过事后放宽 20 ms 阈值改写。

## 主张边界

末端位移只表示保留了多少原始命令，不代表任务进度。每个案例独立 reset，没有把
Panda 观察反馈给策略。本结果不是任务成功实验、硬实时保证、实体机器人测试、安全
认证或跨模型结论。
