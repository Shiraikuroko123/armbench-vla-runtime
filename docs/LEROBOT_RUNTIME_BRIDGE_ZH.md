# LeRobot 风格运行时桥接与命令 watchdog

这个 CPU-only 模块定义了未来接入 LeRobot 或机器人驱动时必须满足的边界。当前不
依赖 `lerobot` 包，也不声称兼容某一个版本的 `LeRobotDataset` 磁盘格式。

## 帧接口

`LeRobotFrameAdapter` 将一条 ArmBench 观测和经过 runtime 处理的 Panda 命令映射
为 LeRobot `add_frame` 常用的内存字段：

```text
observation.images.exterior  uint8[224,224,3]
observation.images.wrist     uint8[224,224,3]
observation.state            float32[8]
action                       float32[8]
task                         string
```

动作语义 ID 和规范化 SHA-256 必须与注册的 Panda 关节速度/夹爪契约完全一致。
adapter 会复制所有数组，数据记录的调用方无法反向修改 runtime 观测。

## 执行器 watchdog

`ActuatorCommandWatchdog` 位于 dispatcher 和轨迹修复之后、未来的执行器 transport
之前。它检查 8 维有限动作、动作语义哈希、严格递增的命令序号、不可回退的观测
序号、capture/issue/evaluate 时间包络与各自单调性、独立的观测/动作 deadline 和
命令 heartbeat。任意协议故障都会锁存为“七关节零速度、保持当前夹爪位置”的 hold。
恢复必须显式 reset；reset 保留防重放高水位，并拒绝 reset 前已排队的旧命令。

这是确定性时钟契约下的软件 fail-closed 逻辑，不是安全 PLC、硬件急停、硬实时
调度或机器人安全认证。

## Episode 与离线重放

```powershell
& '..\.venv\Scripts\python.exe' -m armbench vla-lerobot-smoke `
  --output-directory reports\lerobot_style_watchdog_001

& '..\.venv\Scripts\python.exe' -m armbench `
  vla-lerobot-validate reports\lerobot_style_watchdog_001

& '..\.venv\Scripts\python.exe' -m armbench `
  vla-lerobot-replay reports\lerobot_style_watchdog_001
```

保留的 5 帧 fixture 覆盖两条正常命令、一次过期观测拒绝、随后锁存、显式 reset 和
恢复执行。`episode.npz` 保存图像、状态、请求/下发动作、序列号与时间戳；
`frames.jsonl` 交叉绑定标量字段、watchdog 决策和逐字段哈希；metadata、summary
及所有文件均由根 SHA-256 清单保护。

validator 不只是确认文件可读：它检查 dtype/shape 和全局单调性，重建每条观测，
连同 reset 一起重放 watchdog 状态机，重新生成 LeRobot 风格帧、核对所有内容哈希，
并重算 summary。因此，即使篡改者重新生成 manifest，伪造的 decision 仍会被拒绝。

## 尚未完成的真机部分

真正接入仍需要固定 LeRobot 版本并通过官方 loader、具体机器人驱动、标定、transport
层命令频率约束、断线重连、硬件急停以及实体故障注入。CPU fixture 不支持这些主张。

