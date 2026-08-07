# Provider-neutral action-chunk 契约

ArmBench 不会因为两个模型都输出 7 维数组，就默认它们的动作含义相同。provider
描述文件明确记录模型族、上游版本、checkpoint 身份状态、响应来源、动作顺序、控制
周期、坐标系、归一化范围、平移/旋转尺度、夹爪极性和控制器语义，并为这些字段生成
规范化哈希。任意一项不一致都会在动作进入 Panda runtime 前 fail closed。

`FrozenResponseProvider` 先校验 `provider.json`、`responses.npz` 和 SHA-256
清单，再返回 provider 原生 `RawActionChunk`。可选 observation hash 将响应绑定到
准确的双相机图像、机器人状态、指令和序列号。`AdaptedActionChunkPolicy` 随后执行
语义门禁，并通过现有 LIBERO 笛卡尔到 Panda 的 adapter，只有转换后的 `Hx8`
关节速度/夹爪动作才能进入异步 worker、dispatcher、guard 和控制器。

## CPU 审计

```powershell
& '..\.venv\Scripts\python.exe' -m armbench vla-provider-audit `
  --output-directory reports\provider_contract_audit_001

& '..\.venv\Scripts\python.exe' -m armbench `
  vla-provider-audit-validate reports\provider_contract_audit_001
```

保留的审计使用一个以 `OpenVLA-OFT` 标识的合成接口 fixture：验证输入/响应绑定，
将 `6x7` LIBERO 风格动作转换为 `6x8` Panda runtime 动作，并拒绝 5 类维度相同
但语义不同的输入。根目录和嵌套 provider bundle 均有哈希清单，validator 会重新
执行适配并逐项核对。

## 主张边界

这个 fixture 只验证第二模型族的 ABI 接入，不是 OpenVLA-OFT checkpoint 输出。
本次没有加载或执行 OpenVLA-OFT，没有 checkpoint 内容哈希，也没有跨模型任务成功
率、视觉语言能力、GPU 时延或泛化结论。获得真实 checkpoint 响应并完成预注册闭环
对比后，才能支持这些主张。

