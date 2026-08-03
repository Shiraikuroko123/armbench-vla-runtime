# OpenPI wire fault matrix

This matrix uses an ephemeral scripted non-learned server. It sends real DROID requests through the official OpenPI MessagePack/WebSocket client; it does not run pi0/pi0.5 inference.

- Matrix passed: `true`
- Injected faults that failed closed: `4/4`
- Total physical safety violation steps: `0`

| Fault | Requests | Valid chunks | Fallbacks | Client failure | Safe | Violation steps | Hashes match | Case passed |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| none | 1 | 1 | 0 | - | True | 0 | True | True |
| malformed_shape | 1 | 0 | 1 | ValueError | True | 0 | True | True |
| nonfinite | 1 | 0 | 1 | ValueError | True | 0 | True | True |
| disconnect | 1 | 0 | 1 | ConnectionClosedError | True | 0 | True | True |
| timeout | 1 | 0 | 1 | TimeoutError | True | 0 | True | True |

`case_passed` requires one server-audited request, matching camera hashes, zero physical safety violations, and the expected response authority. Fault cases must have zero validated remote chunks and one policy-inference runtime fallback.

These are deterministic fault injections, not estimates of server availability or certified safety guarantees.
