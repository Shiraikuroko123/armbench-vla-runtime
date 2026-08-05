# pi0.5 RTC overlap v3 pairing smoke

This artifact is an implementation gate, not outcome evidence. It covers tasks
3, 8, and 9 at initial-state index 2 with all three overlap methods (9 rollouts,
3 triplets) using evaluator commit
`44c358731c5493284b74bb29eefa7d538d0f38dd`.

The v3 validator verified that each query-0 triplet has exactly one shared
canonical policy-input hash, response-action hash, sampling-key hash, and
sampling-noise hash. This directly exercises the cells that failed the v2
pairing audit.

- Runtime schema: `armbench.pi05_rtc_overlap_pilot.v3`
- Sampling seed: `20260808`
- Cloud/local archive SHA-256:
  `ec9c386c535180a625ac5c74cac0c349c32a144a65fbf926ef9e6dfecfb9a991`
- Role: preflight only; excluded from the corrected 300-rollout analysis
- Protocol:
  [`RTC_OVERLAP_PRIMARY_300_V3_PROTOCOL.md`](../../docs/research/RTC_OVERLAP_PRIMARY_300_V3_PROTOCOL.md)
