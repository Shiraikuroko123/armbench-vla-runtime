# Corrected pi0.5 RTC overlap primary evidence: seed 20260807

This is one of the two required raw artifacts for the corrected held-out
RTC-overlap study. It was produced by evaluator commit
`44c358731c5493284b74bb29eefa7d538d0f38dd` and must be interpreted only with
the independently preserved seed `20260806` artifact.

- Runtime schema: `armbench.pi05_rtc_overlap_pilot.v3`
- Matrix: 10 LIBERO-10 tasks x 5 initial states x 3 methods
- Evidence: 150 rollouts and 50 complete matched triplets
- Pairing gate: shared query-0 policy input, response action, sampling key, and
  sampling noise SHA-256 values within every method triplet
- Frozen launch: environment seed `7`, failure-only video capture, and no
  `--max-task-steps` override
- Cloud/local archive SHA-256:
  `a04b6c66477ebe3d0db184beccb10d8d9022c036ac72202b5632075a8bdf60bd`
- Video validation: 4 H.264 failure videos, `224x224` at 10 fps, 2,080 decoded
  frames

The raw artifact passed the v3 root-manifest, transition-archive, reference
chain, sampler-conditioning, and query-pairing validator on both the cloud host
and the local checkout. The per-seed runtime summary is not a held-out method
effect estimate; the two corrected seeds must be combined by
`integrations.openpi.rtc_overlap_primary_analysis`.

See the frozen
[`RTC_OVERLAP_PRIMARY_300_V3_PROTOCOL.md`](../../docs/research/RTC_OVERLAP_PRIMARY_300_V3_PROTOCOL.md)
and the rejected-v2
[`RTC_OVERLAP_PAIRING_AUDIT_20260805.md`](../../docs/research/RTC_OVERLAP_PAIRING_AUDIT_20260805.md).
