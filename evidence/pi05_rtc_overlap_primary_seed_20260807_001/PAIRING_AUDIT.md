# Pairing audit: rejected primary attempt

This v2 seed artifact is byte-preserved but rejected from matched-effect
analysis. Query 0 used identical sampling key/noise hashes across methods, yet
the response-action hashes differed in 15/50 triplets: every state 2-6 for tasks
3, 8, and 9.

Do not use `evaluation/summary.json` or `evaluation/summary.md` as method-effect
evidence. The raw artifact remains useful for reproducing the audit failure.

- Sampling seed: `20260807`
- Cloud/local archive SHA-256:
  `0449734c0a51ee482a5fa040a883f014c3559b541a6b05b3aa2f4b9bf620f45f`
- Detailed diagnosis:
  [`RTC_OVERLAP_PAIRING_AUDIT_20260805.md`](../../docs/research/RTC_OVERLAP_PAIRING_AUDIT_20260805.md)
