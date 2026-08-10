# pi0.5 independent-clock selection smoke: response-relative chunk

This one-rollout mechanism gate was run before the registered 240-rollout
action-selection matrix. It uses LIBERO Spatial task 0, held-out episode 4,
joint seed 7, a 175 ms deadline, and the `response_relative_chunk` selector.

- Completed / task success: 1 / 1
- Execute / hold ticks: 87 / 11
- Inference/simulation overlap: yes
- Provider failures: 0
- Worker stopped cleanly: yes

The paired report in
[`reports/pi05_selection_smoke_report_20260810_001`](../../reports/pi05_selection_smoke_report_20260810_001)
verifies equality of the initial state and query-0 policy input, sampling key,
sampling noise, and action chunk hashes against the age-aligned smoke. The
verified transport archive SHA-256 is in `transport_sha256.txt`.

This is a pairing and integration gate, not method-effect evidence.
