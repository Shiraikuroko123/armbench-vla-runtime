# Frozen pi0.5-LIBERO latency-alignment analysis

This is a read-only intention-to-treat analysis of the manifest-bound `evaluation/per_episode.csv`.

- Source SHA-256: `d9c14a651dcfdeb90eb50b5997793808d67aa23f7d8672a5a4fe6155559ead23`
- Rollouts/pairs: `300/150`
- Runtime failures retained in ITT: `0`
- Difference direction: `latency_aligned - async_unguarded`
- Primary comparison: delay `4`; delays `0` and `2` are prespecified secondary
- Paired bootstrap seed/resamples: `20260804/10000`
- Confirmatory decisions: Holm-adjusted exact McNemar; bootstrap intervals are descriptive marginal uncertainty and need not agree with test decisions
- Task x latency rows: descriptive only; no task-level significance tests

| Delay | Role | Async success (95% Wilson) | Aligned success (95% Wilson) | Aligned - async (bootstrap 95%) | Wins/losses/ties | McNemar raw/Holm | Mean queries async/aligned/(aligned - async) | Runtime failures async/aligned |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | prespecified_secondary | 49/50 (0.980, [0.895, 0.996]) | 50/50 (1.000, [0.929, 1.000]) | 0.020 ([0.000, 0.060]) | 1/0/49 | 1/1 | 22.14/21.50/-0.64 | 0/0 |
| 2 | prespecified_secondary | 41/50 (0.820, [0.692, 0.902]) | 48/50 (0.960, [0.865, 0.989]) | 0.140 ([0.020, 0.260]) | 9/2/39 | 0.06543/0.1309 | 21.80/16.18/-5.62 | 0/0 |
| 4 | primary | 18/50 (0.360, [0.241, 0.499]) | 50/50 (1.000, [0.929, 1.000]) | 0.640 ([0.500, 0.760]) | 32/0/18 | 4.657e-10/1.397e-09 | 22.96/12.82/-10.14 | 0/0 |

Training-free temporal alignment under deterministic injected LIBERO delay; not a real-time, safety, dynamics, or real-robot guarantee.
