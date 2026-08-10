# pi0.5-LIBERO independent-clock deadline report

Every source artifact passed the independent validator before this report was built.
Tick-level deadline holds and response-level deadline rejections remain separate metrics.

| Suite | Seed | Deadline | Task success (Wilson 95% CI) | Execute duty | Deadline hold ticks | Response rejections | Provider failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| libero_object | 7 | 150 ms | 0/40 (0.0%, 0.0%-8.8%) | 26.9% | 8,067 | 9 | 0 |
| libero_object | 7 | 175 ms | 37/40 (92.5%, 80.1%-97.4%) | 88.4% | 624 | 0 | 0 |
| libero_object | 7 | 200 ms | 39/40 (97.5%, 87.1%-99.6%) | 88.3% | 613 | 0 | 0 |
| libero_object | 8 | 150 ms | 0/40 (0.0%, 0.0%-8.8%) | 26.4% | 8,122 | 1 | 0 |
| libero_object | 8 | 175 ms | 39/40 (97.5%, 87.1%-99.6%) | 86.6% | 714 | 1 | 0 |
| libero_object | 8 | 200 ms | 39/40 (97.5%, 87.1%-99.6%) | 88.2% | 593 | 0 | 0 |
| libero_spatial | 7 | 150 ms | 0/40 (0.0%, 0.0%-8.8%) | 26.2% | 6,371 | 16 | 0 |
| libero_spatial | 7 | 155 ms | 38/40 (95.0%, 83.5%-98.6%) | 86.8% | 489 | 2 | 0 |
| libero_spatial | 7 | 175 ms | 38/40 (95.0%, 83.5%-98.6%) | 86.5% | 495 | 0 | 0 |
| libero_spatial | 7 | 200 ms | 38/40 (95.0%, 83.5%-98.6%) | 87.2% | 472 | 0 | 0 |
| libero_spatial | 8 | 150 ms | 0/40 (0.0%, 0.0%-8.8%) | 26.6% | 6,338 | 2 | 0 |
| libero_spatial | 8 | 155 ms | 36/40 (90.0%, 76.9%-96.0%) | 87.3% | 492 | 2 | 0 |
| libero_spatial | 8 | 175 ms | 37/40 (92.5%, 80.1%-97.4%) | 86.7% | 501 | 0 | 0 |
| libero_spatial | 8 | 200 ms | 39/40 (97.5%, 87.1%-99.6%) | 86.8% | 469 | 0 | 0 |
| libero_spatial | 9 | 150 ms | 0/40 (0.0%, 0.0%-8.8%) | 26.5% | 6,347 | 5 | 0 |
| libero_spatial | 9 | 155 ms | 40/40 (100.0%, 91.2%-100.0%) | 85.6% | 532 | 2 | 0 |
| libero_spatial | 9 | 175 ms | 39/40 (97.5%, 87.1%-99.6%) | 86.8% | 486 | 0 | 0 |
| libero_spatial | 9 | 200 ms | 38/40 (95.0%, 83.5%-98.6%) | 86.9% | 503 | 0 | 0 |

## Adjacent registered deadlines

| Suite | Seed | Comparison | Success-rate difference | Execute-duty difference | Deadline-hold difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| libero_object | 7 | 150 -> 175 ms | +92.5% | +61.5% | -7,443 |
| libero_object | 7 | 175 -> 200 ms | +5.0% | -0.1% | -11 |
| libero_object | 8 | 150 -> 175 ms | +97.5% | +60.2% | -7,408 |
| libero_object | 8 | 175 -> 200 ms | +0.0% | +1.6% | -121 |
| libero_spatial | 7 | 150 -> 155 ms | +95.0% | +60.6% | -5,882 |
| libero_spatial | 7 | 155 -> 175 ms | +0.0% | -0.3% | +6 |
| libero_spatial | 7 | 175 -> 200 ms | +0.0% | +0.7% | -23 |
| libero_spatial | 8 | 150 -> 155 ms | +90.0% | +60.7% | -5,846 |
| libero_spatial | 8 | 155 -> 175 ms | +2.5% | -0.6% | +9 |
| libero_spatial | 8 | 175 -> 200 ms | +5.0% | +0.0% | -32 |
| libero_spatial | 9 | 150 -> 155 ms | +100.0% | +59.1% | -5,815 |
| libero_spatial | 9 | 155 -> 175 ms | -2.5% | +1.2% | -46 |
| libero_spatial | 9 | 175 -> 200 ms | -2.5% | +0.1% | +17 |

## Statistical boundary

Registered benchmark episodes are reported by suite, seed, and task cluster. They are not pooled as iid deployment draws or a universal VLA deadline estimate.
Wilson intervals describe the registered episode cells and do not correct for task clustering.

## Source artifacts

- `pi05_object_deadline150_seed7_40_20260810_001`: manifest `19a9060e779ae6042096e53002d96df7a0eacf2f7a38f7fd8ad89e3427cf22c5`
- `pi05_object_deadline175_seed7_40_20260810_001`: manifest `fe1b5b655f7f37c1d864cecbdf66b5cc30ae2d62d2d1df08742bdfc2a6d99b5b`
- `g03_independent_clock_object_40_20260810_001`: manifest `e9481fdefa179cbc5d9d36d010a98ccf005b7893f1a1943ea8db609a45f604c0`
- `pi05_object_deadline150_seed8_40_20260810_001`: manifest `7656edc083b9920451676ef08514916dc2aece41bbcd2ab8c72e9ab37406522d`
- `pi05_object_deadline175_seed8_40_20260810_001`: manifest `9cbfc9a5253611e38e4ec415dc98272c31fcb5b43c8802cea3a1890f418b3320`
- `pi05_object_deadline200_seed8_40_20260810_001`: manifest `f1d1bf874414280dfa1bcfdfce14932edece12ba75291b7248816c6e091fe77e`
- `g05_spatial_deadline150_40_20260810_001`: manifest `8a8eb72341c5f0cccfb7af3ecf2f66b405a273174efa4969691af978f211ae51`
- `pi05_spatial_deadline155_seed7_40_20260810_001`: manifest `3d24e3a122fc3ed66991bb4d6c4b7fb26f4320e40cba3d4a3324de208d57bea7`
- `g06_spatial_deadline175_40_20260810_001`: manifest `d6d15f8cd64733da7a63b00890fdbc6ef6bb859b9e86542c7db841fec5f03aac`
- `pi05_libero_independent_clock_core_40_001`: manifest `8346840626373428cd00723aff248567264ebc164c2df48ea620a248fe6111d9`
- `pi05_libero_spatial_deadline150_seed8_40_20260810_001`: manifest `345b54aa4a99d1ffa96f913f5f7b8eed17f0f4ccb981d33b2fc706d5423967bf`
- `pi05_spatial_deadline155_seed8_40_20260810_001`: manifest `d84437eb5bda813c7dcf274d6755256bac98a0af7415b72686a3a76c2f4d19ef`
- `pi05_libero_spatial_deadline175_seed8_40_20260810_001`: manifest `362b69537e0c186a420523f6737b3f970ec2bc8cf09de47742a832153fd19b2b`
- `pi05_spatial_deadline200_seed8_40_20260810_001`: manifest `51802399c51b8e842288e845c6acdb7d7a1e93e634445189c4ac1a4f8ebdb499`
- `pi05_spatial_deadline150_seed9_40_20260810_001`: manifest `c158dfc3dca5d78d62d3fa9244c2cf472653aed55054e12715b436b47b78744b`
- `pi05_spatial_deadline155_seed9_40_20260810_001`: manifest `9d06d8a322933f27112636244e2b4db8d15790613c430766afd34b38d1b9e070`
- `pi05_spatial_deadline175_seed9_40_20260810_001`: manifest `c1dd5243e8521f87f221af3193d78af573d5d0a985b141329eceeefde82d5349`
- `pi05_spatial_deadline200_seed9_40_20260810_001`: manifest `350f50fdad437fd82c4047863d7812928c6f45be128d51c6741a441852884a03`

## Claim boundaries

- not an official LIBERO leaderboard score
- not a hard-real-time guarantee
- not hardware safety or real-robot deployment evidence
- not cross-model superiority
