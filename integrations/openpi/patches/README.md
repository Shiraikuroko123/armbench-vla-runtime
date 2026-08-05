# OpenPI extension patches

These patches are based on immutable official OpenPI commits. Apply each patch
only to the base recorded in its adjacent JSON manifest. The committed
extension hash is the identity used by ArmBench attestation; a locally applied
patch with a different commit hash must not claim that identity.

For projected pi0.5 conditioning:

```bash
git checkout 15a9616a00943ada6c20a0f158e3adb39df2ccac
git am /path/to/0001-Add-projected-pi0-action-conditioning.patch
git diff 15a9616a00943ada6c20a0f158e3adb39df2ccac --check
```

The patch adds environment-to-model action transformation in `Policy.infer`
and hard projected flow inpainting in `Pi0.sample_actions`. It does not
implement RTC pseudoinverse guidance and must not be labeled as RTC.

For the RTC VJP extension on top of projected conditioning:

```bash
git checkout 2c8e61d5fbfde4b670ae428ef4b8440d35c1c7fd
git am /path/to/0001-Add-RTC-VJP-guidance-for-pi0-sampling.patch
git diff 2c8e61d5fbfde4b670ae428ef4b8440d35c1c7fd --check
```

This second patch keeps the legacy and hard-projection samplers intact and
adds a separately named `rtc_pseudoinverse_guidance` path. The reverse-time
formula, official RTC reference commit, source hashes, and exact extension
identity are recorded in `rtc_guidance_v2.json`.
