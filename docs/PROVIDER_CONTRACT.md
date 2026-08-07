# Provider-neutral action-chunk contract

ArmBench keeps provider-native actions outside the Panda runtime until their
meaning is explicit. A provider descriptor records model-family identity,
upstream revision, checkpoint-attestation status, response origin, action
ordering, control period, coordinate frame, normalization, translation and
rotation scales, gripper polarity, and controller semantics. The descriptor has
a canonical semantic hash; matching only the number of action columns is not
accepted.

`FrozenResponseProvider` verifies `provider.json`, `responses.npz`, and a
SHA-256 manifest before returning a `RawActionChunk`. Optional observation
hashes bind each response to the exact dual-camera image, state, prompt, and
sequence ID that produced it. `AdaptedActionChunkPolicy` then applies a
fail-closed semantic gate and the existing LIBERO Cartesian-to-Panda adapter.
Only the resulting `Hx8` Panda joint-velocity/gripper chunk can enter the
asynchronous worker, dispatcher, guard, and controller.

## CPU audit

```powershell
& '..\.venv\Scripts\python.exe' -m armbench vla-provider-audit `
  --output-directory reports\provider_contract_audit_001

& '..\.venv\Scripts\python.exe' -m armbench `
  vla-provider-audit-validate reports\provider_contract_audit_001
```

The preserved audit loads an `OpenVLA-OFT`-named synthetic contract fixture,
verifies its observation/response binding, converts a `6x7` LIBERO-style chunk
to a `6x8` Panda runtime chunk, and rejects five same-width semantic mismatches.
The report and nested provider bundle are hash-manifested and replayed by the
validator.

## Claim boundary

The fixture exercises a second model-family ABI; it is not an OpenVLA-OFT
checkpoint response. The OpenVLA-OFT checkpoint was not loaded or executed,
its content hash is not attested, and this audit provides no cross-model task
success, visual-language ability, GPU latency, or generalization result. Those
claims require checkpoint-backed captures followed by a preregistered
closed-loop comparison.

