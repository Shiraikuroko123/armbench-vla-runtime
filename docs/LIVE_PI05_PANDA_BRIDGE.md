# Live pi0.5-LIBERO to Panda bridge

This path connects an attested remote OpenPI `pi05_libero` checkpoint to the
local MuJoCo Panda runtime. The provider's native Hx7 Cartesian action is
validated and explicitly adapted to the runtime's Hx8 joint-velocity/gripper
contract before it reaches the deadline-aware guard:

```text
OpenPI WebSocket
  -> attestation + native Hx7 response
  -> LIBERO action-semantics gate
  -> PandaCartesianActionAdapter (differential IK and limits)
  -> latest-only policy worker
  -> deadline-aware guard / braking repair
  -> torque-controlled MuJoCo Panda trace
```

The integration smoke is intentionally narrower than a task benchmark. It
records the server attestation, checkpoint content hash, response SHA-256,
provider and adapter timing, worker scheduling, and the complete Panda trace.
It does not claim a physical robot, hard real-time scheduling, or official
LIBERO task-success performance.

## Run

On the GPU host, after starting the attested server on port 8000:

```bash
export PYTHONPATH=/workspace/armbench/project/src
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

/workspace/armbench-venv/bin/python \
  scripts/run_live_pi05_panda_smoke.py \
  --host 127.0.0.1 --port 8000 \
  --scenario free_space --mode braking_invariant \
  --steps 12 --extra-steps 6 \
  --output-directory /workspace/armbench-results/g01_live_panda_smoke_001 \
  --video
```

The output contains `summary.json`, `events.json`, `trace.npz`, an optional
`panda_trace.mp4`, and a SHA-256 `manifest.json`. A valid live smoke must show
`response_origin=live_checkpoint_inference`, `scripted_policy=false`,
`policy_checkpoint_executed=true`, and a source string containing the provider
ID and response digest. This is evidence that the learned checkpoint entered
the Panda simulation boundary; it is not evidence of a real-robot deployment
or a complete independent-clock LIBERO study.
