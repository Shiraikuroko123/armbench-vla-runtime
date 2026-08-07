# LeRobot-style runtime bridge and command watchdog

This CPU-only bridge defines the boundary that a future LeRobot or robot-driver
integration must satisfy. It does not depend on the `lerobot` package and does
not claim compatibility with a particular on-disk `LeRobotDataset` version.

## Frame boundary

`LeRobotFrameAdapter` maps one validated ArmBench observation and dispatched
Panda command to the in-memory keys commonly passed to LeRobot `add_frame`:

```text
observation.images.exterior  uint8[224,224,3]
observation.images.wrist     uint8[224,224,3]
observation.state            float32[8]
action                       float32[8]
task                         string
```

The action semantic ID and canonical SHA-256 must match the registered Panda
joint-velocity/gripper contract. The adapter copies every array so callers
cannot mutate the runtime observation through a dataset record.

## Actuator watchdog

`ActuatorCommandWatchdog` runs after dispatch and trajectory repair, immediately
before a future actuator transport. It enforces finite 8D actions, an exact
semantic hash, strictly increasing command IDs, non-regressing observation IDs,
ordered and monotonic capture/issue/evaluation timestamps, independent
observation/action deadlines, and command heartbeat. A protocol fault latches a
zero-joint-velocity hold at the measured gripper position. Recovery requires an
explicit timestamped reset, retains replay-protection high-water marks, and
rejects commands queued before that reset.

This is software fail-closed behavior under a deterministic clock contract. It
is not a safety PLC, hardware emergency stop, hard-real-time scheduler, or robot
safety certificate.

## Episode artifact and replay

```powershell
& '..\.venv\Scripts\python.exe' -m armbench vla-lerobot-smoke `
  --output-directory reports\lerobot_style_watchdog_001

& '..\.venv\Scripts\python.exe' -m armbench `
  vla-lerobot-validate reports\lerobot_style_watchdog_001

& '..\.venv\Scripts\python.exe' -m armbench `
  vla-lerobot-replay reports\lerobot_style_watchdog_001
```

The preserved five-frame fixture exercises two normal commands, a stale
observation rejection, the resulting fault latch, an explicit reset, and a
recovered command. `episode.npz` stores images, state, requested/dispatched
actions, sequence IDs, and timestamps. `frames.jsonl` cross-binds scalar fields,
watchdog decisions, and per-field hashes. Metadata, summary, and all bytes are
covered by a root SHA-256 manifest.

Validation does more than load the files: it checks dtype/shape and global
monotonicity, reconstructs every observation, replays the watchdog state
machine including reset, regenerates every LeRobot-style frame, compares all
content hashes, and recomputes the summary. Re-signing a forged decision does
not bypass this replay.

## Remaining hardware work

A real integration still needs a pinned LeRobot release and official dataset
loader test, a concrete robot driver, calibration, command-rate enforcement in
the transport, reconnect behavior, a hardware emergency stop, and physical
fault-injection trials. None of those claims are represented by the CPU fixture.

