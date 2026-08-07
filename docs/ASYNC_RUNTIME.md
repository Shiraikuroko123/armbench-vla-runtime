# Non-blocking runtime harness

Status: Current component-level validation

The completed `pi0.5`-LIBERO studies use blocking inference followed by
simulator catch-up. The maintained runtime now includes a separate development
harness for removing that architectural limitation without requiring a GPU.

## Runtime structure

```text
observation capture
       |
       v
latest-only request mailbox ---> blocking policy worker
                                      |
control tick loop <--- outcome mailbox+
       |
       v
measured-age suffix selection -> execute or fail-closed hold
```

`LatestPolicyWorker` permits one in-flight policy call and retains at most one
pending observation. A newer observation replaces only the pending request;
Python cannot cancel an inference call already in progress. Completed outcomes
are bounded and drained without waiting from the control side.

`AsyncChunkDispatcher` uses the observation capture timestamp at every control
tick. It rejects policy failures, superseded responses, deadline misses, and
exhausted action horizons. A newer policy failure clears the active chunk and
enters hold rather than continuing an older command sequence.

## Local acceptance

```powershell
& '.\.venv\Scripts\python.exe' -m armbench vla-async-smoke
```

The JSON report records worker and control thread IDs, policy latency, control
ticks observed while inference was blocked, maximum observed tick gap, suffix
offset, hold count, and final dispatch decision. A deadline-failure path can be
checked explicitly:

```powershell
& '.\.venv\Scripts\python.exe' -m armbench vla-async-smoke `
  --policy-latency-ms 240 --deadline-ms 200
```

The second command should still pass the harness while reporting a rejected
response and a `deadline_exceeded` hold.

## Evidence boundary

This harness validates threading, mailbox, ordering, and deadline state-machine
behavior with a delayed scripted policy. It does not run OpenPI, `pi0.5`,
LIBERO, or MuJoCo; it produces no task-success evidence. It also does not set
real-time thread priorities and cannot provide a worst-case scheduling bound.

The next integration step is to place this worker/dispatcher between the
official policy client and a continuously advancing simulator, then pass the
selected suffix through the existing kinematic guard before actuator dispatch.
