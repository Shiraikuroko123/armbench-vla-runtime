# Asynchronous Panda closed-loop runtime

Status: current engineering and acceptance guide.

This runtime connects the previously separate ArmBench components into one
CPU-only closed loop:

```text
MuJoCo Panda state snapshot
        |
        +--> latest-only camera worker --> 224x224 exterior/wrist images
                                            |
                                            v
                                  blocking policy worker
                                            |
                                            v
                                  timestamped action chunk
                                            |
                                            v
                         observation-age suffix dispatcher
                                            |
                    +-----------------------+-----------------------+
                    |                       |                       |
                unguarded             legacy greedy       braking invariant
                    |                       |                       |
                    +-----------------------+-----------------------+
                                            |
                              torque-controlled MuJoCo Panda
```

The policy in this benchmark is `scripted_non_learned_async_reference`. It
blocks for a configured wall-clock delay and can inject jitter, dropped
responses, or corrupted joint velocities. It is deliberately not `pi0.5`, and
no model checkpoint is downloaded or executed.

## What changed from the offline repair

The frozen-response braking study checked action chunks without executing a
Panda feedback loop. This runtime additionally:

- advances MuJoCo from a best-effort periodic wall-clock control loop;
- captures live Panda state and two simulated policy cameras;
- keeps rendering and policy inference off the control loop with latest-only
  pending requests;
- checks the dispatcher on every control tick, not only at action boundaries;
- skips action indices whose time slots have passed;
- rechecks the deadline after repair before activating a command;
- applies PD plus bias compensation through torque-limited Panda dynamics;
- rebuilds a collision-checked, acceleration-limited stop from measured state
  when a response expires or fails; and
- records controller jitter, response age, tracking error, interventions,
  contact state, command switches, and worker identities.

The last action before a deadline may run for only the remaining wall-clock
window. ArmBench checks that action's complete joint-space edge, which is
conservative for its partial position motion. At expiry, the terminal stop is
rebuilt from the measured state and validated at the control period.

## Run locally

No GPU, OpenPI server, or physical robot is required. Complete local setup
first, then run a short nine-case matrix:

```powershell
& '.\.venv\Scripts\python.exe' -m armbench vla-panda-async-run `
  --output-directory results\async_panda_quick `
  --scenario single_block --quick --deadline-ms 400

& '.\.venv\Scripts\python.exe' -m armbench `
  vla-panda-async-validate results\async_panda_quick
```

`--quick` runs all three runtime modes at 0, 80, and 240 ms over a bounded
reference prefix. Omit it to run the full default matrix: five fixed delays,
jitter, response drops, a 0.5 kg payload, and a persistent action fault.

The repository configuration uses a 200 ms response deadline. On a machine
whose CPU camera acquisition itself takes a substantial fraction of that
budget, high hold rates are expected. `--deadline-ms` is an explicit experiment
parameter, not a hidden relaxation. The artifact records the resolved value.

Reference planning retains the configured 20 mm obstacle inflation. Runtime
checks inherit that clearance by default so the guard includes room for
closed-loop tracking error and braking distance. Pass `--runtime-clearance-mm
0` only for an explicit true-geometry ablation. Both values and the source of
the runtime clearance are recorded in provenance; MuJoCo contact remains the
separate physical-outcome metric.

## Inspect motion

Every case writes a trace that the existing viewer can replay:

```powershell
& '.\.venv\Scripts\python.exe' -m armbench mujoco-view `
  --scenario single_block `
  --trace results\async_panda_quick\traces\case_000__fixed_000ms__unguarded.npz `
  --array actual_positions --play --loop
```

Add `--videos` to the benchmark command for MP4 files. Videos are rendered from
the measured trace after execution so encoding and rendering cannot improve or
degrade the recorded control timing.

## Modes

| Mode | Behavior |
| --- | --- |
| `unguarded` | Executes the measured-age-aligned raw suffix and directly holds after expiry. |
| `legacy_greedy` | Applies the existing per-step bounds, slew, and collision backtracking, then directly holds after expiry. |
| `braking_invariant` | Repairs a bounded chunk, validates its terminal stop, and performs a fresh control-rate stop on expiry or failure. |

The modes are engineering controls, not learned-policy baselines. Wall-clock
scheduling can vary between sequential cases, so this artifact is not a formal
claim that one method has statistically superior task success.

## Artifact contract

Each run contains:

- `per_case.csv`: one row per condition/mode;
- `events.jsonl`: observation, policy, dispatch, repair, and command events;
- `traces/*.npz`: wall-clock timing, desired/actual positions, commands,
  response ages, contacts, and per-stage latency arrays;
- `summary.json` and `summary.md`: aggregates and claim boundaries;
- `provenance.json`: resolved matrix, runtime settings, package/model identity,
  implementation hashes, and limitations; and
- `manifest.json`: exact file inventory, sizes, SHA-256 hashes, and aggregate
  inventory hash.

The validator does more than verify hashes. It reloads every NPZ and recomputes
tracking error, final error, tick jitter, latency percentiles, contact counts,
and safe-success flags. It independently recounts stale action switches,
accepted/rejected responses, holds, braking commands, interventions, and
acceleration violations from `events.jsonl`. It also recounts braking-repair
selection-budget exceedances; a software budget is not reported as a hard
deadline. Re-signing a modified CSV is therefore insufficient to make a false
metric pass.

## Debugging order

1. Run `python -m armbench doctor` and fix the Panda scene or Python environment
   before investigating timing.
2. Run one mode, one fixed latency, and `--max-reference-steps 10`.
3. Check `per_case.csv` for observation latency, policy latency, deadline
   rejections, hold rate, and maximum control-tick lateness.
4. Filter `events.jsonl` by `case_id`; compare `observation_outcome`,
   `policy_outcome`, `plan_prepared`, and `command_switch` in order.
5. Replay `actual_positions` and inspect contact/command arrays in the same NPZ.
6. Run `vla-panda-async-validate` before quoting any number.

## Claim boundary

This is closed-loop Panda physics and asynchronous runtime evidence. It is not
a learned-policy task result, an official `pi0.5`-to-Panda deployment, an OS
hard-real-time guarantee, continuous collision certification, or physical
robot safety validation. The next research step is to replace the scripted
policy behind the same `ActionChunkPolicy` contract and repeat a preregistered
multi-seed matrix without changing the runtime or validator.
