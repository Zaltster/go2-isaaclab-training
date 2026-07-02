# WTW Go2 hierarchical navigation integration issue

## Goal

Train a high-level LiDAR/navigation policy while keeping the public Walk These Ways Go2 walking policy frozen.

The intended architecture is:

- Frozen low-level WTW TorchScript modules:
  - `adaptation_module_latest.jit`
  - `body_latest.jit`
- Trainable high-level nav actor/critic only.
- Nav policy emits forward velocity and yaw commands.
- WTW policy converts those commands plus proprioception history into Go2 joint actions in IsaacLab.

## Current status

The WTW TorchScript weights themselves work on the Spark runtime.

The standalone probe app `go2-wtw-policy-probe` loaded both modules and ran CPU/CUDA inference for batch sizes 1, 8, and 32. Example observed output:

```text
torch=2.9.0+cu130 cuda_available=true cuda_device=NVIDIA GB10
adaptation_cpu_batch_32 ok shape=[32, 2]
body_cpu_batch_32 ok shape=[32, 12]
adaptation_cuda:0_batch_32 ok shape=[32, 2]
body_cuda:0_batch_32 ok shape=[32, 12]
```

The integration hang appears inside the IsaacLab adapter/training loop, not inside the public WTW weights.

## Reproduction apps

### WTW policy probe

```bash
wendy run --device spark-edeb.local --prefix go2-wtw-policy-probe --no-restart -y
```

Expected: loads WTW artifacts and prints successful adaptation/body inference timings.

### Hierarchical nav trainer

```bash
wendy run --device spark-edeb.local --prefix go2-train-smooth-wtw-hier-nav --detach --no-restart --verbose -y
```

Then probe persisted logs:

```bash
wendy run --device spark-edeb.local --prefix go2-wtw-hier-status-probe --no-restart -y
```

## Observed failure sequence

Earlier versions hung immediately when the first real WTW action was requested:

```text
update_start update=2
rollout_step_start update=2 step=0
wtw_action_start update=2 step=0
wtw_clock_start update=2 step=0
wtw_obs_start update=2 step=0
```

No `wtw_obs_done` was emitted.

That means execution entered `WtwIsaacAdapter.wtw_observation()` and did not return. The standalone TorchScript probe still passed, so the suspected issue is IsaacLab robot state access / synchronization in the adapter path, not the WTW modules.

The latest local code adds:

- global warmup before the first WTW action, matching the WTW smoke script more closely,
- single-env default for easier debugging,
- CPU WTW inference by default,
- trace markers around WTW action phases,
- an `GO2_WTW_HIER_NAV_USE_ENV_REWARD=1` path to avoid custom reward bookkeeping while debugging the adapter.

## Important constraints

- Do not train or modify the WTW walking modules.
- `trainable_wtw_params` should stay `0`.
- Checkpoints saved by the hierarchical trainer are nav-policy checkpoints only.
- The public WTW `.jit` artifacts are included only so the probe and trainer are reproducible.

## Files to inspect first

- `go2-train-smooth-wtw-hier-nav/scripts/train_wtw_hierarchical_nav.py`
- `go2-wtw-policy-probe/probe_wtw_policy.py`
- `go2-wtw-hier-status-probe/status_probe.sh`
- `go2-train-smooth-wtw-hier-nav/source/go2_isaaclab_tasks/go2_isaaclab_tasks/go2_lidar_env_cfg.py`
- `go2-train-smooth-wtw-hier-nav/source/go2_isaaclab_tasks/go2_isaaclab_tasks/mdp.py`

## Requested fix

Make the hierarchical trainer run in IsaacLab using the frozen WTW walking policy without hanging when building the WTW observation/action path. Once it advances normally, the high-level nav policy can train against LiDAR/obstacle rewards while the walking policy remains unchanged.
