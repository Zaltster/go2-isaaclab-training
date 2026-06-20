# Go2 IsaacLab Restart

This directory was reset from the MJLab experiment to a small IsaacLab baseline.

Current goal:

1. Train from IsaacLab's built-in Unitree Go2 locomotion stack.
2. Add a small custom task with the real robot's sensor layout.
3. Measure SPS against the previous `~120k` IsaacLab run.

Main file:

- `train_wallclock.py`
- `source/go2_isaaclab_tasks/go2_isaaclab_tasks/go2_lidar_env_cfg.py`
- `source/go2_isaaclab_tasks/go2_isaaclab_tasks/mdp.py`

Important environment variables:

- `GO2_TASK`: IsaacLab task id. Default is `Go2-Lidar-Walk-Flat-v0`.
- `GO2_NUM_ENVS`: Spark container default `1024`.
- `GO2_MAX_ITERATIONS`: Spark container default `100000`.
- `GO2_HEADLESS`: default `1`.
- `GO2_EXPERIMENT_NAME`: Spark container default `go2_lidar_walk_flat_analytic`.
- `GO2_RECORD_VIDEO`: default `1` in the Spark container.
- `GO2_VIDEO_INTERVAL`: optional IsaacLab video interval override. If unset, the launcher approximates one video every 30 minutes on Spark from the configured SPS estimate.
- `GO2_VIDEO_LENGTH`: default `300` policy steps per recorded clip.
- `GO2_VIDEO_WALLCLOCK_SECONDS`: default `1800`, used to compute the video interval.
- `GO2_SAVE_INTERVAL_ITERATIONS`: Spark container default `100`; optional exact RSL-RL checkpoint save interval.
- `GO2_CHECKPOINT_SECONDS`: default `1800`, used to compute checkpoint interval when `GO2_SAVE_INTERVAL_ITERATIONS` is unset.
- `GO2_CHECKPOINT_MAX_STEPS`: default `1000000000`; checkpoint interval targets the smaller of this and `GO2_CHECKPOINT_SECONDS * GO2_CHECKPOINT_SPS`.
- `GO2_CHECKPOINT_SPS`: default `70000`, the Spark throughput estimate used for checkpoint interval computation.
- `GO2_OTEL_ENABLED`: default `1` when `OTEL_EXPORTER_OTLP_ENDPOINT` or `GO2_OTEL_CONSOLE=1` is set, otherwise `0`.
- `GO2_OTEL_CONSOLE`: container default `0`; set to `1` to print OTEL spans/metrics to stdout for Wendy log capture when no collector endpoint is configured.
- `GO2_OTEL_HEARTBEAT_SECONDS`: default `30`; emits training heartbeat telemetry while the IsaacLab subprocess is still running.
- `GO2_OTEL_METRIC_EXPORT_INTERVAL_MS`: container default `30000`.
- `OTEL_EXPORTER_OTLP_ENDPOINT`: OTLP/HTTP collector endpoint, for example `http://collector-host:4318`.
- `OTEL_SERVICE_NAME`: default `go2-isaaclab-training`.
- `GO2_TRAIN_EXTRA_ARGS`: appended directly to IsaacLab's RSL-RL train command.
- `ISAACLAB_ROOT`: default `/workspace/IsaacLab`.

Custom task:

- `Go2-Lidar-Walk-Flat-v0`
- `Go2-Lidar-Walk-Flat-Play-v0`

Current custom sensor layout:

- top LiDAR attached to `{ENV_REGEX_NS}/Robot/base` at `(0.18, 0.0, 0.26)`
- front depth/LiDAR fan attached to `{ENV_REGEX_NS}/Robot/base` at `(0.31, 0.0, 0.08)`

The custom task currently adds fast analytic LiDAR-style range observations on
top of the stock flat Go2 locomotion task. The analytic sensor reads randomized
obstacle positions and produces normalized range bins for training, while the
simulated obstacle collision bodies remain in the world for contact penalties
and video inspection. This avoids the mesh ray-caster target path during long
training runs.

Current custom reward changes:

- keeps IsaacLab's default Go2 walking rewards and fall termination
- rewards forward progress toward a lane goal and gives a goal-zone bonus
- commands forward walking with yaw freedom for obstacle avoidance
- penalizes close forward LiDAR returns and body/obstacle contact
- weakly penalizes sideways velocity and lane drift
- terminates on goal reach, base fall, timeout, or sustained no-progress blocking

Current obstacle layout:

- randomized cuboid obstacles per environment
- LiDAR-like observations are computed analytically from the obstacle centers

Training checkpoints are written under `logs/` inside the run directory. Training
video MP4 clips are also written under `logs/`.
On Spark, `/logs` is a Wendy persistent volume and the launcher links workspace
`logs/` to that mount so generated clips and `model_*.pt` checkpoints survive
the container working directory. The Spark container saves checkpoints every 100
PPO iterations by default while the run is being stabilized. If
`GO2_SAVE_INTERVAL_ITERATIONS` is unset, the fallback schedule targets roughly 30
minutes at the configured `GO2_CHECKPOINT_SPS` estimate and is earlier than 1B env-steps. The default run length
is 100000 iterations so a normal run can continue well past 12 hours.

OpenTelemetry:

- the launcher emits a `go2.training.launch` span around the IsaacLab process
- heartbeat telemetry is emitted every 30 seconds by default
- run count, exit count, running-process count, and duration metrics are emitted
- OTEL is optional; the Spark training image does not install OTEL packages by default, and if no collector endpoint is configured, training runs normally without export
