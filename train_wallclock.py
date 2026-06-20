#!/usr/bin/env python3
"""Small IsaacLab training entrypoint for the Go2 restart."""

from __future__ import annotations

import os
import shlex
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_TASK_CANDIDATES = (
    "Go2-Lidar-Walk-Flat-v0",
    "Isaac-Velocity-Flat-Unitree-Go2-v0",
    "Isaac-Velocity-Rough-Unitree-Go2-v0",
    "Isaac-Velocity-Flat-Go2-v0",
    "Isaac-Velocity-Rough-Go2-v0",
)
DEFAULT_VIDEO_WALLCLOCK_SECONDS = 30.0 * 60.0
DEFAULT_SPARK_SPS = 70_000.0
DEFAULT_OTEL_HEARTBEAT_SECONDS = 30.0
DEFAULT_STALL_GRACE_SECONDS = 10 * 60
DEFAULT_STALL_TIMEOUT_SECONDS = 5 * 60


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def default_video_interval_steps(num_envs: str | int | float) -> int:
    """Approximate the requested wall-clock video cadence from measured Spark SPS."""
    try:
        env_count = max(float(num_envs), 1.0)
    except (TypeError, ValueError):
        env_count = 4096.0
    video_seconds = env_float("GO2_VIDEO_WALLCLOCK_SECONDS", DEFAULT_VIDEO_WALLCLOCK_SECONDS)
    spark_sps = env_float("GO2_VIDEO_SPS", env_float("GO2_CHECKPOINT_SPS", DEFAULT_SPARK_SPS))
    return max(1, int(round(video_seconds * spark_sps / env_count)))


def resolved_video_interval(args) -> str:
    return str(args.video_interval or default_video_interval_steps(args.num_envs))


class TrainingTelemetry:
    def __init__(self, args, command: list[str], video_interval: str | None):
        self.enabled = False
        self.provider = None
        self.meter_provider = None
        self.tracer = None
        self.run_counter = None
        self.heartbeat_counter = None
        self.duration_histogram = None
        self.exit_counter = None
        self.running_counter = None
        self.attrs = {
            "robot.model": "unitree_go2",
            "simulator": "isaaclab",
            "rl.library": "rsl_rl",
            "go2.task": str(args.task),
            "go2.num_envs": int(args.num_envs),
            "go2.max_iterations": int(args.max_iterations),
            "go2.experiment_name": str(args.experiment_name or ""),
            "go2.run_name": str(args.run_name or ""),
            "go2.record_video": bool(args.record_video),
            "go2.video_length": int(args.video_length) if str(args.video_length).isdigit() else str(args.video_length),
            "go2.video_interval": int(video_interval) if video_interval and str(video_interval).isdigit() else str(video_interval or ""),
        }
        self.command = shlex.join(command)
        self._setup()

    def _setup(self) -> None:
        otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        console_export = env_bool("GO2_OTEL_CONSOLE", False)
        if not env_bool("GO2_OTEL_ENABLED", bool(otlp_endpoint or console_export)):
            return

        try:
            from opentelemetry import metrics, trace
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
            from opentelemetry.sdk.resources import SERVICE_NAME, Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            if console_export:
                from opentelemetry.sdk.metrics.export import ConsoleMetricExporter
                from opentelemetry.sdk.trace.export import ConsoleSpanExporter

                span_exporter = ConsoleSpanExporter()
                metric_exporter = ConsoleMetricExporter()
            else:
                from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

                span_exporter = OTLPSpanExporter()
                metric_exporter = OTLPMetricExporter()

            os.environ.setdefault("OTEL_SERVICE_NAME", "go2-isaaclab-training")
            resource = Resource.create(
                {
                    SERVICE_NAME: os.environ["OTEL_SERVICE_NAME"],
                    "service.version": os.environ.get("GO2_APP_VERSION", "0.1.0"),
                    "deployment.environment": os.environ.get("GO2_DEPLOYMENT_ENV", "spark"),
                    "device.name": os.environ.get("WENDY_DEVICE", "spark-edeb.local"),
                }
            )

            self.provider = TracerProvider(resource=resource)
            self.provider.add_span_processor(BatchSpanProcessor(span_exporter))
            trace.set_tracer_provider(self.provider)
            self.tracer = trace.get_tracer("go2_isaaclab_training")

            interval_ms = int(os.environ.get("GO2_OTEL_METRIC_EXPORT_INTERVAL_MS", "10000"))
            reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=interval_ms)
            self.meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
            metrics.set_meter_provider(self.meter_provider)
            meter = metrics.get_meter("go2_isaaclab_training")
            self.run_counter = meter.create_counter("go2.training.runs")
            self.heartbeat_counter = meter.create_counter("go2.training.heartbeats")
            self.duration_histogram = meter.create_histogram("go2.training.duration_seconds")
            self.exit_counter = meter.create_counter("go2.training.exits")
            self.running_counter = meter.create_up_down_counter("go2.training.processes.running")
            self.enabled = True
            print("[go2] OpenTelemetry enabled", flush=True)
        except Exception as exc:
            print(f"[go2] OpenTelemetry disabled: {exc}", flush=True)

    @contextmanager
    def launch_span(self):
        if not self.enabled or self.tracer is None:
            yield None
            return
        span_attrs = dict(self.attrs)
        span_attrs["process.command"] = self.command[:4096]
        with self.tracer.start_as_current_span("go2.training.launch", attributes=span_attrs) as span:
            yield span

    def start(self, pid: int) -> None:
        if not self.enabled:
            return
        attrs = dict(self.attrs)
        attrs["process.pid"] = pid
        self.run_counter.add(1, attrs)
        self.running_counter.add(1, attrs)

    def heartbeat(self, elapsed_seconds: float) -> None:
        if not self.enabled:
            return
        attrs = dict(self.attrs)
        attrs["go2.elapsed_seconds"] = int(elapsed_seconds)
        self.heartbeat_counter.add(1, attrs)
        if self.tracer is not None:
            with self.tracer.start_as_current_span("go2.training.heartbeat", attributes=attrs):
                pass

    def finish(self, return_code: int, duration_seconds: float) -> None:
        if not self.enabled:
            return
        attrs = dict(self.attrs)
        attrs["process.exit_code"] = int(return_code)
        attrs["go2.success"] = return_code == 0
        self.duration_histogram.record(float(duration_seconds), attrs)
        self.exit_counter.add(1, attrs)
        self.running_counter.add(-1, attrs)

    def shutdown(self) -> None:
        if self.provider is not None:
            self.provider.shutdown()
        if self.meter_provider is not None:
            self.meter_provider.shutdown()


def find_isaaclab_root() -> Path:
    candidates = [
        os.environ.get("ISAACLAB_ROOT"),
        "/workspace/IsaacLab",
        "/IsaacLab",
        str(Path.cwd() / "IsaacLab"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(candidate)
        if (root / "isaaclab.sh").exists():
            return root
    raise FileNotFoundError(
        "Could not find IsaacLab. Set ISAACLAB_ROOT to the directory containing isaaclab.sh."
    )


def parse_args(argv: list[str]):
    import argparse

    parser = argparse.ArgumentParser(description="Launch local Go2 IsaacLab training/video jobs.")
    parser.add_argument("--task", default=os.environ.get("GO2_TASK"))
    parser.add_argument("--num-envs", default=os.environ.get("GO2_NUM_ENVS", "4096"))
    parser.add_argument("--max-iterations", default=os.environ.get("GO2_MAX_ITERATIONS", "20000"))
    parser.add_argument("--experiment-name", default=os.environ.get("GO2_EXPERIMENT_NAME", "go2_lidar_walk_flat"))
    parser.add_argument("--run-name", default=os.environ.get("GO2_RUN_NAME"))
    parser.add_argument("--headless", action="store_true", default=env_bool("GO2_HEADLESS", True))
    parser.add_argument("--no-headless", action="store_false", dest="headless")
    parser.add_argument("--record-video", action="store_true", default=env_bool("GO2_RECORD_VIDEO", True))
    parser.add_argument("--no-record-video", action="store_false", dest="record_video")
    parser.add_argument("--video-length", default=os.environ.get("GO2_VIDEO_LENGTH", "300"))
    parser.add_argument("--video-interval", default=os.environ.get("GO2_VIDEO_INTERVAL"))
    parser.add_argument("--export-video", help="Stream newest generated MP4 to HOST:PORT after training exits.")
    parser.add_argument("--auto-resume", action="store_true", default=env_bool("GO2_AUTO_RESUME", True))
    parser.add_argument("--no-auto-resume", action="store_false", dest="auto_resume")
    parser.add_argument(
        "extra_args",
        nargs="*",
        help="Additional arguments passed through to IsaacLab's RSL-RL trainer.",
    )
    return parser.parse_args(argv)


def build_train_command(args, video_interval: str | None) -> list[str]:
    root = find_isaaclab_root()
    wrapper_script = Path(__file__).resolve().parent / "scripts" / "run_isaaclab_train.py"
    if not wrapper_script.exists():
        raise FileNotFoundError(f"Missing local IsaacLab train wrapper: {wrapper_script}")

    cmd = [
        str(root / "isaaclab.sh"),
        "-p",
        str(wrapper_script),
        "--task",
        args.task,
        "--num_envs",
        str(args.num_envs),
        "--max_iterations",
        str(args.max_iterations),
        "--livestream",
        "0",
        "--experience",
        str(root / "apps" / ("isaaclab.python.headless.rendering.kit" if args.record_video else "isaaclab.python.headless.kit")),
    ]

    if args.experiment_name:
        cmd += ["--experiment_name", args.experiment_name]

    if args.run_name:
        cmd += ["--run_name", args.run_name]

    if args.headless:
        cmd.append("--headless")

    if args.record_video:
        cmd += ["--video", "--video_length", str(args.video_length), "--video_interval", str(video_interval)]

    if args.auto_resume:
        resume_args = latest_resume_args(Path(__file__).resolve().parent / "logs", args.experiment_name)
        if resume_args:
            print("[go2] auto-resume enabled:", flush=True)
            print("[go2] " + shlex.join(resume_args), flush=True)
            cmd.extend(resume_args)
        else:
            print("[go2] auto-resume enabled but no checkpoint was found", flush=True)

    extra_args = os.environ.get("GO2_TRAIN_EXTRA_ARGS", "").split()
    extra_args.extend(args.extra_args)
    cmd.extend(extra_args)
    return cmd


def checkpoint_iteration(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def latest_resume_args(log_root: Path, experiment_name: str | None) -> list[str]:
    experiment = experiment_name or "go2_lidar_walk_flat"
    exp_root = log_root / "rsl_rl" / experiment
    if not exp_root.exists():
        return []
    candidates: list[tuple[int, float, Path, Path]] = []
    for run_dir in exp_root.iterdir():
        if not run_dir.is_dir():
            continue
        for checkpoint in run_dir.glob("model_*.pt"):
            iteration = checkpoint_iteration(checkpoint)
            if iteration <= 0:
                continue
            candidates.append((iteration, checkpoint.stat().st_mtime, run_dir, checkpoint))
    if not candidates:
        return []
    _, _, run_dir, checkpoint = max(candidates, key=lambda item: (item[0], item[1]))
    return ["--resume", "--load_run", run_dir.name, "--checkpoint", checkpoint.name]


def artifact_snapshot(log_root: Path) -> tuple[float, int]:
    newest_mtime = 0.0
    total_size = 0
    if not log_root.exists():
        return newest_mtime, total_size
    for path in log_root.rglob("*"):
        try:
            if not path.is_file():
                continue
            if path.name in {
                "go2_training_heartbeat.txt",
                "go2_launcher_status.log",
                "go2_train_child_output.log",
            }:
                continue
            stat = path.stat()
        except OSError:
            continue
        newest_mtime = max(newest_mtime, stat.st_mtime)
        total_size += stat.st_size
    return newest_mtime, total_size


def run_training_process(cmd: list[str], env: dict[str, str], telemetry: TrainingTelemetry) -> int:
    heartbeat_seconds = float(os.environ.get("GO2_OTEL_HEARTBEAT_SECONDS", str(DEFAULT_OTEL_HEARTBEAT_SECONDS)))
    watchdog_enabled = env_bool("GO2_STALL_WATCHDOG_ENABLED", True)
    stall_grace_seconds = env_int("GO2_STALL_GRACE_SECONDS", DEFAULT_STALL_GRACE_SECONDS)
    stall_timeout_seconds = env_int("GO2_STALL_TIMEOUT_SECONDS", DEFAULT_STALL_TIMEOUT_SECONDS)
    log_root = Path(__file__).resolve().parent / "logs"
    heartbeat_path = log_root / "go2_training_heartbeat.txt"
    status_path = log_root / "go2_launcher_status.log"
    child_output_path = log_root / "go2_train_child_output.log"
    last_snapshot = artifact_snapshot(log_root)
    last_progress_at = time.monotonic()
    started_at = time.monotonic()
    log_root.mkdir(parents=True, exist_ok=True)
    try:
        child_output_file = child_output_path.open("a", encoding="utf-8")
        child_output_file.write(
            f"\n\n== child launch utc={datetime.now(timezone.utc).isoformat()} ==\n"
        )
        child_output_file.write(shlex.join(cmd) + "\n")
        child_output_file.flush()
    except OSError:
        child_output_file = None
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=child_output_file if child_output_file is not None else None,
        stderr=subprocess.STDOUT if child_output_file is not None else None,
        text=True,
    )
    telemetry.start(process.pid)
    try:
        status_path.parent.mkdir(parents=True, exist_ok=True)
        with status_path.open("a", encoding="utf-8") as status_file:
            status_file.write(
                f"utc={datetime.now(timezone.utc).isoformat()} event=child_started "
                f"pid={process.pid} snapshot_mtime={last_snapshot[0]:.3f} "
                f"snapshot_bytes={last_snapshot[1]}\n"
            )
            status_file.write(
                f"utc={datetime.now(timezone.utc).isoformat()} event=watchdog_config "
                f"enabled={int(watchdog_enabled)} grace_seconds={stall_grace_seconds} "
                f"timeout_seconds={stall_timeout_seconds} heartbeat_seconds={heartbeat_seconds:.1f} "
                f"num_envs={os.environ.get('GO2_NUM_ENVS', '')} "
                f"record_video={os.environ.get('GO2_RECORD_VIDEO', '')} "
                f"auto_resume={os.environ.get('GO2_AUTO_RESUME', '')}\n"
            )
    except OSError:
        pass
    try:
        while True:
            try:
                return_code = process.wait(timeout=heartbeat_seconds)
                try:
                    with status_path.open("a", encoding="utf-8") as status_file:
                        status_file.write(
                            f"utc={datetime.now(timezone.utc).isoformat()} event=child_exit "
                            f"pid={process.pid} return_code={return_code}\n"
                        )
                except OSError:
                    pass
                break
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - started_at
                telemetry.heartbeat(elapsed)
                try:
                    heartbeat_path.write_text(
                        f"utc={datetime.now(timezone.utc).isoformat()}\n"
                        f"elapsed_seconds={elapsed:.1f}\n"
                        f"pid={process.pid}\n",
                        encoding="utf-8",
                    )
                except OSError:
                    pass
                snapshot = artifact_snapshot(log_root)
                if snapshot != last_snapshot:
                    last_snapshot = snapshot
                    last_progress_at = time.monotonic()
                    progress_state = "advanced"
                else:
                    progress_state = "stale"
                stalled_for = time.monotonic() - last_progress_at
                try:
                    with status_path.open("a", encoding="utf-8") as status_file:
                        status_file.write(
                            f"utc={datetime.now(timezone.utc).isoformat()} "
                            f"event=heartbeat elapsed_seconds={elapsed:.1f} "
                            f"pid={process.pid} progress={progress_state} "
                            f"stalled_for_seconds={stalled_for:.1f} "
                            f"snapshot_mtime={snapshot[0]:.3f} snapshot_bytes={snapshot[1]}\n"
                        )
                except OSError:
                    pass
                if progress_state == "stale" and watchdog_enabled and elapsed >= stall_grace_seconds:
                    if stalled_for >= stall_timeout_seconds:
                        print(
                            f"[go2] stall watchdog: no log/checkpoint/video file progress for "
                            f"{stalled_for:.1f}s after {elapsed:.1f}s; terminating child for restart",
                            flush=True,
                        )
                        try:
                            with status_path.open("a", encoding="utf-8") as status_file:
                                status_file.write(
                                    f"utc={datetime.now(timezone.utc).isoformat()} "
                                    f"event=stall_terminate elapsed_seconds={elapsed:.1f} "
                                    f"pid={process.pid} stalled_for_seconds={stalled_for:.1f}\n"
                                )
                        except OSError:
                            pass
                        process.terminate()
                        try:
                            return_code = process.wait(timeout=30.0)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            return_code = process.wait()
                        return int(return_code or 124)
    except KeyboardInterrupt:
        process.terminate()
        return_code = process.wait()
        raise
    finally:
        if child_output_file is not None:
            child_output_file.flush()
            child_output_file.close()
        duration_seconds = time.monotonic() - started_at
        if process.poll() is None:
            process.terminate()
            return_code = process.wait()
        telemetry.finish(int(return_code or 0), duration_seconds)
    return int(return_code or 0)


def newest_video(root: Path) -> Path | None:
    videos = list(root.rglob("*.mp4"))
    if not videos:
        return None
    return max(videos, key=lambda path: path.stat().st_mtime)


def export_video(video_path: Path, target: str) -> None:
    host, port_text = target.rsplit(":", 1)
    port = int(port_text)
    size = video_path.stat().st_size
    print(f"[go2] exporting video {video_path} ({size} bytes) to {host}:{port}", flush=True)
    with socket.create_connection((host, port), timeout=30.0) as sock:
        with video_path.open("rb") as video_file:
            while True:
                chunk = video_file.read(1024 * 1024)
                if not chunk:
                    break
                sock.sendall(chunk)
    print("[go2] video export complete", flush=True)


def prepare_log_dir() -> None:
    workspace_logs = Path(__file__).resolve().parent / "logs"
    device_logs = Path("/logs")
    if not device_logs.exists() or not device_logs.is_dir():
        return
    if workspace_logs.is_symlink():
        if workspace_logs.resolve() != device_logs:
            workspace_logs.unlink()
            workspace_logs.symlink_to(device_logs, target_is_directory=True)
        return
    if workspace_logs.exists():
        if not workspace_logs.is_dir() or any(workspace_logs.iterdir()):
            print(
                f"[go2] preserving existing non-empty log path {workspace_logs}; "
                f"persistent /logs will not be linked",
                flush=True,
            )
            return
        workspace_logs.rmdir()
    workspace_logs.symlink_to(device_logs, target_is_directory=True)
    print(f"[go2] linked {workspace_logs} -> {device_logs}", flush=True)


def write_log_marker() -> None:
    logs_path = Path(__file__).resolve().parent / "logs"
    if not logs_path.exists() or not logs_path.is_dir():
        return
    marker = logs_path / "go2_launcher_persistence_marker.txt"
    marker.write_text(
        f"created_utc={datetime.now(timezone.utc).isoformat()}\n"
        f"logs_path={logs_path}\n"
        f"resolved_path={logs_path.resolve()}\n",
        encoding="utf-8",
    )
    print(f"[go2] wrote persistence marker {marker}", flush=True)


def main() -> int:
    args = parse_args(sys.argv[1:])
    if not args.task:
        args.task = DEFAULT_TASK_CANDIDATES[0]
        print(f"[go2] GO2_TASK not set; trying default task {args.task}", flush=True)
        print(f"[go2] fallback candidates: {', '.join(DEFAULT_TASK_CANDIDATES)}", flush=True)

    prepare_log_dir()
    write_log_marker()
    video_interval = resolved_video_interval(args) if args.record_video else None
    cmd = build_train_command(args, video_interval)
    env = os.environ.copy()
    source_root = Path(__file__).resolve().parent / "source"
    task_source = source_root / "go2_isaaclab_tasks"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(task_source), str(source_root), env.get("PYTHONPATH", "")]
    )
    print("[go2] launching IsaacLab training:", flush=True)
    print("[go2] " + shlex.join(cmd), flush=True)
    telemetry = TrainingTelemetry(args, cmd, video_interval)
    try:
        with telemetry.launch_span():
            return_code = run_training_process(cmd, env, telemetry)
        if args.export_video:
            video_path = newest_video(Path(__file__).resolve().parent / "logs")
            if video_path is None:
                print("[go2] no generated MP4 found under logs", flush=True)
                return return_code or 2
            export_video(video_path, args.export_video)
        return return_code
    finally:
        telemetry.shutdown()


if __name__ == "__main__":
    sys.exit(main())
