from __future__ import annotations

import argparse
import atexit
import collections
import copy
import importlib
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from env import HaberBoschEnv


WORKSPACE_DIR = Path(__file__).resolve().parent
FLOWSHEET_PATH = WORKSPACE_DIR / "DWSIM Simulation" / "Ammonia_RL.dwxmz"
BRIDGE_SCRIPT = WORKSPACE_DIR / ".dwsim_support" / "dwsim_rosetta_bridge.py"
ROSETTA_WRAPPER = WORKSPACE_DIR / ".venv311-rosetta" / "bin" / "python-rosetta"
ROSETTA_PYTHON = WORKSPACE_DIR / ".venv311-rosetta" / "bin" / "python"
MONO_ROOT = WORKSPACE_DIR / ".mono-runtime-x64" / "mono" / "6.14.1"
DWSIM_AUTOMATION_DIR = Path("/Applications/DWSIM.app/Contents/MonoBundle")

PHYSICAL_PARAMETER_NAMES = [
    "Feed Ratio",
    "Stage 1 Pressure [bar]",
    "Stage 2 Pressure [bar]",
    "Loop Pressure [bar]",
    "Reactor Temperature [C]",
    "Condenser Temperature [C]",
    "Purge Fraction [-]",
]
NORMALIZED_PARAMETER_NAMES = [f"a{index}" for index in range(1, 8)]
OBSERVATION_NAMES = [
    "NH3 Flow [kg/s]",
    "Total Compression Duty [kW]",
    "Reactor Temperature [C]",
    "Loop Pressure [bar]",
    "Recycle Inert Mole Fraction [-]",
]

DREAMER_BOOTSTRAP_REPO_URL = "https://github.com/NM512/dreamerv3-torch.git"
DREAMER_BOOTSTRAP_REPO_DIR = WORKSPACE_DIR / "third_party" / "dreamerv3_torch"


BACKEND_STATE: dict[str, Any] = {}
DREAMER_LAST_LOGGER_STEP = 0
DREAMER_TENSORBOARD_DIR: Path | None = None
DREAMER_CHECKPOINT_DIR: Path | None = None
DREAMER_DEVICE = "cpu"
DREAMER_EPISODE_REPORT_FREQUENCY = 25
TORCH_ORIGINAL_SAVE_ATTR = "_haberbosch_original_torch_save"


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    sns.set_palette(["#0F766E", "#D97706", "#2563EB", "#B91C1C"])
    plt.rcParams.update(
        {
            "figure.figsize": (14, 6),
            "figure.dpi": 120,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelweight": "semibold",
            "savefig.bbox": "tight",
        }
    )


def build_preflight_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"item": "Workspace", "value": str(WORKSPACE_DIR), "ok": WORKSPACE_DIR.is_dir()},
            {"item": "Flowsheet", "value": str(FLOWSHEET_PATH), "ok": FLOWSHEET_PATH.is_file()},
            {"item": "Bridge Script", "value": str(BRIDGE_SCRIPT), "ok": BRIDGE_SCRIPT.is_file()},
            {"item": "Rosetta Wrapper", "value": str(ROSETTA_WRAPPER), "ok": ROSETTA_WRAPPER.is_file()},
            {"item": "Rosetta Python", "value": str(ROSETTA_PYTHON), "ok": ROSETTA_PYTHON.is_file()},
            {"item": "Mono Root", "value": str(MONO_ROOT), "ok": MONO_ROOT.is_dir()},
            {"item": "DWSIM Automation Dir", "value": str(DWSIM_AUTOMATION_DIR), "ok": DWSIM_AUTOMATION_DIR.is_dir()},
            {"item": "Notebook Kernel Architecture", "value": platform.machine(), "ok": True},
        ]
    )


def make_haber_bosch_env() -> HaberBoschEnv:
    env = HaberBoschEnv(
        flowsheet_path=str(FLOWSHEET_PATH),
        bridge_script=str(BRIDGE_SCRIPT),
        mono_root=str(MONO_ROOT),
        dwsim_automation_dir=str(DWSIM_AUTOMATION_DIR),
        rosetta_wrapper=str(ROSETTA_WRAPPER),
        rosetta_python=str(ROSETTA_PYTHON),
    )
    atexit.register(env.close)
    return env


def run_smoke_test(environment: HaberBoschEnv) -> pd.Series:
    baseline_obs, _ = environment.reset()
    step_obs, reward, terminated, truncated, info = environment.step(np.zeros(7, dtype=np.float64))
    if baseline_obs.shape != (5,):
        raise ValueError(f"Expected reset observation shape (5,), received {baseline_obs.shape}")
    if step_obs.shape != (5,):
        raise ValueError(f"Expected step observation shape (5,), received {step_obs.shape}")
    if environment.action_space.shape != (7,):
        raise ValueError(
            f"Expected normalized action shape (7,), received {environment.action_space.shape}"
        )
    return pd.Series(
        {
            "baseline_observation": baseline_obs.tolist(),
            "step_observation": step_obs.tolist(),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": dict(info),
        }
    )


def run(command: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    print(" ".join(command))
    return subprocess.run(command, cwd=cwd, check=check, text=True)


def pip_install(arguments: list[str]) -> None:
    run([sys.executable, "-m", "pip", "install", *arguments])


def ensure_repo_present(repo_url: str, repo_dir: Path) -> None:
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    if repo_dir.exists():
        print(f"Using existing repository at {repo_dir}")
        return
    run(["git", "clone", "--depth", "1", repo_url, str(repo_dir)])


def bootstrap_dreamer_runtime(
    repo_url: str = DREAMER_BOOTSTRAP_REPO_URL,
    repo_dir: Path = DREAMER_BOOTSTRAP_REPO_DIR,
) -> Path:
    pip_install(["--upgrade", "pip", "wheel", "setuptools<82"])

    requirements_path = WORKSPACE_DIR / "requirements.txt"
    if requirements_path.is_file():
        pip_install(["-r", str(requirements_path)])

    pip_install(
        [
            "torch>=2.4.1",
            "gym==0.22.0",
            "ruamel.yaml",
            "tensorboard",
            "matplotlib",
            "moviepy",
            "einops",
        ]
    )
    ensure_repo_present(repo_url, repo_dir)
    return repo_dir


@dataclass
class DreamerExperimentConfig:
    repo_url: str = DREAMER_BOOTSTRAP_REPO_URL
    seed: int = 7
    repo_dir: Path = DREAMER_BOOTSTRAP_REPO_DIR
    logdir: Path = WORKSPACE_DIR / "runs" / "dreamerv3_haber_bosch"
    seed_logdir_from: Path | None = None
    target_episodes: int = 2_000
    prefill_steps: int = 64
    replay_size: int = 2_500
    replay_policy: str = "recent"
    replay_keep_top_episodes: int = 0
    replay_keep_recent_episodes: int | None = None
    eval_every_steps: int = 200
    log_every_steps: int = 50
    eval_episodes: int = 5
    time_limit: int = 1
    envs: int = 1
    batch_size: int = 8
    batch_length: int = 2
    train_ratio: float = 16.0
    imag_horizon: int = 3
    discount: float = 0.99
    precision: int = 32
    dyn_hidden: int = 128
    dyn_deter: int = 128
    dyn_stoch: int = 16
    dyn_discrete: int = 16
    units: int = 128
    encoder_decoder_mlp_layers: int = 2
    encoder_decoder_mlp_units: int = 128
    dummy_image_size: tuple[int, int] = (8, 8)
    device_preference: str = "mps"
    install_backend: bool = True
    episode_report_frequency: int = 25
    tensorboard_subdir: str = "tensorboard"
    checkpoint_subdir: str = "checkpoints"
    archive_root: Path = WORKSPACE_DIR / "artifacts" / "dreamerv3_haber_bosch"
    reset_existing_run: bool = False
    resume_from_checkpoint: str | None = None
    dreamer_overrides: dict[str, Any] = field(default_factory=dict)


def build_quick_validation_config(base_config: DreamerExperimentConfig | None = None) -> DreamerExperimentConfig:
    base = base_config or DreamerExperimentConfig()
    return replace(
        base,
        logdir=WORKSPACE_DIR / "runs" / "dreamerv3_haber_bosch_quick_validation",
        target_episodes=12,
        prefill_steps=8,
        replay_size=64,
        eval_every_steps=6,
        log_every_steps=3,
        eval_episodes=2,
        batch_size=4,
        train_ratio=8.0,
        episode_report_frequency=4,
        reset_existing_run=True,
        resume_from_checkpoint=None,
        seed_logdir_from=None,
        dreamer_overrides=copy.deepcopy(base.dreamer_overrides),
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    return value


def serialize_experiment_config(config: DreamerExperimentConfig) -> dict[str, Any]:
    return _json_ready(asdict(config))


def _validate_replay_policy(config: DreamerExperimentConfig) -> None:
    valid_policies = {"recent", "elite_recent"}
    if config.replay_policy not in valid_policies:
        raise ValueError(
            f"Unsupported replay_policy={config.replay_policy!r}. Expected one of {sorted(valid_policies)}."
        )
    if int(config.replay_keep_top_episodes) < 0:
        raise ValueError("replay_keep_top_episodes must be >= 0.")
    if config.replay_keep_recent_episodes is not None and int(config.replay_keep_recent_episodes) < 0:
        raise ValueError("replay_keep_recent_episodes must be >= 0 when provided.")


def _episode_transition_count(episode: dict[str, Any]) -> int:
    rewards = np.asarray(episode["reward"])
    return max(int(rewards.shape[0]) - 1, 0)


def _episode_total_reward(episode: dict[str, Any]) -> float:
    rewards = np.asarray(episode["reward"], dtype=np.float64)
    return float(rewards.sum())


def _resolve_recent_episode_budget(config: DreamerExperimentConfig) -> int:
    if config.replay_keep_recent_episodes is not None:
        return max(int(config.replay_keep_recent_episodes), 0)
    return max(int(config.replay_size) - max(int(config.replay_keep_top_episodes), 0), 0)


def _select_replay_entries(
    entries: list[dict[str, Any]],
    config: DreamerExperimentConfig,
    dataset_size: int | None,
) -> tuple[list[dict[str, Any]], int]:
    if not entries:
        return [], 0

    _validate_replay_policy(config)
    step_limit = None if dataset_size in (None, 0) else int(dataset_size)
    recent_budget = _resolve_recent_episode_budget(config)

    recent_entries = sorted(entries, key=lambda entry: entry["sort_key"], reverse=True)
    if config.replay_policy == "recent" or int(config.replay_keep_top_episodes) == 0:
        selected: list[dict[str, Any]] = []
        selected_steps = 0
        for entry in recent_entries:
            if recent_budget and len(selected) >= recent_budget:
                break
            if step_limit is not None and selected_steps + entry["transitions"] > step_limit:
                continue
            selected.append(entry)
            selected_steps += entry["transitions"]
        return selected, selected_steps

    elite_budget = max(int(config.replay_keep_top_episodes), 0)
    elite_keys = {
        entry["key"]
        for entry in sorted(entries, key=lambda entry: (entry["reward"], entry["sort_key"]), reverse=True)[:elite_budget]
    }
    recent_keys = {entry["key"] for entry in recent_entries[:recent_budget]}

    candidates = []
    for entry in entries:
        is_elite = entry["key"] in elite_keys
        is_recent = entry["key"] in recent_keys
        if is_elite or is_recent:
            candidates.append({**entry, "is_elite": is_elite, "is_recent": is_recent})

    selected = []
    selected_steps = 0
    for entry in sorted(
        candidates,
        key=lambda item: (
            int(item["is_elite"] and item["is_recent"]),
            int(item["is_elite"]),
            int(item["is_recent"]),
            item["reward"],
            item["sort_key"],
        ),
        reverse=True,
    ):
        if step_limit is not None and selected_steps + entry["transitions"] > step_limit:
            continue
        selected.append(entry)
        selected_steps += entry["transitions"]

    if selected:
        return selected, selected_steps

    fallback_selected: list[dict[str, Any]] = []
    for entry in recent_entries:
        if step_limit is not None and selected_steps + entry["transitions"] > step_limit:
            continue
        fallback_selected.append(entry)
        selected_steps += entry["transitions"]
    return fallback_selected, selected_steps


def _build_cache_replay_entries(cache: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    entries = []
    for key, episode in cache.items():
        entries.append(
            {
                "key": str(key),
                "sort_key": str(key),
                "reward": _episode_total_reward(episode),
                "transitions": _episode_transition_count(episode),
            }
        )
    return entries


def _load_directory_replay_entries(directory: Path) -> list[dict[str, Any]]:
    directory = Path(directory).expanduser()
    if not directory.exists():
        return []

    entries = []
    for path in sorted(directory.glob("*.npz")):
        try:
            with path.open("rb") as file_handle:
                episode_np = np.load(file_handle)
                episode = {key: episode_np[key] for key in episode_np.keys()}
        except Exception as error:
            print(f"Could not load episode: {error}")
            continue
        entries.append(
            {
                "key": path.stem,
                "sort_key": path.stem,
                "reward": _episode_total_reward(episode),
                "transitions": _episode_transition_count(episode),
                "episode": episode,
                "path": path,
            }
        )
    return entries


def inspect_replay_inventory(
    directory: Path | str,
    config: DreamerExperimentConfig | None = None,
    *,
    dataset_size: int | None = None,
    target_reward: float | None = None,
) -> pd.DataFrame:
    entries = _load_directory_replay_entries(Path(directory))
    if not entries:
        return pd.DataFrame(
            columns=[
                "episode",
                "reward",
                "transitions",
                "recent_rank",
                "reward_rank",
                "selected_for_replay",
                "meets_target",
            ]
        )

    selected_keys: set[str] = set()
    if config is not None:
        selected_entries, _ = _select_replay_entries(
            entries,
            config,
            config.replay_size if dataset_size is None else dataset_size,
        )
        selected_keys = {entry["key"] for entry in selected_entries}

    recent_rank = {
        entry["key"]: rank
        for rank, entry in enumerate(sorted(entries, key=lambda item: item["sort_key"], reverse=True), start=1)
    }
    reward_rank = {
        entry["key"]: rank
        for rank, entry in enumerate(
            sorted(entries, key=lambda item: (item["reward"], item["sort_key"]), reverse=True),
            start=1,
        )
    }

    rows = []
    for entry in entries:
        rows.append(
            {
                "episode": entry["key"],
                "reward": entry["reward"],
                "transitions": entry["transitions"],
                "recent_rank": recent_rank[entry["key"]],
                "reward_rank": reward_rank[entry["key"]],
                "selected_for_replay": entry["key"] in selected_keys,
                "meets_target": entry["reward"] >= target_reward if target_reward is not None else pd.NA,
            }
        )

    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["selected_for_replay", "reward", "recent_rank"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def build_config_comparison_frame(
    full_config: DreamerExperimentConfig,
    quick_config: DreamerExperimentConfig,
) -> pd.DataFrame:
    rows = []
    for label, config in (("full", full_config), ("quick_validation", quick_config)):
        rows.append(
            {
                "profile": label,
                "logdir": str(config.logdir),
                "target_episodes": int(config.target_episodes),
                "prefill_steps": int(config.prefill_steps),
                "batch_size": int(config.batch_size),
                "batch_length": int(config.batch_length),
                "train_ratio": float(config.train_ratio),
                "eval_episodes": int(config.eval_episodes),
                "eval_every_steps": int(config.eval_every_steps),
                "log_every_steps": int(config.log_every_steps),
                "imag_horizon": int(config.imag_horizon),
                "resume_from_checkpoint": config.resume_from_checkpoint,
            }
        )
    return pd.DataFrame(rows).set_index("profile")


def patch_dreamer_backend_for_device(repo_dir: Path) -> dict[str, bool]:
    patch_report: dict[str, bool] = {}

    networks_path = repo_dir / "networks.py"
    original_networks = networks_path.read_text()
    updated_networks = original_networks
    updated_networks = updated_networks.replace('        device="cuda",\n', '        device="cpu",\n')
    updated_networks = updated_networks.replace(
        '        self.logits = logits\n        self.probs = torch.softmax(logits, -1)\n        self.buckets = torch.linspace(low, high, steps=255, device=device)\n',
        '        self.logits = logits\n        self.probs = torch.softmax(logits, -1)\n        resolved_device = logits.device if hasattr(logits, "device") else device\n        self.buckets = torch.linspace(low, high, steps=255, device=resolved_device)\n',
    )
    if updated_networks != original_networks:
        networks_path.write_text(updated_networks)
        patch_report["networks.py"] = True
    else:
        patch_report["networks.py"] = False

    return patch_report


def clear_dreamer_modules() -> None:
    for module_name in (
        "dreamer",
        "tools",
        "models",
        "networks",
        "exploration",
        "envs.wrappers",
    ):
        if module_name in sys.modules:
            del sys.modules[module_name]
    importlib.invalidate_caches()


def select_torch_device(torch_module: Any, preference: str) -> str:
    normalized = preference.strip().lower()
    if normalized not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError(f"Unsupported device preference: {preference!r}")
    if normalized == "cpu":
        return "cpu"
    mps_backend = getattr(torch_module.backends, "mps", None)
    if normalized in {"auto", "mps"} and mps_backend is not None and torch_module.backends.mps.is_available():
        return "mps"
    if normalized in {"auto", "cuda"} and torch_module.cuda.is_available():
        return "cuda:0"
    return "cpu"


def dreamer_run_has_state(config: DreamerExperimentConfig) -> bool:
    if not config.logdir.exists():
        return False
    if (config.logdir / "latest.pt").exists() or (config.logdir / "metrics.jsonl").exists():
        return True
    for directory in (
        config.logdir / "train_eps",
        config.logdir / "eval_eps",
        config.logdir / config.tensorboard_subdir,
        config.logdir / config.checkpoint_subdir,
    ):
        if directory.exists() and any(directory.iterdir()):
            return True
    return False


def archive_existing_dreamer_run(config: DreamerExperimentConfig, suffix: str = "snapshot") -> Path | None:
    if not dreamer_run_has_state(config):
        return None
    timestamp = time.strftime("%Y-%m-%d--%H-%M-%S")
    archive_dir = config.archive_root / f"{timestamp}--{suffix}"
    archive_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(config.logdir, archive_dir)
    manifest = {
        "source_logdir": str(config.logdir),
        "archived_at": time.time(),
        "latest_checkpoint": str(config.logdir / "latest.pt") if (config.logdir / "latest.pt").exists() else None,
    }
    (archive_dir / "archive_manifest.json").write_text(json.dumps(manifest, indent=2))
    return archive_dir


def seed_logdir_if_requested(config: DreamerExperimentConfig) -> None:
    if config.seed_logdir_from is None:
        return
    source_logdir = Path(config.seed_logdir_from)
    if config.logdir.exists():
        print(f"Using existing logdir at {config.logdir}; skipping logdir seeding.")
        return
    if not source_logdir.exists():
        raise FileNotFoundError(f"Seed logdir not found: {source_logdir}")
    print(f"Cloning Dreamer run directory: {source_logdir} -> {config.logdir}")
    shutil.copytree(source_logdir, config.logdir)


def resolve_resume_checkpoint(config: DreamerExperimentConfig) -> Path | None:
    if config.resume_from_checkpoint is None:
        return None
    if config.resume_from_checkpoint == "latest":
        candidate = config.logdir / "latest.pt"
        return candidate if candidate.is_file() else None
    candidate = Path(config.resume_from_checkpoint).expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Dreamer resume checkpoint not found: {candidate}")
    return candidate


def prepare_dreamer_artifacts(config: DreamerExperimentConfig) -> tuple[Path, Path, Path | None]:
    global DREAMER_TENSORBOARD_DIR, DREAMER_CHECKPOINT_DIR, DREAMER_LAST_LOGGER_STEP

    if config.reset_existing_run:
        archive_path = archive_existing_dreamer_run(config, suffix="pre-reset")
        if archive_path is not None:
            shutil.rmtree(config.logdir)

    seed_logdir_if_requested(config)
    config.logdir.mkdir(parents=True, exist_ok=True)

    tensorboard_dir = config.logdir / config.tensorboard_subdir
    checkpoint_dir = config.logdir / config.checkpoint_subdir
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    resume_path = resolve_resume_checkpoint(config)
    if resume_path is not None and resume_path.resolve() != (config.logdir / "latest.pt").resolve():
        staged_checkpoint = config.logdir / "latest.pt"
        shutil.copy2(resume_path, staged_checkpoint)
        print(f"Staged Dreamer checkpoint: {resume_path} -> {staged_checkpoint}")
        resume_path = staged_checkpoint
    elif resume_path is not None:
        print(f"Continuing from existing latest checkpoint: {resume_path}")
    else:
        print("No resume checkpoint selected. Training starts from scratch.")

    DREAMER_TENSORBOARD_DIR = tensorboard_dir
    DREAMER_CHECKPOINT_DIR = checkpoint_dir
    DREAMER_LAST_LOGGER_STEP = 0
    return tensorboard_dir, checkpoint_dir, resume_path


def _require_backend() -> dict[str, Any]:
    if not BACKEND_STATE:
        raise RuntimeError("Dreamer backend is not ready. Run prepare_dreamer_backend(config) first.")
    return BACKEND_STATE


def save_dreamer_checkpoint(obj: Any, path: str | Path) -> None:
    global DREAMER_LAST_LOGGER_STEP

    backend = _require_backend()
    target_path = Path(path)
    backend["original_save"](obj, target_path)
    if target_path.name != "latest.pt" or DREAMER_CHECKPOINT_DIR is None:
        return

    snapshot_step = max(int(DREAMER_LAST_LOGGER_STEP), 0)
    snapshot_path = DREAMER_CHECKPOINT_DIR / f"step_{snapshot_step:012d}.pt"
    if snapshot_path.exists():
        return

    shutil.copy2(target_path, snapshot_path)
    snapshot_record = {
        "step": snapshot_step,
        "snapshot": str(snapshot_path),
        "created_at": time.time(),
    }
    with (DREAMER_CHECKPOINT_DIR / "index.jsonl").open("a") as handle:
        handle.write(json.dumps(snapshot_record) + "\n")


class RollingEpisodeLogger:
    def __init__(self, logdir: Path, step: int) -> None:
        backend = _require_backend()
        self._delegate = backend["logger_cls"](logdir, step)
        self._train_returns = collections.deque(maxlen=DREAMER_EPISODE_REPORT_FREQUENCY)
        self._persistent_scalars: dict[str, float] = {}
        self._episode_count = 0
        self._best_reward = -np.inf
        self._started_at = time.time()
        self._device_label = DREAMER_DEVICE
        if DREAMER_TENSORBOARD_DIR is not None:
            self._delegate._writer.close()
            self._delegate._writer = backend["SummaryWriter"](
                log_dir=str(DREAMER_TENSORBOARD_DIR),
                max_queue=1000,
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def scalar(self, name: str, value: float) -> None:
        self._delegate.scalar(name, value)
        self._persistent_scalars[name] = float(value)

    def write(self, fps: bool = False, step: int | bool = False) -> None:
        global DREAMER_LAST_LOGGER_STEP

        if not step:
            step = self._delegate.step
        DREAMER_LAST_LOGGER_STEP = int(step)

        episode_summary_line = None
        if "train_return" in self._delegate._scalars:
            reward = float(self._delegate._scalars["train_return"])
            nh3 = float(self._delegate._scalars.get("log_nh3_kg_per_s", math.nan))
            power = float(self._delegate._scalars.get("log_power_kw", math.nan))
            revenue = float(self._delegate._scalars.get("log_revenue_per_h", math.nan))
            energy_cost = float(self._delegate._scalars.get("log_energy_cost_per_h", math.nan))
            self._episode_count += 1
            self._train_returns.append(reward)
            self._best_reward = max(self._best_reward, reward)
            rolling_reward = float(np.mean(self._train_returns))

            self._delegate._scalars["train_avg_reward_window"] = rolling_reward
            self._delegate._scalars["train_best_reward"] = self._best_reward
            self._delegate._scalars["train_episode_index"] = float(self._episode_count)

            if self._episode_count % max(DREAMER_EPISODE_REPORT_FREQUENCY, 1) == 0:
                elapsed_minutes = (time.time() - self._started_at) / 60.0
                episode_summary_line = (
                    f"[Dreamer][Episode {self._episode_count}] "
                    f"Reward={reward:.2f} | "
                    f"AvgReward={rolling_reward:.2f} | "
                    f"BestReward={self._best_reward:.2f} | "
                    f"NH3={nh3:.4f} kg/s | "
                    f"Power={power:.2f} kW | "
                    f"Revenue={revenue:.2f} $/h | "
                    f"EnergyCost={energy_cost:.2f} $/h | "
                    f"EnvSteps={int(step)} | "
                    f"Updates={int(self._persistent_scalars.get('update_count', 0.0))} | "
                    f"DatasetSize={int(self._persistent_scalars.get('dataset_size', 0.0))} | "
                    f"Device={self._device_label} | "
                    f"ElapsedMin={elapsed_minutes:.1f}"
                )

        scalars = list(self._delegate._scalars.items())
        if fps:
            scalars.append(("fps", self._delegate._compute_fps(step)))

        should_print_default = "train_return" not in self._delegate._scalars
        if should_print_default:
            print(f"[{step}]", " / ".join(f"{name} {value:.1f}" for name, value in scalars))
        if episode_summary_line is not None:
            print(episode_summary_line)

        with (self._delegate._logdir / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps({"step": step, **dict(scalars)}) + "\n")
        for name, value in scalars:
            if "/" not in name:
                self._delegate._writer.add_scalar("scalars/" + name, value, step)
            else:
                self._delegate._writer.add_scalar(name, value, step)
        for name, value in self._delegate._images.items():
            self._delegate._writer.add_image(name, value, step)
        for name, value in self._delegate._videos.items():
            video_name = name if isinstance(name, str) else name.decode("utf-8")
            if np.issubdtype(value.dtype, np.floating):
                value = np.clip(255 * value, 0, 255).astype(np.uint8)
            batch, horizon, height, width, channels = value.shape
            value = value.transpose(1, 4, 2, 0, 3).reshape((1, horizon, channels, height, batch * width))
            self._delegate._writer.add_video(video_name, value, step, 16)

        self._delegate._writer.flush()
        self._delegate._scalars = {}
        self._delegate._images = {}
        self._delegate._videos = {}


class HaberBoschDreamerAdapter:
    metadata: dict[str, Any] = {}

    def __init__(self, seed: int = 0, image_size: tuple[int, int] = (8, 8)) -> None:
        self._seed = int(seed)
        self._image_size = tuple(int(value) for value in image_size)
        self._env = make_haber_bosch_env()
        self._zero_image = np.zeros((*self._image_size, 3), dtype=np.uint8)
        self.reward_range = (-np.inf, np.inf)

    @property
    def observation_space(self):
        gym = importlib.import_module("gym")
        scalar = lambda: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32)
        bool_scalar = lambda: gym.spaces.Box(low=0, high=1, shape=(), dtype=np.bool_)
        spaces = {
            "state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
            "image": gym.spaces.Box(low=0, high=255, shape=(*self._image_size, 3), dtype=np.uint8),
            "is_first": bool_scalar(),
            "is_last": bool_scalar(),
            "is_terminal": bool_scalar(),
            "log_reward": scalar(),
            "log_nh3_kg_per_s": scalar(),
            "log_nh3_ton_per_h": scalar(),
            "log_power_kw": scalar(),
            "log_revenue_per_h": scalar(),
            "log_energy_cost_per_h": scalar(),
        }
        for name in NORMALIZED_PARAMETER_NAMES:
            spaces[f"log_{name}_normalized"] = scalar()
            spaces[f"log_{name}_physical"] = scalar()
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        gym = importlib.import_module("gym")
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

    def _obs(
        self,
        state: np.ndarray,
        *,
        is_first: bool,
        is_last: bool,
        is_terminal: bool,
        reward: float,
        info: dict[str, Any] | None = None,
        action: np.ndarray | None = None,
    ) -> dict[str, Any]:
        info = info or {}
        normalized_action = np.asarray(action if action is not None else np.zeros(7, dtype=np.float32), dtype=np.float32)
        physical_action = np.asarray(info.get("scaled_action", np.zeros(7, dtype=np.float64)), dtype=np.float32)
        metric_defaults = {
            "nh3_kg_per_s": 0.0,
            "nh3_ton_per_h": 0.0,
            "power_kw": 0.0,
            "revenue_per_h": 0.0,
            "energy_cost_per_h": 0.0,
        }

        obs = {
            "state": np.asarray(state, dtype=np.float32),
            "image": self._zero_image.copy(),
            "is_first": bool(is_first),
            "is_last": bool(is_last),
            "is_terminal": bool(is_terminal),
            "log_reward": np.float32(reward),
            "log_nh3_kg_per_s": np.float32(info.get("nh3_kg_per_s", metric_defaults["nh3_kg_per_s"])),
            "log_nh3_ton_per_h": np.float32(info.get("nh3_ton_per_h", metric_defaults["nh3_ton_per_h"])),
            "log_power_kw": np.float32(info.get("power_kw", metric_defaults["power_kw"])),
            "log_revenue_per_h": np.float32(info.get("revenue_per_h", metric_defaults["revenue_per_h"])),
            "log_energy_cost_per_h": np.float32(info.get("energy_cost_per_h", metric_defaults["energy_cost_per_h"])),
        }
        for index, name in enumerate(NORMALIZED_PARAMETER_NAMES):
            obs[f"log_{name}_normalized"] = np.float32(normalized_action[index])
            obs[f"log_{name}_physical"] = np.float32(physical_action[index])
        return obs

    def reset(self):
        state, _ = self._env.reset(seed=self._seed)
        return self._obs(
            state,
            is_first=True,
            is_last=False,
            is_terminal=False,
            reward=0.0,
        )

    def step(self, action: np.ndarray):
        normalized_action = np.asarray(action, dtype=np.float64).reshape(-1)
        state, reward, _, _, info = self._env.step(normalized_action)
        transition_info = dict(info)
        transition_info["discount"] = np.array(0.0, dtype=np.float32)
        obs = self._obs(
            state,
            is_first=False,
            is_last=True,
            is_terminal=True,
            reward=float(reward),
            info=transition_info,
            action=normalized_action,
        )
        return obs, float(reward), True, transition_info

    def render(self, *args: Any, **kwargs: Any):
        del args, kwargs
        return self._zero_image.copy()

    def close(self) -> None:
        self._env.close()


def recursive_update(base: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            recursive_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def build_dreamer_namespace(config: DreamerExperimentConfig) -> argparse.Namespace:
    backend = _require_backend()
    yaml_loader = backend["YAML"](typ="safe")
    repo_configs = yaml_loader.load((config.repo_dir / "configs.yaml").read_text())

    merged: dict[str, Any] = {}
    for name in ("defaults", "dmc_proprio"):
        recursive_update(merged, repo_configs[name])

    recursive_update(
        merged,
        {
            "task": "haberbosch_single_step",
            "seed": int(config.seed),
            "logdir": str(config.logdir),
            "steps": int(config.target_episodes),
            "envs": int(config.envs),
            "action_repeat": 1,
            "time_limit": int(config.time_limit),
            "eval_every": int(config.eval_every_steps),
            "log_every": int(config.log_every_steps),
            "eval_episode_num": int(config.eval_episodes),
            "prefill": int(config.prefill_steps),
            "dataset_size": int(config.replay_size),
            "train_ratio": float(config.train_ratio),
            "batch_size": int(config.batch_size),
            "batch_length": int(config.batch_length),
            "discount": float(config.discount),
            "imag_horizon": int(config.imag_horizon),
            "video_pred_log": False,
            "compile": False,
            "precision": int(config.precision),
            "device": DREAMER_DEVICE,
            "dyn_hidden": int(config.dyn_hidden),
            "dyn_deter": int(config.dyn_deter),
            "dyn_stoch": int(config.dyn_stoch),
            "dyn_discrete": int(config.dyn_discrete),
            "units": int(config.units),
            "expl_behavior": "greedy",
            "encoder": {
                **merged["encoder"],
                "mlp_keys": "state",
                "cnn_keys": "$^",
                "mlp_layers": int(config.encoder_decoder_mlp_layers),
                "mlp_units": int(config.encoder_decoder_mlp_units),
            },
            "decoder": {
                **merged["decoder"],
                "mlp_keys": "state",
                "cnn_keys": "$^",
                "mlp_layers": int(config.encoder_decoder_mlp_layers),
                "mlp_units": int(config.encoder_decoder_mlp_units),
            },
            "size": list(config.dummy_image_size),
        },
    )
    if config.dreamer_overrides:
        recursive_update(merged, copy.deepcopy(config.dreamer_overrides))
    return argparse.Namespace(**merged)


def prepare_dreamer_backend(config: DreamerExperimentConfig) -> dict[str, Any]:
    global BACKEND_STATE, DREAMER_DEVICE, DREAMER_EPISODE_REPORT_FREQUENCY

    _validate_replay_policy(config)

    if config.install_backend:
        bootstrap_dreamer_runtime(config.repo_url, config.repo_dir)
    else:
        ensure_repo_present(config.repo_url, config.repo_dir)

    patch_report = patch_dreamer_backend_for_device(config.repo_dir)
    clear_dreamer_modules()

    if str(config.repo_dir) not in sys.path:
        sys.path.insert(0, str(config.repo_dir))

    torch_module = importlib.import_module("torch")
    original_torch_save = getattr(torch_module, TORCH_ORIGINAL_SAVE_ATTR, None)
    if original_torch_save is None:
        current_torch_save = torch_module.save
        serialization_save = getattr(getattr(torch_module, "serialization", None), "save", None)
        if getattr(current_torch_save, "__name__", "") == "save_dreamer_checkpoint" and callable(serialization_save):
            original_torch_save = serialization_save
        else:
            original_torch_save = current_torch_save
        setattr(torch_module, TORCH_ORIGINAL_SAVE_ATTR, original_torch_save)
    torch_module.save = original_torch_save

    dreamer_module = importlib.import_module("dreamer")
    dreamer_tools_module = importlib.import_module("tools")
    dreamer_wrappers_module = importlib.import_module("envs.wrappers")
    yaml_cls = importlib.import_module("ruamel.yaml").YAML
    summary_writer_cls = importlib.import_module("torch.utils.tensorboard").SummaryWriter

    DREAMER_DEVICE = select_torch_device(torch_module, config.device_preference)
    DREAMER_EPISODE_REPORT_FREQUENCY = max(1, int(config.episode_report_frequency))
    tensorboard_dir, checkpoint_dir, resume_path = prepare_dreamer_artifacts(config)

    BACKEND_STATE = {
        "torch": torch_module,
        "dreamer": dreamer_module,
        "dreamer_tools": dreamer_tools_module,
        "dreamer_wrappers": dreamer_wrappers_module,
        "YAML": yaml_cls,
        "SummaryWriter": summary_writer_cls,
        "logger_cls": dreamer_tools_module.Logger,
        "original_make_env": dreamer_module.make_env,
        "original_save": original_torch_save,
        "original_load_episodes": dreamer_tools_module.load_episodes,
        "original_erase_over_episodes": dreamer_tools_module.erase_over_episodes,
        "patch_report": patch_report,
        "tensorboard_dir": tensorboard_dir,
        "checkpoint_dir": checkpoint_dir,
        "resume_path": resume_path,
    }

    namespace = build_dreamer_namespace(config)

    def haberbosch_make_env(config_namespace: argparse.Namespace, mode: str, env_id: int):
        suite, _ = config_namespace.task.split("_", 1)
        if suite != "haberbosch":
            return BACKEND_STATE["original_make_env"](config_namespace, mode, env_id)
        env = HaberBoschDreamerAdapter(
            seed=int(config_namespace.seed) + env_id,
            image_size=tuple(config_namespace.size),
        )
        env = BACKEND_STATE["dreamer_wrappers"].TimeLimit(env, int(config_namespace.time_limit))
        env = BACKEND_STATE["dreamer_wrappers"].SelectAction(env, key="action")
        env = BACKEND_STATE["dreamer_wrappers"].UUID(env)
        return env

    train_episode_dir = (config.logdir / "train_eps").expanduser().resolve()

    def load_episodes_with_policy(directory, limit=None, reverse=True):
        resolved_directory = Path(directory).expanduser().resolve()
        if resolved_directory != train_episode_dir:
            return BACKEND_STATE["original_load_episodes"](directory, limit=limit, reverse=reverse)

        selected_entries, _ = _select_replay_entries(
            _load_directory_replay_entries(resolved_directory),
            config,
            config.replay_size if limit in (None, 0) else int(limit),
        )
        episodes = collections.OrderedDict()
        for entry in selected_entries:
            episodes[entry["key"]] = entry["episode"]
        return episodes

    def erase_over_episodes_with_policy(cache, dataset_size):
        entries = _build_cache_replay_entries(cache)
        selected_entries, selected_steps = _select_replay_entries(entries, config, dataset_size)
        selected_keys = {entry["key"] for entry in selected_entries}
        for key in list(cache.keys()):
            if str(key) not in selected_keys:
                del cache[key]
        return selected_steps

    dreamer_module.make_env = haberbosch_make_env
    dreamer_tools_module.Logger = RollingEpisodeLogger
    dreamer_tools_module.load_episodes = load_episodes_with_policy
    dreamer_tools_module.erase_over_episodes = erase_over_episodes_with_policy
    if hasattr(dreamer_module, "tools"):
        dreamer_module.tools.Logger = RollingEpisodeLogger
        dreamer_module.tools.load_episodes = load_episodes_with_policy
        dreamer_module.tools.erase_over_episodes = erase_over_episodes_with_policy
    dreamer_module.torch.save = save_dreamer_checkpoint

    return {
        "config_namespace": namespace,
        "patch_report": patch_report,
        "device": DREAMER_DEVICE,
        "tensorboard_dir": str(tensorboard_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "resume_path": str(resume_path) if resume_path is not None else None,
    }


def inspect_dreamer_progress(config: DreamerExperimentConfig) -> dict[str, Any]:
    backend = _require_backend()
    train_episode_dir = config.logdir / "train_eps"
    current_env_steps = backend["dreamer"].count_steps(train_episode_dir) if train_episode_dir.exists() else 0
    completed_episodes = sum(1 for _ in train_episode_dir.glob("*.npz")) if train_episode_dir.exists() else 0
    return {
        "current_env_steps": int(current_env_steps),
        "completed_episodes": int(completed_episodes),
        "remaining_episodes": int(max(config.target_episodes - completed_episodes, 0)),
        "target_episodes": int(config.target_episodes),
        "resume_checkpoint": str(resolve_resume_checkpoint(config)) if resolve_resume_checkpoint(config) is not None else None,
        "logdir": str(config.logdir),
    }


def preflight_dreamer_agent(config: DreamerExperimentConfig) -> dict[str, Any]:
    backend_info = prepare_dreamer_backend(config)
    namespace: argparse.Namespace = backend_info["config_namespace"]
    backend = _require_backend()

    validation_env = backend["dreamer"].make_env(namespace, "train", 0)
    action_space = validation_env.action_space
    namespace.num_actions = action_space.n if hasattr(action_space, "n") else action_space.shape[0]
    validation_logger = RollingEpisodeLogger(Path(namespace.logdir), 0)
    validation_agent = backend["dreamer"].Dreamer(
        validation_env.observation_space,
        validation_env.action_space,
        namespace,
        validation_logger,
        None,
    ).to(namespace.device)
    validation_env.close()
    validation_logger._delegate._writer.close()
    del validation_agent
    return {
        "dreamer_preflight": "ok",
        "device": namespace.device,
        "num_actions": int(namespace.num_actions),
        **backend_info,
    }


def run_dreamer_training(
    config: DreamerExperimentConfig,
    *,
    run_training: bool = True,
) -> dict[str, Any]:
    backend_info = prepare_dreamer_backend(config)
    namespace: argparse.Namespace = backend_info["config_namespace"]
    backend = _require_backend()

    progress_before = inspect_dreamer_progress(config)
    (config.logdir / "paper_config.json").write_text(
        json.dumps(serialize_experiment_config(config), indent=2)
    )

    if not run_training:
        print("Dreamer launch is disabled.")
        return {"ran_training": False, **progress_before, **backend_info}

    if progress_before["completed_episodes"] >= config.target_episodes:
        print("Target episode budget already reached for this logdir.")
        return {"ran_training": False, **progress_before, **backend_info}

    backend["dreamer"].main(namespace)
    progress_after = inspect_dreamer_progress(config)
    return {
        "ran_training": True,
        "before": progress_before,
        "after": progress_after,
        **backend_info,
    }


def load_metrics_frame(logdir: Path) -> pd.DataFrame:
    metrics_path = Path(logdir) / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics file found at {metrics_path}")
    records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError(f"Metrics file is empty: {metrics_path}")
    return pd.DataFrame(records)


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary = {
        "records": int(len(metrics_df)),
        "train_episode_logs": int(metrics_df["train_return"].notna().sum()) if "train_return" in metrics_df else 0,
        "best_train_reward": float(metrics_df["train_return"].max()) if "train_return" in metrics_df else math.nan,
        "best_eval_return": float(metrics_df["eval_return"].max()) if "eval_return" in metrics_df else math.nan,
        "last_logged_reward": float(metrics_df["train_return"].dropna().iloc[-1]) if "train_return" in metrics_df and metrics_df["train_return"].notna().any() else math.nan,
        "last_logged_nh3_kg_per_s": float(metrics_df["log_nh3_kg_per_s"].dropna().iloc[-1]) if "log_nh3_kg_per_s" in metrics_df and metrics_df["log_nh3_kg_per_s"].notna().any() else math.nan,
        "last_logged_revenue_per_h": float(metrics_df["log_revenue_per_h"].dropna().iloc[-1]) if "log_revenue_per_h" in metrics_df and metrics_df["log_revenue_per_h"].notna().any() else math.nan,
        "last_logged_energy_cost_per_h": float(metrics_df["log_energy_cost_per_h"].dropna().iloc[-1]) if "log_energy_cost_per_h" in metrics_df and metrics_df["log_energy_cost_per_h"].notna().any() else math.nan,
    }
    return pd.DataFrame([summary])


def load_checkpoint_index(logdir: Path) -> pd.DataFrame:
    index_path = Path(logdir) / "checkpoints" / "index.jsonl"
    if not index_path.exists():
        return pd.DataFrame(columns=["step", "snapshot", "created_at"])
    records = [json.loads(line) for line in index_path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(records)


def plot_training_diagnostics(metrics_df: pd.DataFrame):
    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.ravel()

    if "train_return" in metrics_df:
        train_df = metrics_df[metrics_df["train_return"].notna()].copy()
        axes[0].plot(train_df["step"], train_df["train_return"], label="Train Return", linewidth=1.8)
        if "train_avg_reward_window" in train_df:
            axes[0].plot(
                train_df["step"],
                train_df["train_avg_reward_window"],
                label="Rolling Reward",
                linewidth=2.2,
            )
    if "eval_return" in metrics_df:
        eval_df = metrics_df[metrics_df["eval_return"].notna()].copy()
        axes[0].plot(eval_df["step"], eval_df["eval_return"], label="Eval Return", linewidth=2.0)
    axes[0].set_title("Reward Progress")
    axes[0].set_xlabel("Environment steps")
    axes[0].set_ylabel("Reward")
    axes[0].legend(frameon=True)

    for column, label, color in (
        ("log_nh3_kg_per_s", "NH3 Flow [kg/s]", "#0F766E"),
        ("log_revenue_per_h", "Revenue [$/h]", "#2563EB"),
        ("log_energy_cost_per_h", "Energy Cost [$/h]", "#B91C1C"),
    ):
        if column in metrics_df:
            axes[1].plot(metrics_df["step"], metrics_df[column], label=label, linewidth=1.8, color=color)
    axes[1].set_title("Process Economics")
    axes[1].set_xlabel("Environment steps")
    axes[1].set_ylabel("Value")
    axes[1].legend(frameon=True)

    for column, label, color in (
        ("log_power_kw", "Power [kW]", "#D97706"),
        ("dataset_size", "Replay Size", "#111827"),
        ("update_count", "Update Count", "#2563EB"),
    ):
        if column in metrics_df:
            axes[2].plot(metrics_df["step"], metrics_df[column], label=label, linewidth=1.8, color=color)
    axes[2].set_title("System Load and Training Volume")
    axes[2].set_xlabel("Environment steps")
    axes[2].set_ylabel("Value")
    axes[2].legend(frameon=True)

    plotted_loss = False
    for column, label, color in (
        ("model_loss", "Model Loss", "#0F766E"),
        ("actor_loss", "Actor Loss", "#2563EB"),
        ("value_loss", "Value Loss", "#B91C1C"),
    ):
        if column in metrics_df:
            axes[3].plot(metrics_df["step"], metrics_df[column], label=label, linewidth=1.8, color=color)
            plotted_loss = True
    if not plotted_loss and "fps" in metrics_df:
        axes[3].plot(metrics_df["step"], metrics_df["fps"], label="FPS", linewidth=1.8, color="#2563EB")
    axes[3].set_title("Optimization Diagnostics")
    axes[3].set_xlabel("Environment steps")
    axes[3].set_ylabel("Value")
    axes[3].legend(frameon=True)

    fig.tight_layout()
    return fig, axes


def build_resume_recipes(logdir: Path) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario": "Continue in place from latest",
                "logdir": str(logdir),
                "seed_logdir_from": None,
                "resume_from_checkpoint": "latest",
                "what_it_preserves": "Weights, optimizer state, replay, and logged metrics in the same run directory.",
            },
            {
                "scenario": "Branch from explicit checkpoint",
                "logdir": str(logdir.parent / f"{logdir.name}_branch"),
                "seed_logdir_from": str(logdir),
                "resume_from_checkpoint": "latest",
                "what_it_preserves": "Whole prior logdir if seeded first; otherwise only the staged checkpoint state.",
            },
        ]
    )


__all__ = [
    "DREAMER_BOOTSTRAP_REPO_DIR",
    "DREAMER_BOOTSTRAP_REPO_URL",
    "DWSIM_AUTOMATION_DIR",
    "FLOWSHEET_PATH",
    "MONO_ROOT",
    "NORMALIZED_PARAMETER_NAMES",
    "OBSERVATION_NAMES",
    "PHYSICAL_PARAMETER_NAMES",
    "ROSETTA_PYTHON",
    "ROSETTA_WRAPPER",
    "WORKSPACE_DIR",
    "DreamerExperimentConfig",
    "archive_existing_dreamer_run",
    "bootstrap_dreamer_runtime",
    "build_config_comparison_frame",
    "build_preflight_frame",
    "build_quick_validation_config",
    "build_resume_recipes",
    "inspect_dreamer_progress",
    "inspect_replay_inventory",
    "load_checkpoint_index",
    "load_metrics_frame",
    "make_haber_bosch_env",
    "patch_dreamer_backend_for_device",
    "plot_training_diagnostics",
    "preflight_dreamer_agent",
    "prepare_dreamer_backend",
    "run_dreamer_training",
    "run_smoke_test",
    "serialize_experiment_config",
    "set_plot_style",
    "summarize_metrics",
]