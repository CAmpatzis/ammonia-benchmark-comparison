import atexit
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import gymnasium as gym
import numpy as np

WORKSPACE_DIR = Path(__file__).resolve().parent
SUPPORT_DIR = WORKSPACE_DIR / ".dwsim_support"
DEFAULT_FLOWSHEET = WORKSPACE_DIR / "DWSIM Simulation" / "Ammonia_RL.dwxmz"
DEFAULT_BRIDGE_SCRIPT = SUPPORT_DIR / "dwsim_rosetta_bridge.py"
DEFAULT_ROSETTA_WRAPPER = WORKSPACE_DIR / ".venv311-rosetta" / "bin" / "python-rosetta"
DEFAULT_ROSETTA_PYTHON = WORKSPACE_DIR / ".venv311-rosetta" / "bin" / "python"
DEFAULT_MONO_ROOT = WORKSPACE_DIR / ".mono-runtime-x64" / "mono" / "6.14.1"
DEFAULT_DWSIM_AUTOMATION_DIR = Path("/Applications/DWSIM.app/Contents/MonoBundle")


class RosettaDWSIMBridge:
    def __init__(
        self,
        flowsheet_path: Path,
        bridge_script: Path,
        mono_root: Path,
        dwsim_automation_dir: Path,
        rosetta_wrapper: Path,
        rosetta_python: Path,
    ) -> None:
        self.flowsheet_path = flowsheet_path
        self.bridge_script = bridge_script
        self.mono_root = mono_root
        self.dwsim_automation_dir = dwsim_automation_dir
        self.rosetta_wrapper = rosetta_wrapper
        self.rosetta_python = rosetta_python
        self.process: Optional[subprocess.Popen[str]] = None

    def _command(self) -> Sequence[str]:
        if self.rosetta_wrapper.is_file():
            return [str(self.rosetta_wrapper), "-u", str(self.bridge_script), str(self.flowsheet_path)]

        if self.rosetta_python.is_file():
            return ["arch", "-x86_64", str(self.rosetta_python), "-u", str(self.bridge_script), str(self.flowsheet_path)]

        raise FileNotFoundError(
            "Rosetta Python launcher not found. Checked: "
            f"{self.rosetta_wrapper} and {self.rosetta_python}"
        )

    def start(self) -> subprocess.Popen[str]:
        if self.process is not None and self.process.poll() is None:
            return self.process

        env = os.environ.copy()
        env.setdefault("DWSIM_AUTOMATION_DIR", str(self.dwsim_automation_dir))
        env.setdefault("MONO_ROOT", str(self.mono_root))

        self.process = subprocess.Popen(
            self._command(),
            cwd=str(WORKSPACE_DIR),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )

        if self.process.stdout is None:
            raise RuntimeError("DWSIM bridge stdout pipe is unavailable.")

        ready_line = self.process.stdout.readline().strip()
        if not ready_line:
            stderr_output = ""
            if self.process.stderr is not None:
                stderr_output = self.process.stderr.read().strip()
            raise RuntimeError("DWSIM bridge failed to start.\n" + stderr_output)

        ready = json.loads(ready_line)
        if not ready.get("ok"):
            message = ready.get("error", "Unknown DWSIM bridge startup error")
            stderr_output = ""
            if self.process.stderr is not None:
                stderr_output = self.process.stderr.read().strip()
            if stderr_output:
                message += "\n" + stderr_output
            raise RuntimeError(message)

        return self.process

    def request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        process = self.start()

        if process.stdin is None or process.stdout is None:
            raise RuntimeError("DWSIM bridge pipes are unavailable.")

        process.stdin.write(json.dumps(payload) + "\n")
        process.stdin.flush()

        response_line = process.stdout.readline().strip()
        if not response_line:
            stderr_output = ""
            if process.stderr is not None:
                stderr_output = process.stderr.read().strip()
            message = "DWSIM bridge returned no response."
            if stderr_output:
                message += "\n" + stderr_output
            raise RuntimeError(message)

        response = json.loads(response_line)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "Unknown DWSIM bridge error"))

        return response

    def close(self) -> None:
        if self.process is None or self.process.poll() is not None:
            self.process = None
            return

        try:
            self.request({"command": "shutdown"})
        except Exception:
            pass
        finally:
            self.process.wait(timeout=5)
            self.process = None


class HaberBoschEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        flowsheet_path: str = str(DEFAULT_FLOWSHEET),
        bridge_script: str = str(DEFAULT_BRIDGE_SCRIPT),
        mono_root: str = str(DEFAULT_MONO_ROOT),
        dwsim_automation_dir: str = str(DEFAULT_DWSIM_AUTOMATION_DIR),
        rosetta_wrapper: str = str(DEFAULT_ROSETTA_WRAPPER),
        rosetta_python: str = str(DEFAULT_ROSETTA_PYTHON),
    ) -> None:
        super().__init__()

        self.flowsheet_path = Path(flowsheet_path).expanduser().resolve()
        self.bridge_script = Path(bridge_script).expanduser().resolve()
        self.mono_root = Path(mono_root).expanduser().resolve()
        self.dwsim_automation_dir = Path(dwsim_automation_dir).expanduser().resolve()
        self.rosetta_wrapper = Path(rosetta_wrapper).expanduser().resolve()
        self.rosetta_python = Path(rosetta_python).expanduser().resolve()

        if not self.flowsheet_path.is_file():
            raise FileNotFoundError(f"Flowsheet file not found: {self.flowsheet_path}")
        if not self.bridge_script.is_file():
            raise FileNotFoundError(f"DWSIM bridge script not found: {self.bridge_script}")
        if not self.mono_root.is_dir():
            raise FileNotFoundError(f"Mono runtime root not found: {self.mono_root}")
        if not self.dwsim_automation_dir.is_dir():
            raise FileNotFoundError(f"DWSIM automation directory not found: {self.dwsim_automation_dir}")

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),
            dtype=np.float64,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(5,),
            dtype=np.float64,
        )
        self.bounds = np.array(
            [
                [2.5, 3.5],
                [30.0, 80.0],
                [60.0, 150.0],
                [150.0, 300.0],
                [350.0, 500.0],
                [-30.0, 10.0],
                [0.01, 0.15],
            ],
            dtype=np.float64,
        )
        self.bridge = RosettaDWSIMBridge(
            flowsheet_path=self.flowsheet_path,
            bridge_script=self.bridge_script,
            mono_root=self.mono_root,
            dwsim_automation_dir=self.dwsim_automation_dir,
            rosetta_wrapper=self.rosetta_wrapper,
            rosetta_python=self.rosetta_python,
        )

        atexit.register(self.close)
        self.bridge.start()

    def _scale_action(self, action: Sequence[float]) -> np.ndarray:
        normalized = np.asarray(action, dtype=np.float64)
        clipped = np.clip(normalized, self.action_space.low, self.action_space.high)
        low = self.bounds[:, 0]
        high = self.bounds[:, 1]
        return low + ((clipped + 1.0) * 0.5) * (high - low)

    def step(self, action: Sequence[float]):
        physical = self._scale_action(action)

        try:
            response = self.bridge.request(
                {
                    "command": "evaluate",
                    "action": [float(value) for value in physical],
                }
            )
            observation = np.asarray(response["observation"], dtype=np.float64)
            reward = float(response["reward"])
            terminated = bool(response.get("terminated", False))
            truncated = bool(response.get("truncated", False))
            info = dict(response.get("info", {}))
            info["scaled_action"] = physical.copy()
            return observation, reward, terminated, truncated, info
        except Exception as error:
            observation = np.zeros(self.observation_space.shape, dtype=np.float64)
            return observation, -10000.0, True, False, {"error": str(error), "scaled_action": physical.copy()}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.bridge.request({"command": "reset"})
        observation = np.zeros(self.observation_space.shape, dtype=np.float64)
        return observation, {}

    def close(self) -> None:
        self.bridge.close()
