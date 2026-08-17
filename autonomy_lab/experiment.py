"""Controller-independent single-episode recording to JSON and CSV."""

import csv
import json
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .environment import Environment


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments"
SUMMARY_FIELDS = (
    "episode_id",
    "scenario",
    "controller",
    "seed",
    "result",
    "termination_reason",
    "elapsed_time",
    "path_length",
    "collision_count",
    "bt_tick_count",
    "bt_transition_count",
)
VALID_RESULTS = {"SUCCESS", "TIMEOUT", "FAILURE", "INTERRUPTED"}


class ExperimentRecorder:
    """Collect and persist metrics for one active Episode at a time."""

    def __init__(
        self, output_dir: Path = DEFAULT_OUTPUT_DIR, trajectory_interval: float = 0.1
    ) -> None:
        self.output_dir = Path(output_dir)
        self.runs_dir = self.output_dir / "runs"
        self.results_path = self.output_dir / "results.csv"
        if trajectory_interval <= 0.0:
            raise ValueError("trajectory_interval must be positive")
        self.trajectory_interval = trajectory_interval
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.active = False

    def start_episode(
        self,
        environment: "Environment",
        scenario: str,
        controller: str,
        track_bt: bool = False,
    ) -> str:
        if self.active:
            raise RuntimeError("finish the active Episode before starting another")

        self.episode_id = self._next_episode_id()
        self.scenario = scenario
        self.controller = controller
        self.seed = environment.seed
        self.elapsed_time = 0.0
        self.path_length = 0.0
        self.collision_count = 0
        self.bt_tick_count = 0 if track_bt else None
        self.bt_transition_count = 0 if track_bt else None
        self.trajectory = [self._trajectory_point(environment)]
        self._next_trajectory_time = self.trajectory_interval
        self._last_position = (
            float(environment.agent.position.x),
            float(environment.agent.position.y),
        )
        self._collision_active = False
        self._last_active_action: str | None = None
        self._track_bt = track_bt
        self._wall_start = time.perf_counter()
        self.active = True
        return self.episode_id

    def update(
        self,
        dt: float,
        environment: "Environment",
        active_action: str | None = None,
        bt_ticked: bool = False,
    ) -> None:
        if not self.active:
            raise RuntimeError("start an Episode before recording updates")
        if dt < 0.0:
            raise ValueError("dt must be non-negative")

        self.elapsed_time += dt
        current_position = (
            float(environment.agent.position.x),
            float(environment.agent.position.y),
        )
        self.path_length += math.dist(self._last_position, current_position)
        self._last_position = current_position

        colliding = bool(environment.collision_this_step)
        if colliding and not self._collision_active:
            self.collision_count += 1
        self._collision_active = colliding

        if self._track_bt and bt_ticked:
            self.bt_tick_count += 1
            if active_action not in (None, "None", "Target Reached"):
                if (
                    self._last_active_action is not None
                    and active_action != self._last_active_action
                ):
                    self.bt_transition_count += 1
                self._last_active_action = active_action

        if self.elapsed_time + 1e-9 >= self._next_trajectory_time:
            self.trajectory.append(self._trajectory_point(environment))
            while self._next_trajectory_time <= self.elapsed_time + 1e-9:
                self._next_trajectory_time += self.trajectory_interval

    def finish_episode(self, result: str, termination_reason: str) -> dict:
        if not self.active:
            raise RuntimeError("there is no active Episode to finish")
        if result not in VALID_RESULTS:
            raise ValueError(f"unsupported Episode result: {result}")

        final_point = [
            round(self.elapsed_time, 6),
            round(self._last_position[0], 3),
            round(self._last_position[1], 3),
        ]
        if self.trajectory[-1] != final_point:
            self.trajectory.append(final_point)

        payload = {
            "episode_id": self.episode_id,
            "scenario": self.scenario,
            "controller": self.controller,
            "seed": self.seed,
            "result": result,
            "termination_reason": termination_reason,
            "elapsed_time": round(self.elapsed_time, 6),
            "wall_clock_time": round(time.perf_counter() - self._wall_start, 3),
            "path_length": round(self.path_length, 3),
            "collision_count": self.collision_count,
            "bt_tick_count": self.bt_tick_count,
            "bt_transition_count": self.bt_transition_count,
            "trajectory": self.trajectory,
        }
        json_path = self.runs_dir / f"episode_{self.episode_id}.json"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._append_summary(payload)
        self.active = False
        self._print_summary(payload, json_path)
        return payload

    def _next_episode_id(self) -> str:
        existing_ids = []
        for path in self.runs_dir.glob("episode_*.json"):
            try:
                existing_ids.append(int(path.stem.removeprefix("episode_")))
            except ValueError:
                continue
        return f"{max(existing_ids, default=0) + 1:04d}"

    def _trajectory_point(self, environment: "Environment") -> list[float]:
        return [
            round(self.elapsed_time, 6),
            round(float(environment.agent.position.x), 3),
            round(float(environment.agent.position.y), 3),
        ]

    def _append_summary(self, payload: dict) -> None:
        write_header = not self.results_path.exists() or self.results_path.stat().st_size == 0
        with self.results_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow({field: payload.get(field) for field in SUMMARY_FIELDS})

    def _print_summary(self, payload: dict, json_path: Path) -> None:
        print(f"\nExperiment #{payload['episode_id']} completed")
        print(f"Scenario:        {payload['scenario']}")
        print(f"Controller:      {payload['controller']}")
        print(f"Result:          {payload['result']}")
        print(f"Time:            {payload['elapsed_time']:.2f} s")
        print(f"Path Length:     {payload['path_length']:.1f} px")
        print(f"Collisions:      {payload['collision_count']}")
        if payload["bt_transition_count"] is not None:
            print(f"BT Transitions:  {payload['bt_transition_count']}")
        print(f"Saved:           {json_path}")
