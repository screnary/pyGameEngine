"""以 Controller 无关方式记录单个 Episode，并写入 JSON 明细与 CSV 摘要。"""

import csv
import json
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.environment import Environment


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments"
# CSV 只保存适合跨 Episode 比较的标量；完整 trajectory 留在 JSON。
SUMMARY_FIELDS = (
    "episode_id",
    "scenario",
    "controller",
    "bt_config_id",
    "seed",
    "result",
    "termination_reason",
    "elapsed_time",
    "path_length",
    "collision_count",
    "bt_tick_count",
    "bt_transition_count",
)
PREVIOUS_SUMMARY_FIELDS = tuple(
    field for field in SUMMARY_FIELDS if field != "bt_config_id"
)
VALID_RESULTS = {"SUCCESS", "TIMEOUT", "FAILURE", "INTERRUPTED"}


class ExperimentRecorder:
    """一次只管理一个活动 Episode 的状态、指标和持久化。

    Recorder 不驱动 Environment，也不决定终止条件。main.py 在正确时机调用
    start/update/finish，因此同一个类可以记录手动和 BT 控制。
    """

    def __init__(
        self, output_dir: Path = DEFAULT_OUTPUT_DIR, trajectory_interval: float = 0.1
    ) -> None:
        # output_dir 下固定分成 runs/*.json 和一个累积 results.csv。
        self.output_dir = Path(output_dir)
        self.runs_dir = self.output_dir / "runs"
        self.results_path = self.output_dir / "results.csv"
        if trajectory_interval <= 0.0:
            raise ValueError("trajectory_interval must be positive")
        self.trajectory_interval = trajectory_interval
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        # active 防止忘记结束旧 Episode 或重复 finish 同一个结果。
        self.active = False

    def start_episode(
        self,
        environment: "Environment",
        scenario: str,
        controller: str,
        track_bt: bool = False,
        bt_config_id: str | None = None,
    ) -> str:
        """初始化新 Episode 的全部累计量，并返回新编号。

        ``track_bt=False`` 时 BT 专属指标保存为 None，而不是伪造为 0。
        """
        if self.active:
            raise RuntimeError("finish the active Episode before starting another")

        self.episode_id = self._next_episode_id()
        self.scenario = scenario
        self.controller = controller
        self.bt_config_id = bt_config_id
        self.seed = environment.seed
        # elapsed_time 使用传入 dt 累计，wall_start 单独测量真实墙钟耗时。
        self.elapsed_time = 0.0
        self.path_length = 0.0
        self.collision_count = 0
        self.bt_tick_count = 0 if track_bt else None
        self.bt_transition_count = 0 if track_bt else None
        # 轨迹始终包含 t=0 起点，后续按 trajectory_interval 采样。
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
        """在运动应用完成后，累计一帧仿真时间、路径、碰撞和 BT 指标。"""
        if not self.active:
            raise RuntimeError("start an Episode before recording updates")
        if dt < 0.0:
            raise ValueError("dt must be non-negative")

        self.elapsed_time += dt
        current_position = (
            float(environment.agent.position.x),
            float(environment.agent.position.y),
        )
        # 使用真实前后位置计算路径，碰撞导致的回退不会被算作成功位移。
        self.path_length += math.dist(self._last_position, current_position)
        self._last_position = current_position

        # 连续贴墙只计一次碰撞，离开后再次接触才产生新事件。
        colliding = bool(environment.collision_this_step)
        if colliding and not self._collision_active:
            self.collision_count += 1
        self._collision_active = colliding

        if self._track_bt and bt_ticked:
            self.bt_tick_count += 1
            # None/None 文本和 Target Reached 不是实际运行 Action，不参与转移。
            if active_action not in (None, "None", "Target Reached"):
                if (
                    self._last_active_action is not None
                    and active_action != self._last_active_action
                ):
                    # 只有连续两个有效 Action 名称不同时才记录一次状态转移。
                    self.bt_transition_count += 1
                self._last_active_action = active_action

        # 轨迹按仿真时间采样，不受实际渲染帧率波动影响。
        if self.elapsed_time + 1e-9 >= self._next_trajectory_time:
            self.trajectory.append(self._trajectory_point(environment))
            while self._next_trajectory_time <= self.elapsed_time + 1e-9:
                self._next_trajectory_time += self.trajectory_interval

    def finish_episode(self, result: str, termination_reason: str) -> dict:
        """结束并持久化当前 Episode，返回与 JSON 文件一致的 payload。"""
        if not self.active:
            raise RuntimeError("there is no active Episode to finish")
        if result not in VALID_RESULTS:
            raise ValueError(f"unsupported Episode result: {result}")

        # 即使最后一帧未落在固定采样时刻，也补充准确终点。
        final_point = [
            round(self.elapsed_time, 6),
            round(self._last_position[0], 3),
            round(self._last_position[1], 3),
        ]
        if self.trajectory[-1] != final_point:
            self.trajectory.append(final_point)

        # wall_clock_time 用于性能观察；其他时间相关实验指标使用 elapsed_time。
        payload = {
            "episode_id": self.episode_id,
            "scenario": self.scenario,
            "controller": self.controller,
            "bt_config_id": self.bt_config_id,
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
        # 先写完整 JSON，再追加 CSV；失败时不会把不完整 Episode 伪装成摘要行。
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._append_summary(payload)
        self.active = False
        self._print_summary(payload, json_path)
        return payload

    def _next_episode_id(self) -> str:
        """扫描已有 JSON 文件，返回四位递增 Episode 编号。"""
        existing_ids = []
        for path in self.runs_dir.glob("episode_*.json"):
            try:
                existing_ids.append(int(path.stem.removeprefix("episode_")))
            except ValueError:
                # 忽略不符合数字命名约定的人工文件，避免阻止正常记录。
                continue
        return f"{max(existing_ids, default=0) + 1:04d}"

    def _trajectory_point(self, environment: "Environment") -> list[float]:
        """把当前仿真时刻和 Agent 坐标压缩为一个可序列化列表。"""
        return [
            round(self.elapsed_time, 6),
            round(float(environment.agent.position.x), 3),
            round(float(environment.agent.position.y), 3),
        ]

    def _append_summary(self, payload: dict) -> None:
        """确保 CSV schema 兼容后，追加当前 Episode 的标量摘要。"""
        self._upgrade_previous_summary_schema()
        write_header = not self.results_path.exists() or self.results_path.stat().st_size == 0
        # newline="" 避免 Windows 下 csv 模块产生额外空行。
        with self.results_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow({field: payload.get(field) for field in SUMMARY_FIELDS})

    def _upgrade_previous_summary_schema(self) -> None:
        """把唯一已知旧 CSV 表头原位升级，历史行的 bt_config_id 留空。"""
        if not self.results_path.exists() or self.results_path.stat().st_size == 0:
            return
        with self.results_path.open(encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            fieldnames = tuple(reader.fieldnames or ())
            rows = list(reader)
        if fieldnames == SUMMARY_FIELDS:
            return
        if fieldnames != PREVIOUS_SUMMARY_FIELDS:
            raise ValueError(
                "results.csv has an unsupported header; expected the current "
                "or immediately previous experiment schema"
            )
        # 只迁移明确认识的旧表头，避免静默破坏未知格式的历史数据。
        with self.results_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            for row in rows:
                row["bt_config_id"] = ""
                writer.writerow({field: row.get(field) for field in SUMMARY_FIELDS})

    def _print_summary(self, payload: dict, json_path: Path) -> None:
        """在终端打印适合人工快速检查的一次 Episode 摘要。"""
        print(f"\nExperiment #{payload['episode_id']} completed")
        print(f"Scenario:        {payload['scenario']}")
        print(f"Controller:      {payload['controller']}")
        if payload["bt_config_id"] is not None:
            print(f"BT Config:       {payload['bt_config_id']}")
        print(f"Result:          {payload['result']}")
        print(f"Time:            {payload['elapsed_time']:.2f} s")
        print(f"Path Length:     {payload['path_length']:.1f} px")
        print(f"Collisions:      {payload['collision_count']}")
        if payload["bt_transition_count"] is not None:
            print(f"BT Transitions:  {payload['bt_transition_count']}")
        print(f"Saved:           {json_path}")
