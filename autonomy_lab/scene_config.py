"""集中保存可编辑的场景预设和当前阶段所需的轻量配置。"""

from copy import deepcopy


DEFAULT_SCENARIO = "simple"

# 行为节点未在 JSON params 中覆盖的数值，从这里读取场景默认值。
# 这些值描述“节点如何执行”，BT JSON 则描述“节点如何组织”。
DEFAULT_BT_CONFIG = {
    "obstacle_detection_distance": 90.0,
    "obstacle_detection_half_angle_degrees": 75.0,
    "target_reached_distance": 30.0,
    "avoid_duration": 0.9,
    "avoid_throttle": 0.75,
    "search_throttle": 0.0,
    "search_turn": 0.25,
    "gap_ray_count": 31,
    "gap_min_travel_distance": 100.0,
    "gap_safety_margin": 8.0,
    "gap_throttle": 0.5,
    "gap_open_ratio": 0.85,
    "gap_entry_ratio": 0.8,
    "gap_entry_reached_distance": 24.0,
    "gap_commit_emergency_distance": 4.0,
}

DEFAULT_SENSOR_CONFIG = {
    # range 单位为像素，fov_degrees 是以 Agent 朝向为中心的完整视场角。
    "range": 300.0,
    "fov_degrees": 120.0,
    "los_enabled": True,
}

DEFAULT_EXPERIMENT_CONFIG = {
    # 到达阈值属于 Episode 终止标准，可与 Behavior 内部阈值独立配置。
    "max_episode_time": 60.0,
    "target_reached_distance": 30.0,
}


# 预设只保存数据；get_scene() 会返回深拷贝供运行时安全修改。
SCENES = {
    "ppo_simple_obstacle": {
        "name": "PPO Simple Obstacle Beacon",
        "world_size": (850, 600),
        "seed": 44,
        "agent": {
            "position": (100, 300),
            "heading_degrees": 0.0,
            "initial_speed": 0.0,
            "max_speed": 220.0,
            "turn_speed_degrees": 150.0,
            "radius": 16,
        },
        "target": {"position": (750, 300), "radius": 18},
        # 单个障碍偏向场景上方：直线路径被挡住，但下方存在宽裕绕行空间。
        # 当前 nearest-obstacle distance/bearing 足以描述唯一的避障对象。
        "obstacles": [(350, 100, 80, 240)],
        "target_information_mode": "perceived",
        "sensor": {
            **DEFAULT_SENSOR_CONFIG,
            "range": 700.0,
            "fov_degrees": 160.0,
            # M4.1a 把 Target 视作 beacon：仍检查 range/FOV，只跳过 Target
            # 遮挡线段。AgentPerception 的 obstacle perception 不读取此开关。
            "los_enabled": False,
        },
        "behavior_tree": {**DEFAULT_BT_CONFIG},
        "experiment": {
            **DEFAULT_EXPERIMENT_CONFIG,
            "max_episode_time": 20.0,
        },
        "display": {
            "background_color": (22, 27, 36),
            "agent_color": (60, 170, 255),
            "target_color": (80, 220, 120),
            "obstacle_color": (105, 112, 125),
            "text_color": (225, 230, 238),
            "show_fps": True,
        },
    },
    "ppo_simple_obstacles": {
        "name": "PPO Simple Obstacles",
        "world_size": (850, 600),
        "seed": 43,
        "agent": {
            "position": (100, 295),
            "heading_degrees": 0.0,
            "initial_speed": 0.0,
            "max_speed": 220.0,
            "turn_speed_degrees": 150.0,
            "radius": 16,
        },
        "target": {"position": (750, 295), "radius": 18},
        # 两个矩形之间保留 y=280..310 的 30 px LOS 狭缝。Target 中心线
        # y=295 可见，但 Agent 直径为 32 px，碰撞体无法从狭缝直穿。
        # 因而策略必须从整个障碍组合的上方或下方绕行，而不是解迷宫。
        "obstacles": [
            (340, 170, 70, 110),
            (340, 310, 70, 120),
        ],
        "target_information_mode": "perceived",
        "sensor": {
            **DEFAULT_SENSOR_CONFIG,
            "range": 700.0,
            "fov_degrees": 160.0,
        },
        "behavior_tree": {**DEFAULT_BT_CONFIG},
        "experiment": {
            **DEFAULT_EXPERIMENT_CONFIG,
            "max_episode_time": 20.0,
        },
        "display": {
            "background_color": (22, 27, 36),
            "agent_color": (60, 170, 255),
            "target_color": (80, 220, 120),
            "obstacle_color": (105, 112, 125),
            "text_color": (225, 230, 238),
            "show_fps": True,
        },
    },
    "rl_sanity": {
        "name": "RL Sanity",
        "world_size": (700, 500),
        "seed": 2024,
        "agent": {
            "position": (120, 250),
            "heading_degrees": 0.0,
            "initial_speed": 0.0,
            "max_speed": 220.0,
            "turn_speed_degrees": 150.0,
            "radius": 16,
        },
        "target": {"position": (520, 250), "radius": 18},
        # M4.0 只验证 RL pipeline 是否可学习，因此刻意排除障碍导航变量。
        # 目标距离 400 px，小于传感距离 500 px，且初始 bearing=0，第一帧即可感知。
        "obstacles": [],
        "target_information_mode": "perceived",
        "sensor": {
            **DEFAULT_SENSOR_CONFIG,
            "range": 500.0,
            "fov_degrees": 120.0,
        },
        "behavior_tree": {**DEFAULT_BT_CONFIG},
        "experiment": {
            **DEFAULT_EXPERIMENT_CONFIG,
            "max_episode_time": 10.0,
        },
        "display": {
            "background_color": (22, 27, 36),
            "agent_color": (60, 170, 255),
            "target_color": (80, 220, 120),
            "obstacle_color": (105, 112, 125),
            "text_color": (225, 230, 238),
            "show_fps": True,
        },
    },
    "simple": {
        "name": "Simple",
        "world_size": (900, 700),
        "seed": 42,
        "agent": {
            "position": (100, 350),
            "heading_degrees": 0.0,
            "initial_speed": 0.0,
            "max_speed": 220.0,
            "turn_speed_degrees": 150.0,
            "radius": 16,
        },
        "target": {"position": (850, 350), "radius": 18},
        "obstacles": [
            (250, 100, 55, 330),
            (440, 330, 100, 55),  # (440, 330, 190, 55),
            (700, 270, 55, 330),
        ],
        "target_information_mode": "ground_truth",  # if target can be known without perception? ["perceived", "ground_truth"]
        "sensor": {**DEFAULT_SENSOR_CONFIG, "fov_degrees": 140.0},
        "behavior_tree": {**DEFAULT_BT_CONFIG},
        "experiment": {**DEFAULT_EXPERIMENT_CONFIG},
        "display": {
            "background_color": (22, 27, 36),
            "agent_color": (60, 170, 255),
            "target_color": (80, 220, 120),
            "obstacle_color": (105, 112, 125),
            "text_color": (225, 230, 238),
            "show_fps": True,
        },
    },
    "obstacle_course": {
        "name": "Obstacle Course",
        "world_size": (1100, 700),
        "seed": 42,
        "agent": {
            "position": (90, 350),
            "heading_degrees": 0.0,
            "initial_speed": 0.0,
            "max_speed": 220.0,
            "turn_speed_degrees": 150.0,
            "radius": 16,
        },
        "target": {"position": (1010, 350), "radius": 18},
        "obstacles": [
            (250, 100, 60, 250),
            (480, 200, 60, 100),
            (710, 450, 60, 250),
            (900, 250, 45, 200),
        ],
        "target_information_mode": "perceived",
        "sensor": {**DEFAULT_SENSOR_CONFIG},
        "behavior_tree": {**DEFAULT_BT_CONFIG},
        "experiment": {**DEFAULT_EXPERIMENT_CONFIG},
        "display": {
            "background_color": (20, 25, 34),
            "agent_color": (70, 175, 255),
            "target_color": (75, 225, 125),
            "obstacle_color": (110, 116, 130),
            "text_color": (225, 230, 238),
            "show_fps": True,
        },
    },
    "dense_obstacles": {
        "name": "Dense Obstacles",
        "world_size": (1000, 700),
        "seed": 7,
        "agent": {
            "position": (80, 620),
            "heading_degrees": -35.0,
            "initial_speed": 0.0,
            "max_speed": 190.0,
            "turn_speed_degrees": 170.0,
            "radius": 14,
        },
        "target": {"position": (920, 80), "radius": 18},
        "obstacles": [
            (170, 470, 120, 45),
            (170, 190, 45, 170),
            (340, 360, 150, 45),
            (350, 80, 45, 170),
            (540, 510, 45, 130),
            (540, 210, 150, 45),
            (690, 330, 45, 170),
            (810, 150, 120, 45),
            (810, 520, 120, 45),
        ],
        "target_information_mode": "perceived",
        "sensor": {**DEFAULT_SENSOR_CONFIG},
        "behavior_tree": {**DEFAULT_BT_CONFIG},
        "experiment": {**DEFAULT_EXPERIMENT_CONFIG},
        "display": {
            "background_color": (24, 25, 34),
            "agent_color": (80, 185, 255),
            "target_color": (90, 225, 125),
            "obstacle_color": (112, 108, 128),
            "text_color": (230, 232, 240),
            "show_fps": True,
        },
    },
}


def _make_m43_variant(
    name: str,
    *,
    target_position: tuple[int, int] | None = None,
    obstacle: tuple[int, int, int, int] | None = None,
    geometry_change: str,
) -> dict:
    """从冻结的单障碍 baseline 派生一个纯几何、test-only 场景。

    深拷贝保证 Observation、Perception、动力学、BT 参数和 episode horizon
    与训练场景完全一致；这里只允许替换 Target/Obstacle 几何。
    """
    scene = deepcopy(SCENES["ppo_simple_obstacle"])
    scene["name"] = name
    if target_position is not None:
        scene["target"]["position"] = target_position
    if obstacle is not None:
        scene["obstacles"] = [obstacle]
    scene["test_only"] = True
    scene["unseen_during_ppo_training"] = True
    scene["evaluation_group"] = "unseen_mild"
    scene["geometry_change"] = geometry_change
    return scene


# M4.3 只在冻结 Controller 上评价这些手工几何扰动；它们不参与任何训练。
SCENES.update(
    {
        "m43_target_shift": _make_m43_variant(
            "M4.3 Target Shift",
            target_position=(730, 240),
            geometry_change="Target moved from (750, 300) to (730, 240).",
        ),
        "m43_obstacle_shift": _make_m43_variant(
            "M4.3 Obstacle Shift",
            obstacle=(410, 140, 80, 240),
            geometry_change=(
                "Obstacle moved from (350, 100, 80, 240) to "
                "(410, 140, 80, 240)."
            ),
        ),
        "m43_reverse_detour": _make_m43_variant(
            "M4.3 Reverse Detour",
            obstacle=(350, 60, 80, 240),
            geometry_change=(
                "Obstacle moved upward: upper passage stays traversable but "
                "has less clearance; lower passage is the wide near route."
            ),
        ),
        "m43_combined_shift": _make_m43_variant(
            "M4.3 Combined Shift",
            target_position=(730, 350),
            obstacle=(390, 130, 80, 240),
            geometry_change="Target and obstacle both receive mild position shifts.",
        ),
    }
)


def get_scene(name: str) -> dict:
    """返回指定预设的深拷贝，防止运行时修改污染后续实验。

    测试和启动代码经常会调整嵌套的 Agent、sensor 或 behavior_tree 字典；
    浅拷贝无法隔离这些嵌套对象，因此这里必须使用 ``deepcopy``。
    """
    return deepcopy(SCENES[name])
