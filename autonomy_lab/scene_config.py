"""Editable scene presets for the experiment environment."""

from copy import deepcopy


DEFAULT_SCENARIO = "simple"

DEFAULT_BT_CONFIG = {
    "obstacle_detection_distance": 90.0,
    "obstacle_detection_half_angle_degrees": 75.0,
    "target_reached_distance": 30.0,
    "avoid_duration": 0.9,
    "avoid_throttle": 0.75,
}

DEFAULT_EXPERIMENT_CONFIG = {
    "max_episode_time": 60.0,
    "target_reached_distance": 30.0,
}


SCENES = {
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
            (440, 330, 190, 55),
            (700, 270, 55, 330),
        ],
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
            (250, 0, 60, 500),
            (480, 200, 60, 500),
            (710, 0, 60, 500),
            (900, 250, 45, 200),
        ],
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


def get_scene(name: str) -> dict:
    """Return an independent copy so runtime state cannot modify a preset."""
    return deepcopy(SCENES[name])
