"""验证 M4.4 脚本收拢、公共 runner 位置和核心依赖方向。"""

import ast
import importlib
import importlib.util
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_NAMES = (
    "gym_demo",
    "train_ppo",
    "eval_ppo",
    "compare_bt_ppo",
    "eval_m43_generalization",
)
CORE_MODULES = ("agent.py", "environment.py", "perception.py", "scene_config.py")
FORBIDDEN_CORE_PREFIXES = (
    "autonomy_lab.bt",
    "autonomy_lab.gym",
    "autonomy_lab.rendering",
    "autonomy_lab.experiment",
    "scripts",
)


def imported_modules(path: Path) -> set[str]:
    """从源码 AST 提取 import 目标，避免通过实际导入触发运行时副作用。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


class ProjectStructureTests(unittest.TestCase):
    def test_auxiliary_entry_points_live_only_in_scripts_package(self):
        self.assertTrue((PROJECT_ROOT / "scripts" / "__init__.py").is_file())
        for name in SCRIPT_NAMES:
            self.assertTrue((PROJECT_ROOT / "scripts" / f"{name}.py").is_file())
            self.assertFalse((PROJECT_ROOT / f"{name}.py").exists())

    def test_common_episode_runners_are_importable_from_experiment_package(self):
        module_name = "autonomy_lab.experiment.runners"
        spec = importlib.util.find_spec(module_name)
        self.assertIsNotNone(spec)
        if spec is None:
            return

        runners = importlib.import_module(module_name)
        for public_name in (
            "BT_DECISION_FREQUENCY",
            "PPO_ACTION_REPEAT",
            "PPO_DECISION_FREQUENCY",
            "capture_initial_state",
            "assert_initial_states_match",
            "run_bt_episode",
            "run_ppo_episode",
        ):
            self.assertTrue(hasattr(runners, public_name), public_name)

    def test_m43_script_depends_on_common_runner_not_m42_script(self):
        path = PROJECT_ROOT / "scripts" / "eval_m43_generalization.py"
        self.assertTrue(path.is_file())
        if not path.is_file():
            return

        imports = imported_modules(path)
        self.assertIn("autonomy_lab.experiment.runners", imports)
        self.assertNotIn("scripts.compare_bt_ppo", imports)
        self.assertNotIn("compare_bt_ppo", imports)

    def test_core_does_not_import_outer_layers(self):
        for filename in CORE_MODULES:
            imports = imported_modules(PROJECT_ROOT / "autonomy_lab" / filename)
            forbidden = {
                module
                for module in imports
                if module.startswith(FORBIDDEN_CORE_PREFIXES)
            }
            self.assertEqual(forbidden, set(), filename)


if __name__ == "__main__":
    unittest.main()
