"""验证职责分组后的 package、脚本入口和 Core 依赖方向。"""

import ast
import importlib
import importlib.util
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_MODULES = (
    "autonomy_lab.core.agent",
    "autonomy_lab.core.environment",
    "autonomy_lab.core.observation",
    "autonomy_lab.perception.semantic_perception",
    "autonomy_lab.perception.pygame_perception",
    "autonomy_lab.scenarios.config",
    "autonomy_lab.scenarios.scenario_distribution",
)
SCRIPT_MODULES = (
    "scripts.demo.gym_demo",
    "scripts.demo.demo_scenario_distribution",
    "scripts.training.train_ppo",
    "scripts.training.train_hybrid_ppo",
    "scripts.evaluation.eval_ppo",
    "scripts.evaluation.compare_bt_ppo",
    "scripts.evaluation.eval_m43_generalization",
    "scripts.evaluation.eval_m51_hybrid",
    "scripts.evaluation.eval_m52_hybrid",
    "scripts.evaluation.eval_m53_final",
)
OLD_PACKAGE_MODULES = (
    "autonomy_lab.agent",
    "autonomy_lab.environment",
    "autonomy_lab.observation",
    "autonomy_lab.semantic_perception",
    "autonomy_lab.scene_config",
    "autonomy_lab.scenario_distribution",
)
CORE_FILES = ("agent.py", "environment.py", "observation.py")
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


def module_exists(module: str) -> bool:
    """父 package 尚不存在时也返回 False，便于明确报告缺失的新路径。"""
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


class ProjectStructureTests(unittest.TestCase):
    def test_responsibility_grouped_package_modules_are_importable(self):
        """缺少任一新职责模块会使调用者继续依赖旧的扁平 package。"""
        missing = [
            module
            for module in PACKAGE_MODULES
            if not module_exists(module)
        ]
        self.assertEqual(missing, [])

    def test_flat_legacy_modules_are_removed_without_compatibility_layer(self):
        """残留旧 module 会形成两套 import 路径并掩盖未完成的迁移。"""
        existing = [
            module
            for module in OLD_PACKAGE_MODULES
            if module_exists(module)
        ]
        self.assertEqual(existing, [])

    def test_entry_points_are_grouped_and_importable(self):
        """移动脚本却遗漏 package 或内部 import 会让 python -m 入口失效。"""
        missing = [
            module
            for module in SCRIPT_MODULES
            if not module_exists(module)
        ]
        self.assertEqual(missing, [])

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
            "run_hybrid_policy_episode",
        ):
            self.assertTrue(hasattr(runners, public_name), public_name)

    def test_evaluations_depend_on_common_runners_not_older_scripts(self):
        for filename in ("eval_m43_generalization.py", "eval_m51_hybrid.py"):
            path = PROJECT_ROOT / "scripts" / "evaluation" / filename
            self.assertTrue(path.is_file(), filename)
            if not path.is_file():
                continue

            imports = imported_modules(path)
            self.assertIn("autonomy_lab.experiment.runners", imports)
            self.assertNotIn("scripts.evaluation.compare_bt_ppo", imports)
            self.assertNotIn("scripts.evaluation.eval_m43_generalization", imports)

    def test_core_does_not_import_outer_layers(self):
        for filename in CORE_FILES:
            path = PROJECT_ROOT / "autonomy_lab" / "core" / filename
            self.assertTrue(path.is_file(), filename)
            if not path.is_file():
                continue
            imports = imported_modules(path)
            forbidden = {
                module
                for module in imports
                if module.startswith(FORBIDDEN_CORE_PREFIXES)
            }
            self.assertEqual(forbidden, set(), filename)


if __name__ == "__main__":
    unittest.main()
