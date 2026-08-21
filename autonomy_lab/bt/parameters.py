"""Research BT 使用的轻量连续参数定义与运行时存储。"""

from dataclasses import dataclass, replace
import math
from numbers import Real
from typing import ClassVar, Iterable


def _real_number(name: str, value: object, *, finite: bool) -> float:
    """把外部数值统一成 float，并拒绝 Python 中属于 int 的 bool。"""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    numeric = float(value)
    if math.isnan(numeric) or (finite and not math.isfinite(numeric)):
        qualifier = "finite " if finite else ""
        raise ValueError(f"{name} must be a {qualifier}real number")
    return numeric


@dataclass
class ParameterSpec:
    """描述一个连续参数的当前值、baseline 和合法范围。

    Spec 不区分参数属于 Condition 还是 Action，也不包含 optimizer、reward
    或更新频率等训练概念。``ParameterStore`` 是唯一负责修改当前值的对象。
    """

    name: str
    value: float
    default: float
    min_value: float
    max_value: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("parameter name must be a non-empty string")
        self.name = self.name.strip()
        self.min_value = _real_number(
            f"{self.name}.min_value", self.min_value, finite=False
        )
        self.max_value = _real_number(
            f"{self.name}.max_value", self.max_value, finite=False
        )
        if self.min_value > self.max_value:
            raise ValueError(
                f"invalid bounds for {self.name}: "
                f"{self.min_value} > {self.max_value}"
            )
        self.default = self._bounded_value("default", self.default)
        self.value = self._bounded_value("value", self.value)

    def _bounded_value(self, field_name: str, value: object) -> float:
        numeric = _real_number(
            f"{self.name}.{field_name}", value, finite=True
        )
        if not self.min_value <= numeric <= self.max_value:
            raise ValueError(
                f"{self.name}.{field_name} must be within "
                f"[{self.min_value}, {self.max_value}]"
            )
        return numeric


class ParameterStore:
    """按名称保存并约束一组连续参数。

    Manual、未来 optimizer 或其他实验入口只需调用 ``set()``；BT 节点每次
    tick 通过 ``get()`` 读取当前值，因此 Store 更新不会要求重新构建行为树。
    """

    def __init__(self, specs: Iterable[ParameterSpec]) -> None:
        self._specs: dict[str, ParameterSpec] = {}
        for source in specs:
            if not isinstance(source, ParameterSpec):
                raise ValueError("ParameterStore specs must be ParameterSpec values")
            if source.name in self._specs:
                raise ValueError(f"duplicate parameter: {source.name}")
            # Store 持有自己的副本，调用者不能借由构造参数绕过 set() 校验。
            self._specs[source.name] = replace(source)

    def _require_spec(self, name: str) -> ParameterSpec:
        try:
            return self._specs[name]
        except (KeyError, TypeError) as error:
            raise KeyError(f"unknown parameter: {name}") from error

    def get(self, name: str) -> float:
        """读取参数当前值。"""
        return self._require_spec(name).value

    def set(self, name: str, value: float) -> None:
        """校验并设置参数；越界时显式报错，不静默改变实验输入。"""
        spec = self._require_spec(name)
        spec.value = spec._bounded_value("value", value)

    def reset(self, name: str) -> None:
        """把单个参数恢复为自己的 default。"""
        spec = self._require_spec(name)
        spec.value = spec.default

    def reset_all(self) -> None:
        """恢复 Store 中所有参数的 default。"""
        for spec in self._specs.values():
            spec.value = spec.default

    def bounds(self, name: str) -> tuple[float, float]:
        """返回单个参数的闭区间范围。"""
        spec = self._require_spec(name)
        return spec.min_value, spec.max_value

    def spec(self, name: str) -> ParameterSpec:
        """返回 metadata 副本，防止外部直接修改 Store 内部状态。"""
        return replace(self._require_spec(name))


class ConditionParameters(ParameterStore):
    """Research BT 三个 Condition 阈值的兼容入口。

    R0.3/R0.5 已公开的属性和批量方法继续保留，但都委托给通用
    ``ParameterStore``。默认值与合法范围仍是 90/40/30 px 和 [0, +∞)。
    """

    DEFAULTS: ClassVar[dict[str, float]] = {
        "hazard_threshold": 90.0,
        "boundary_threshold": 40.0,
        "goal_threshold": 30.0,
    }
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {
        name: (0.0, math.inf) for name in DEFAULTS
    }
    _FIELDS: ClassVar[frozenset[str]] = frozenset(DEFAULTS)

    def __init__(
        self,
        hazard_threshold: float = DEFAULTS["hazard_threshold"],
        boundary_threshold: float = DEFAULTS["boundary_threshold"],
        goal_threshold: float = DEFAULTS["goal_threshold"],
    ) -> None:
        current_values = {
            "hazard_threshold": hazard_threshold,
            "boundary_threshold": boundary_threshold,
            "goal_threshold": goal_threshold,
        }
        super().__init__(
            ParameterSpec(
                name=name,
                value=current_values[name],
                default=default,
                min_value=self.BOUNDS[name][0],
                max_value=self.BOUNDS[name][1],
            )
            for name, default in self.DEFAULTS.items()
        )

    @property
    def hazard_threshold(self) -> float:
        return self.get("hazard_threshold")

    @hazard_threshold.setter
    def hazard_threshold(self, value: float) -> None:
        self.set("hazard_threshold", value)

    @property
    def boundary_threshold(self) -> float:
        return self.get("boundary_threshold")

    @boundary_threshold.setter
    def boundary_threshold(self, value: float) -> None:
        self.set("boundary_threshold", value)

    @property
    def goal_threshold(self) -> float:
        return self.get("goal_threshold")

    @goal_threshold.setter
    def goal_threshold(self, value: float) -> None:
        self.set("goal_threshold", value)

    def get_values(self) -> dict[str, float]:
        """返回三个 legacy Condition 参数的独立副本。"""
        return {name: self.get(name) for name in self.DEFAULTS}

    def set_values(self, **values: float) -> None:
        """通过通用 ``set()`` 批量更新一个或多个现有 Condition 参数。"""
        unknown = sorted(set(values) - self._FIELDS)
        if unknown:
            raise ValueError(f"unknown ConditionParameters: {', '.join(unknown)}")
        for name, value in values.items():
            self.set(name, value)

    def reset_defaults(self) -> None:
        """兼容 R0.5 API：恢复冻结的 90/40/30 px baseline。"""
        self.reset_all()

    def get_bounds(self) -> dict[str, tuple[float, float]]:
        """兼容 R0.5 API：返回三个参数 bounds 的独立副本。"""
        return {name: self.bounds(name) for name in self.DEFAULTS}
