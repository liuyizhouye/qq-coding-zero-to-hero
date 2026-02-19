import json
import time
from functools import update_wrapper, wraps
from typing import Callable, TypedDict, cast

# 本模块演示三类装饰器：
# 1) 函数装饰器 timed
# 2) 参数化装饰器 with_tag(tag)
# 3) 类装饰器 CountCalls
#
# 统一通过 run_demo 返回 trace，便于观察调用链路和执行顺序。
TraceEvent = dict[str, object]


class DemoResult(TypedDict):
    final_answer: str
    trace: list[TraceEvent]


def _extract_trace(kwargs: dict[str, object]) -> list[TraceEvent] | None:
    """从 kwargs 里读取内部约定的 _trace 通道。"""
    maybe_trace = kwargs.get("_trace")
    if isinstance(maybe_trace, list):
        return cast(list[TraceEvent], maybe_trace)
    return None


def timed(func: Callable[..., float]) -> Callable[..., float]:
    """函数装饰器：记录耗时（毫秒）并写入 trace。"""

    @wraps(func)
    def wrapper(*args: object, **kwargs: object) -> float:
        kw = cast(dict[str, object], kwargs)
        trace = _extract_trace(kw)
        if trace is not None:
            trace.append({"event": "timed_enter", "function": func.__name__})

        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return float(result)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if trace is not None:
                trace.append(
                    {
                        "event": "timed_exit",
                        "function": func.__name__,
                        "elapsed_ms": elapsed_ms,
                    }
                )

    return wrapper


def with_tag(tag: str) -> Callable[[Callable[..., float]], Callable[..., float]]:
    """参数化装饰器：调用前后标记业务标签。"""

    def decorator(func: Callable[..., float]) -> Callable[..., float]:
        @wraps(func)
        def wrapper(*args: object, **kwargs: object) -> float:
            kw = cast(dict[str, object], kwargs)
            trace = _extract_trace(kw)
            if trace is not None:
                trace.append({"event": "with_tag_enter", "tag": tag, "function": func.__name__})
            try:
                result = func(*args, **kwargs)
                return float(result)
            finally:
                if trace is not None:
                    trace.append({"event": "with_tag_exit", "tag": tag, "function": func.__name__})

        return wrapper

    return decorator


class CountCalls:
    """类装饰器：统计包装函数被调用次数。"""

    call_count: int

    def __init__(self, func: Callable[..., float]) -> None:
        self.func = func
        self.call_count = 0
        # 让实例暴露原函数元信息（__name__/__doc__ 等），便于教学观察。
        update_wrapper(self, func)

    def __call__(self, *args: object, **kwargs: object) -> float:
        self.call_count += 1
        kw = cast(dict[str, object], kwargs)
        trace = _extract_trace(kw)
        if trace is not None:
            trace.append(
                {
                    "event": "count_calls_enter",
                    "function": getattr(self, "__name__", self.func.__name__),
                    "call_count": self.call_count,
                }
            )
        try:
            result = self.func(*args, **kwargs)
            return float(result)
        finally:
            if trace is not None:
                trace.append(
                    {
                        "event": "count_calls_exit",
                        "function": getattr(self, "__name__", self.func.__name__),
                        "call_count": self.call_count,
                    }
                )


def _build_decorated_total_calculator() -> CountCalls:
    # 叠加顺序固定：
    # @CountCalls -> @with_tag("billing") -> @timed
    # 运行时进入顺序：count_calls_enter -> with_tag_enter -> timed_enter -> 函数体
    # 退出顺序相反（LIFO）。
    @CountCalls
    @with_tag("billing")
    @timed
    def calc_invoice_total(
        subtotal: float,
        tax_rate: float,
        *,
        discount: float = 0.0,
        _trace: list[TraceEvent] | None = None,
    ) -> float:
        taxable_amount = subtotal - discount
        total = taxable_amount * (1.0 + tax_rate)
        if _trace is not None:
            _trace.append(
                {
                    "event": "decorated_function_body",
                    "subtotal": subtotal,
                    "discount": discount,
                    "tax_rate": tax_rate,
                    "total": total,
                }
            )
        return round(total, 2)

    return calc_invoice_total


def run_demo() -> DemoResult:
    """运行固定案例，展示 decorator 链路的可观测 trace。"""
    trace: list[TraceEvent] = []
    calculator = _build_decorated_total_calculator()

    total_1 = calculator(100.0, 0.13, discount=5.0, _trace=trace)
    total_2 = calculator(80.0, 0.13, _trace=trace)

    final_answer = f"已完成 decorator 演示：第一次总额 {total_1:.2f}，第二次总额 {total_2:.2f}，累计调用 {calculator.call_count} 次。"
    trace.append({"event": "model_final_answer", "content": final_answer})
    return {"final_answer": final_answer, "trace": trace}


def _main() -> None:
    result = run_demo()

    print("=== TRACE ===")
    for index, event in enumerate(result["trace"], start=1):
        print(f"[{index}]")
        print(json.dumps(event, ensure_ascii=False, indent=2))

    print("\n=== FINAL ANSWER ===")
    print(result["final_answer"])


if __name__ == "__main__":
    _main()
