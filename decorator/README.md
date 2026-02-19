# decorator 模块

本模块通过“日志 + 计时”案例讲清 Python 装饰器的三种核心形态：

1. 函数装饰器：`timed`
2. 参数化装饰器：`with_tag(tag)`
3. 类装饰器：`CountCalls`

## 这个模块解决什么问题

当你要在不改业务函数核心逻辑的前提下，统一叠加“横切能力”（日志、计时、统计调用次数）时，装饰器是最直接、可复用的机制。

## 概念背后的原理

### 1) 函数装饰器

函数接收函数，返回一个新函数。新函数把原函数包起来，在调用前后插入额外逻辑。

### 2) 参数化装饰器

多一层函数：先接收配置参数，再返回真正装饰器。适合同一逻辑不同策略（例如不同日志 tag）。

### 3) 类装饰器

用对象保存状态（例如 `call_count`）。类实例实现 `__call__` 后即可像函数一样调用。

### 4) 装饰器执行顺序

本模块固定堆叠顺序：

```python
@CountCalls
@with_tag("billing")
@timed
```

运行时进入顺序是外层到内层，退出顺序相反（LIFO）：

- 进入：`count_calls_enter -> with_tag_enter -> timed_enter -> decorated_function_body`
- 退出：`timed_exit -> with_tag_exit -> count_calls_exit`

## 实现过程（对应 `component.py`）

1. `timed(func)`：记录耗时毫秒并写入 `timed_enter/timed_exit`。
2. `with_tag(tag)`：写入 `with_tag_enter/with_tag_exit`，展示装饰器工厂。
3. `CountCalls`：维护 `call_count` 并写入调用前后事件。
4. `_build_decorated_total_calculator()`：把三类装饰器叠加到同一个业务函数。
5. `run_demo()`：执行两次调用并返回统一协议：

```python
{
  "final_answer": str,
  "trace": list[dict[str, object]]
}
```

## Trace 事件说明

固定事件名：

- `count_calls_enter` / `count_calls_exit`
- `with_tag_enter` / `with_tag_exit`
- `timed_enter` / `timed_exit`
- `decorated_function_body`
- `model_final_answer`

## 运行方式

```bash
python decorator/component.py
```

## Notebook 学习重点

`walkthrough.ipynb` 按顺序展示：

1. 只看 `timed`
2. 只看 `with_tag("billing")`
3. 只看 `CountCalls`
4. 看三者叠加后的完整调用链
5. 看 `run_demo` 的最终 trace 与结论

## 常见问题

- 为什么用了 `_trace` 参数？
  - 用于教学可观测性，真实业务里可替换成日志系统。

- 缺失 `_trace` 会报错吗？
  - 不会。装饰器会自动降级，只执行功能不记录事件。

## 可扩展方向

1. 增加重试装饰器（retry）。
2. 增加结果缓存装饰器（cache）。
3. 增加异步函数装饰器（`async def`）。
