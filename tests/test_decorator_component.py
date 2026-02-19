from decorator.component import CountCalls, run_demo, timed, with_tag


def test_timed_preserves_metadata_and_records_elapsed_ms() -> None:
    @timed
    def plus_one(x: float, *, _trace=None) -> float:
        return x + 1

    trace = []
    result = plus_one(1.0, _trace=trace)

    assert plus_one.__name__ == "plus_one"
    assert result == 2.0
    assert trace[0]["event"] == "timed_enter"
    assert trace[1]["event"] == "timed_exit"
    assert float(trace[1]["elapsed_ms"]) >= 0.0


def test_with_tag_records_enter_and_exit_with_tag() -> None:
    @with_tag("billing")
    def calc(x: float, *, _trace=None) -> float:
        return x * 2

    trace = []
    result = calc(3.0, _trace=trace)

    assert result == 6.0
    assert trace[0]["event"] == "with_tag_enter"
    assert trace[0]["tag"] == "billing"
    assert trace[1]["event"] == "with_tag_exit"
    assert trace[1]["tag"] == "billing"


def test_count_calls_tracks_call_count() -> None:
    @CountCalls
    def identity(x: float, *, _trace=None) -> float:
        return x

    identity(1.0)
    identity(2.0)

    assert identity.call_count == 2


def test_stacked_decorators_have_expected_event_order() -> None:
    @CountCalls
    @with_tag("billing")
    @timed
    def stacked(subtotal: float, tax: float, *, _trace=None) -> float:
        if _trace is not None:
            _trace.append({"event": "decorated_function_body", "subtotal": subtotal})
        return subtotal * (1 + tax)

    trace = []
    stacked(100.0, 0.1, _trace=trace)
    events = [item["event"] for item in trace]

    assert events == [
        "count_calls_enter",
        "with_tag_enter",
        "timed_enter",
        "decorated_function_body",
        "timed_exit",
        "with_tag_exit",
        "count_calls_exit",
    ]


def test_run_demo_returns_protocol_and_final_answer_event() -> None:
    result = run_demo()

    assert "final_answer" in result
    assert "trace" in result
    events = [item["event"] for item in result["trace"]]
    assert "model_final_answer" in events


def test_decorators_work_without_trace_channel() -> None:
    @CountCalls
    @with_tag("billing")
    @timed
    def stacked_no_trace(x: float) -> float:
        return x + 1

    assert stacked_no_trace(1.0) == 2.0
    assert stacked_no_trace.call_count == 1
