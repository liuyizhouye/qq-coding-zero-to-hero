import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.getenv("RUN_LLM_API_TESTS") == "1":
        return

    skip_online = pytest.mark.skip(reason="set RUN_LLM_API_TESTS=1 to run online API tests")
    for item in items:
        if "online" in item.keywords:
            item.add_marker(skip_online)
