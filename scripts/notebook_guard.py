#!/usr/bin/env python3
import json
from pathlib import Path

KERNEL_NAME = "qq-coding-zero-to-hero"
KERNEL_DISPLAY_NAME = "Python (qq-coding-zero-to-hero)"
NOTEBOOKS = [
    Path("function_call/walkthrough.ipynb"),
    Path("mcp/walkthrough.ipynb"),
    Path("skills/walkthrough.ipynb"),
    Path("decorator/walkthrough.ipynb"),
    Path("web_data_flow/walkthrough.ipynb"),
    Path("react_architecture/walkthrough.ipynb"),
    Path("rag/walkthrough.ipynb"),
]


def _normalize_notebook(path: Path) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    changed = False

    metadata = data.setdefault("metadata", {})
    desired_kernelspec = {
        "display_name": KERNEL_DISPLAY_NAME,
        "language": "python",
        "name": KERNEL_NAME,
    }
    if metadata.get("kernelspec") != desired_kernelspec:
        metadata["kernelspec"] = desired_kernelspec
        changed = True

    language_info = metadata.setdefault("language_info", {})
    if language_info.get("name") != "python":
        language_info["name"] = "python"
        changed = True

    for index, cell in enumerate(data.get("cells", []), start=1):
        if not cell.get("id"):
            cell["id"] = f"cell-{index:02d}"
            changed = True
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True

    if changed:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return changed


def main() -> int:
    changed_paths: list[str] = []
    for notebook in NOTEBOOKS:
        if not notebook.exists():
            continue
        if _normalize_notebook(notebook):
            changed_paths.append(str(notebook))

    if changed_paths:
        print("normalized notebooks:")
        for path in changed_paths:
            print(f"  - {path}")
    else:
        print("notebooks already normalized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
