import ast
from pathlib import Path


def test_sft_wrapper_does_not_import_gates_or_fingerprint():
    src = Path("tone_keeper/lora/train.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    joined = " ".join(modules)
    assert "fingerprint" not in joined
    assert "infer" not in joined
    assert "gates" not in joined
