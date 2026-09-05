from tone_keeper.config import load, student_prompt


def test_load_default():
    cfg = load("configs/default.toml")
    assert cfg.unit.min_chars == 20
    assert cfg.unit.max_chars == 200
    assert "{A}" in cfg.prompts.student
    assert "扩写成更长的段落" in cfg.prompts.student
    assert "文风" in cfg.prompts.student
    assert "口语" not in cfg.prompts.student
    rendered = student_prompt(cfg.prompts.student, "HELLO")
    assert "HELLO" in rendered
    assert "{A}" not in rendered
    assert cfg.teacher.provider == "deepseek"
    assert cfg.teacher.model == "deepseek-v4-flash"
    assert "Qwen3-Embedding" in cfg.embed.model
