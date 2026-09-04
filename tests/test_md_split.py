from tone_keeper.text.length import cjk_len
from tone_keeper.tools.md_split import split_markdown


def test_drops_headings_and_does_not_merge():
    doc = """# 标题一

这是第一段正文，长度必须超过二十个汉字才能留下来看。

# 标题二

这是第二段正文，同样需要足够长才不会被丢掉。
"""
    units = split_markdown(doc, min_chars=20, max_chars=200)
    assert len(units) == 2
    assert "标题" not in "".join(units)
    assert "第一段" in units[0]
    assert "第二段" in units[1]


def test_short_dialogue_dropped_not_merged():
    doc = """这是一段足够长的叙述，用来垫在对话前面避免整篇被丢掉。

“他在猎杀我们？”

这是对话后面另一段足够长的叙述，仍然单独成段。
"""
    units = split_markdown(doc, min_chars=20, max_chars=200)
    assert len(units) == 2
    assert all("猎杀" not in u for u in units)


def test_long_paragraph_is_cut_not_glued():
    long = "甲" * 250
    units = split_markdown(long, min_chars=20, max_chars=200)
    assert len(units) >= 2
    assert cjk_len(units[0]) == 200
    assert "".join(units) == long


def test_skips_fence_and_frontmatter():
    doc = """---
title: x
---

正文段落必须足够长这样才会被保留下来作为单元。

```
code should vanish
```

另一段同样足够长的正文跟在代码块后面单独存在。
"""
    units = split_markdown(doc, min_chars=20, max_chars=200)
    assert len(units) == 2
    assert all("code" not in u and "title" not in u for u in units)
