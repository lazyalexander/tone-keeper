from tone_keeper.text.length import cjk_len
from tone_keeper.text.segment import split_units

A = "今天早上上班路上又堵成这样真的让人无语了啊"
B = "刚喝完一杯咖啡整个人还是很困根本不想动了"


def test_does_not_merge_short_neighbors():
    left = "短句甲乙丙丁戊"
    right = "短句己庚辛壬癸"
    assert cjk_len(left) < 20
    assert cjk_len(right) < 20
    text = f"{left}。{right}。"
    assert split_units(text, min_chars=20, max_chars=200) == []


def test_blank_line_is_a_cut():
    units = split_units(f"{A}\n\n{B}", min_chars=20, max_chars=200)
    assert units == [A, B]


def test_sentence_split_keeps_pieces_separate():
    units = split_units(f"{A}。{B}。", min_chars=20, max_chars=200)
    assert len(units) == 2
    assert A in units[0]
    assert B in units[1]


def test_long_piece_is_hard_cut_not_merged_back():
    text = "甲" * 250
    units = split_units(text, min_chars=20, max_chars=200)
    assert len(units) == 2
    assert cjk_len(units[0]) == 200
    assert cjk_len(units[1]) == 50
    assert "".join(units) == text


def test_split_never_joins_two_sentences():
    text = "第一句足够长可以留下作为样本看。第二句同样足够长可以单独留下看。"
    units = split_units(text, min_chars=10, max_chars=200)
    assert len(units) == 2
    assert all("第一句" not in u or "第二句" not in u for u in units)
