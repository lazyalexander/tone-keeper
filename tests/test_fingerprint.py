from tone_keeper.fingerprint.distance import style_distance
from tone_keeper.fingerprint.profile import build_profile, profile_from_dict


ORAL = [
    "今天早上上班路上又堵成这样真的让人无语了啊",
    "刚喝完一杯咖啡整个人还是很困根本不想动了",
    "这周的deadline全叠在一起我有点想直接摆了",
    "晚饭就随便对付一下吧明天再好好吃上一顿",
    "刚看到那个消息我直接笑死了到底是谁写的",
]

FORMAL = [
    "根据有关规定应当依法予以处理并及时报告",
    "现将有关情况函告如下请予以配合落实",
    "本次会议审议并通过了相关工作安排",
]


def test_same_text_near_zero_when_profile_is_itself():
    profile = build_profile([ORAL[0]])
    assert style_distance(ORAL[0], profile) < 1e-8


def test_oral_closer_to_oral_profile_than_formal():
    profile = build_profile(ORAL)
    oral_d = sum(style_distance(text, profile) for text in ORAL[1:]) / (len(ORAL) - 1)
    formal_d = sum(style_distance(text, profile) for text in FORMAL) / len(FORMAL)
    assert oral_d < formal_d


def test_profile_roundtrip():
    profile = build_profile(ORAL)
    restored = profile_from_dict(profile.to_dict())
    assert restored.vocab == profile.vocab
    assert style_distance(ORAL[0], restored) == style_distance(ORAL[0], profile)
