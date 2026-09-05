from rulers.v0_punct_func import PunctFuncV0
from rulers.v1_corpus_func import CorpusFuncV1

RULERS = {
    PunctFuncV0.name: PunctFuncV0,
    CorpusFuncV1.name: CorpusFuncV1,
}

__all__ = ["RULERS", "PunctFuncV0", "CorpusFuncV1"]
