import types
from transformers import CONFIG_MAPPING, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, MODEL_NAMES_MAPPING, TOKENIZER_MAPPING
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, BertConverter, XLMRobertaConverter
from transformers.models.auto.modeling_auto import _BaseAutoModelClass, auto_class_update

from .models.layoutlmv2 import (
    LayoutLMv2Config,
    LayoutLMv2ForRelationExtraction,
    LayoutLMv2ForTokenClassification,
    LayoutLMv2Tokenizer,
    LayoutLMv2TokenizerFast,
)
from .models.layoutxlm import (
    LayoutXLMConfig,
    LayoutXLMForRelationExtraction,
    LayoutXLMForTokenClassification,
    LayoutXLMTokenizer,
    LayoutXLMTokenizerFast,
)


CONFIG_MAPPING.update([("layoutlmv2", LayoutLMv2Config), ("layoutxlm", LayoutXLMConfig)])
MODEL_NAMES_MAPPING.update([("layoutlmv2", "LayoutLMv2"), ("layoutxlm", "LayoutXLM")])
TOKENIZER_MAPPING.update(
    [
        (LayoutLMv2Config, (LayoutLMv2Tokenizer, LayoutLMv2TokenizerFast)),
        (LayoutXLMConfig, (LayoutXLMTokenizer, LayoutXLMTokenizerFast)),
    ]
)
SLOW_TO_FAST_CONVERTERS.update({"LayoutLMv2Tokenizer": BertConverter, "LayoutXLMTokenizer": XLMRobertaConverter})
MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.update(
    [
        (LayoutLMv2Config, LayoutLMv2ForTokenClassification), (LayoutXLMConfig, LayoutXLMForTokenClassification),
        (LayoutLMv2Config, LayoutLMv2ForRelationExtraction), (LayoutXLMConfig, LayoutXLMForRelationExtraction)
     ]
)

cls = types.new_class("AutoModelForTokenClassification", (_BaseAutoModelClass,))
cls._model_mapping = MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING
cls.__name__ = "AutoModelForTokenClassification" 
AutoModelForTokenClassification = auto_class_update(cls, head_doc="token classification")

cls = types.new_class("AutoModelForRelationExtraction", (_BaseAutoModelClass,))
cls._model_mapping = MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING
cls.__name__ = "AutoModelForRelationExtraction"
AutoModelForRelationExtraction = auto_class_update(cls, head_doc="relation extraction")
