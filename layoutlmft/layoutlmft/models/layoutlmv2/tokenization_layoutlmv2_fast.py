# coding=utf-8
from transformers.models.layoutlm.tokenization_layoutlm_fast import LayoutLMTokenizerFast
from transformers.utils import logging

from .tokenization_layoutlmv2 import LayoutLMv2Tokenizer, IDPFormerTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/layoutlmv2-base-uncased": "https://huggingface.co/microsoft/layoutlmv2-base-uncased/resolve/main/vocab.txt",
        "microsoft/layoutlmv2-large-uncased": "https://huggingface.co/microsoft/layoutlmv2-large-uncased/resolve/main/vocab.txt",
    },
    "tokenizer_file": {
        "microsoft/layoutlmv2-base-uncased": "https://huggingface.co/microsoft/layoutlmv2-base-uncased/resolve/main/tokenizer.json",
        "microsoft/layoutlmv2-large-uncased": "https://huggingface.co/microsoft/layoutlmv2-large-uncased/resolve/main/tokenizer.json",
    },
}


PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/layoutlmv2-base-uncased": 512,
    "microsoft/layoutlmv2-large-uncased": 512,
}


PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/layoutlmv2-base-uncased": {"do_lower_case": True},
    "microsoft/layoutlmv2-large-uncased": {"do_lower_case": True},
}


class LayoutLMv2TokenizerFast(LayoutLMTokenizerFast):
    r"""
    Constructs a "Fast" LayoutLMv2Tokenizer.

    Refer to superclass :class:`~transformers.BertTokenizerFast` for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    slow_tokenizer_class = LayoutLMv2Tokenizer

    def __init__(self, model_max_length=512, **kwargs):
        super().__init__(model_max_length=model_max_length, **kwargs)





import os
from shutil import copyfile
from typing import List, Optional, Tuple

from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import is_sentencepiece_available, logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}


class IDPFormerTokenizerFast(PreTrainedTokenizerFast):
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = IDPFormerTokenizer

    prefix_tokens: List[int] = []

    def __init__(
            self,
            vocab_file=None,
            tokenizer_file=None,
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            extra_ids=100,
            loc_extra_ids=4096,
            pixel_extra_ids=16,
            other_extra_ids=200,
            extra_ul2_ids=200,
            additional_special_tokens=None,
            **kwargs
    ):
        # Add extra_ids to the special token list
        if extra_ids > 0:
            if "<extra_id_0>" not in additional_special_tokens:
                additional_special_tokens = ["<extra_id_{}>".format(i) for i in range(extra_ids)]
            if "<extra_l_id_0>" not in additional_special_tokens:
                additional_special_tokens.extend(["<extra_l_id_{}>".format(i) for i in range(extra_ids)])
                additional_special_tokens.extend(["</extra_l_id_{}>".format(i) for i in range(extra_ids)])
            if "<extra_t_id_0>" not in additional_special_tokens:
                additional_special_tokens.extend(["<extra_t_id_{}>".format(i) for i in range(extra_ids)])
                additional_special_tokens.extend(["</extra_t_id_{}>".format(i) for i in range(extra_ids)])

        # elif extra_ids > 0 and additional_special_tokens is not None:
        #     extra_ids = 0

        if loc_extra_ids > 0 and not "<loc_0>" in additional_special_tokens:
            additional_special_tokens.extend(["<loc_{}>".format(i) for i in range(loc_extra_ids)])
            additional_special_tokens.extend(["</loc_{}>".format(i) for i in range(loc_extra_ids)])

        if pixel_extra_ids > 0 and not "<pixel_0>" in additional_special_tokens:
            additional_special_tokens.extend(["<pixel_{}>".format(i) for i in range(pixel_extra_ids)])
            additional_special_tokens.extend(["</pixel_{}>".format(i) for i in range(pixel_extra_ids)])

        if other_extra_ids > 0 and not "<other_0>" in additional_special_tokens:
            additional_special_tokens.extend(["<other_{}>".format(i) for i in range(other_extra_ids)])

        if extra_ul2_ids > 0 and not "<extra_ul2_id_0>" in additional_special_tokens:
            additional_special_tokens.extend(["<extra_ul2_id_{}>".format(i) for i in range(extra_ul2_ids)])

        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self.vocab_file = vocab_file
        self.can_save_slow_tokenizer = False if not self.vocab_file else True
        self._extra_ids = extra_ids
        self._loc_extra_ids = loc_extra_ids
        self._pixel_extra_ids = pixel_extra_ids
        self._other_extra_ids = other_extra_ids
        self._extra_ul2_ids = extra_ul2_ids

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
            logger.info(f"Copy vocab file to {out_vocab_file}")

        return (out_vocab_file,)

    def get_sentinel_tokens(self):
        return self.additional_special_tokens or []

    def get_sentinel_token_ids(self):
        return [self.convert_tokens_to_ids(token) for token in self.get_sentinel_tokens()]
        
