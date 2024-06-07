# coding=utf-8
from transformers.models.layoutlm.tokenization_layoutlm import LayoutLMTokenizer
from transformers.utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/layoutlmv2-base-uncased": "https://huggingface.co/microsoft/layoutlmv2-base-uncased/resolve/main/vocab.txt",
        "microsoft/layoutlmv2-large-uncased": "https://huggingface.co/microsoft/layoutlmv2-large-uncased/resolve/main/vocab.txt",
    }
}


PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/layoutlmv2-base-uncased": 512,
    "microsoft/layoutlmv2-large-uncased": 512,
}


PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/layoutlmv2-base-uncased": {"do_lower_case": True},
    "microsoft/layoutlmv2-large-uncased": {"do_lower_case": True},
}


class LayoutLMv2Tokenizer(LayoutLMTokenizer):
    r"""
    Constructs a LayoutLMv2 tokenizer.

    :class:`~transformers.LayoutLMv2Tokenizer is identical to :class:`~transformers.BertTokenizer` and runs end-to-end
    tokenization: punctuation splitting + wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizer` for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, model_max_length=512, **kwargs):
        super().__init__(model_max_length=model_max_length, **kwargs)





# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
The special tokens of T5Tokenizer is hard-coded with <extra_id_{}>
Created another class UDOPTokenizer extending it to add special visual tokens like <loc_{}>
<extra_l_id_{}> and </extra_l_id_{}> for LayoutModeling task
<extra_t_id_{}> and </extra_t_id_{}> for VisualTextRecognition
<extra_ul2_id_{}> for MaskedLanguageModeling UL2
<other_{} for JointTextLayoutRconstruction
"""

import re

import sentencepiece as spm

from transformers import T5Tokenizer
from transformers import PreTrainedTokenizer


class IDPFormerTokenizer(T5Tokenizer):
    def __init__(
            self,
            vocab_file,
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            extra_ids=100,
            loc_extra_ids=4096,
            pixel_extra_ids=16,
            other_extra_ids=200,
            extra_ul2_ids=200,
            additional_special_tokens=[],
            sp_model_kwargs=None,
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

        PreTrainedTokenizer.__init__(
            self,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        self.vocab_file = vocab_file
        self._extra_ids = extra_ids
        self._loc_extra_ids = loc_extra_ids
        self._pixel_extra_ids = pixel_extra_ids
        self._other_extra_ids = other_extra_ids
        self._extra_ul2_ids = extra_ul2_ids

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

        self._vocab_size = sum((
            self.sp_model.get_piece_size(),
            self._extra_ids * 5,
            self._loc_extra_ids * 2,
            self._pixel_extra_ids * 2,
            self._other_extra_ids,
            self._extra_ul2_ids,
        ))
        self.vocab_size = self._vocab_size

        self._special_tokens_mapping = {}

        vocab_offset = self._vocab_size - 1
        for prefix, n_extra_ids in [
                ("<extra_ul2_id_", self._extra_ul2_ids),
                ("<other_", self._other_extra_ids),
                ("</pixel_", self._pixel_extra_ids),
                ("<pixel_", self._pixel_extra_ids),
                ("</loc_", self._loc_extra_ids),
                ("<loc_", self._loc_extra_ids),
                ("</extra_t_id_", self._extra_ids),
                ("<extra_t_id_", self._extra_ids),
                ("</extra_l_id_", self._extra_ids),
                ("<extra_l_id_", self._extra_ids),
                ("<extra_id_", self._extra_ids),
        ]:
            for num in range(n_extra_ids):
                self._special_tokens_mapping[f"{prefix}{num}>"] = vocab_offset - num
            vocab_offset -= n_extra_ids

    @property
    def vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        try:
            return self._special_tokens_mapping[token]
        except KeyError:
            return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        vocab_offset = self.vocab_size - 1
        for prefix, n_extra_ids in [
            ("<extra_ul2_id_", self._extra_ul2_ids),
            ("<other_", self._other_extra_ids),
            ("</pixel_", self._pixel_extra_ids),
            ("<pixel_", self._pixel_extra_ids),
            ("</loc_", self._loc_extra_ids),
            ("<loc_", self._loc_extra_ids),
            ("</extra_t_id_", self._extra_ids),
            ("<extra_t_id_", self._extra_ids),
            ("</extra_l_id_", self._extra_ids),
            ("<extra_l_id_", self._extra_ids),
            ("<extra_id_", self._extra_ids),
        ]:
            if index > vocab_offset - n_extra_ids:
                index_vocab = vocab_offset - index
                return f"{prefix}{index_vocab}>"
            vocab_offset -= n_extra_ids
        return self.sp_model.IdToPiece(index)
