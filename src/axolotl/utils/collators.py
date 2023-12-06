"""
DataCollator for axolotl to pad labels and position_ids for packed sequences
"""
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels and position_ids

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    position_pad_token_id: int = 0
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        labels = None
        if return_tensors is None:
            return_tensors = self.return_tensors

        # def recode(l):
        #     rv = []
        #     for s in l:
        #         # Convert <0xNN> to \NN
        #         m = s.split("<0x")
        #         if len(m) == 1:
        #             rv.append(s)
        #             continue
        #         r = m[0]
        #         for i in range(1, len(m)):
        #             if not (len(m[i]) > 2 and m[i][2] == ">"):
        #                 print(f"wtf is up with {m[i]}")
        #             assert len(m[i]) > 2 and m[i][2] == ">"
        #             r += chr(int(m[i][:2], 16)) + m[i][3:]
        #         rv.append(r)
        #     return rv

        def decode_tokenized(data):
            t = self.tokenizer.sp_model
            if "input_ids" in data: data = data["input_ids"]
            data = [d["input_ids"] if "input_ids" in d else d for d in data]
            rvf = []
            for d in data:
                rv = []
                for id in d:
                    # print type of id
                    id = id.item()
                    if t.is_unknown(id):
                        # ignore
                        continue
                    if t.is_control(id):
                        # add newline, but not if we have a newline already
                        if len(rv) > 0 and rv[-1] != 10:
                            rv.append(10)
                        continue
                    if t.is_byte(id):
                        # <U+XX> tokens (which may be invalid UTF-8)
                        piece = t.id_to_piece(id)
                        if len(piece) != 6:
                            print(f"bad piece {piece}")
                            assert len(piece) == 6
                        byte_value = int(piece[3:-1], 16)
                        rv.append(byte_value)
                        continue
                    text = t.id_to_piece(id).replace("\u2581", " ").encode("utf-8")
                    # if text.startswith("▁"):
                    #     text = text[1:]
                    try:
                        rv.extend(text)
                    except:
                        print(f"failed to extend {text} to {rv} in {d}")
                        raise
                try:
                    rvf.append(str(bytes(rv), 'utf-8'))
                except:
                    rvf.append("".join(self.tokenizer.convert_ids_to_tokens(d)).replace("▁", " "))
                    # print(f"failed to convert {rv} to utf-8 from {d}")
                    # raise
            return [f"[{len(d)}] " + d for d in rvf]
            # return [f"[{len(d)}] " + ("".join(recode(self.tokenizer.convert_ids_to_tokens(d))).replace("▁", " ")) for d in data]

        print("\n- ".join(decode_tokenized(features)))

        for feature_name, pad_token_id in [
            ("labels", self.label_pad_token_id),
            ("position_ids", self.position_pad_token_id),
        ]:
            feat = (
                [feature[feature_name] for feature in features]
                if feature_name in features[0].keys()
                else None
            )
            labels = feat if feat and feature_name == "labels" else labels
            # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
            # same length to return tensors.
            if feat is not None:
                max_feature_length = max(len(l) for l in feat)  # noqa: E741
                if self.pad_to_multiple_of is not None:
                    max_feature_length = (
                        (max_feature_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                    )

                padding_side = self.tokenizer.padding_side
                for feature in features:
                    remainder = [pad_token_id] * (
                        max_feature_length - len(feature[feature_name])
                    )
                    if isinstance(feature[feature_name], list):
                        feature[feature_name] = (
                            feature[feature_name] + remainder
                            if padding_side == "right"
                            else remainder + feature[feature_name]
                        )
                    elif padding_side == "right":
                        feature[feature_name] = np.concatenate(
                            [feature[feature_name], remainder]
                        ).astype(np.int64)
                    else:
                        feature[feature_name] = np.concatenate(
                            [remainder, feature[feature_name]]
                        ).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=features["labels"]
            )
            features["decoder_input_ids"] = decoder_input_ids

        return features


@dataclass
class BatchSamplerDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    """
    Collator for multipack specific to the using the BatchSampler
    """

    def __call__(self, features, return_tensors=None):
        chunked_data = {}
        for feature in features[0].keys():
            if feature == "length":
                continue
            if feature == "attention_mask":
                arrays = [
                    (1) * np.array(item[feature])
                    for item in features
                    if feature in item
                ]
                chunked_data[feature] = np.concatenate(arrays)
            else:
                arrays = [
                    np.array(item[feature]) for item in features if feature in item
                ]
                chunked_data[feature] = np.concatenate(arrays)
        features = [chunked_data]
        return super().__call__(features, return_tensors=return_tensors)
