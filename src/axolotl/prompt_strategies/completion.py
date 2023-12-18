"""
Basic completion text
"""
import logging

# import random
from collections import defaultdict
from typing import Any, Dict, Generator, Optional, Tuple

from axolotl.prompt_tokenizers import InstructionPromptTokenizingStrategy


class CompletionPromptTokenizingStrategy(InstructionPromptTokenizingStrategy):
    """
    Tokenizing strategy for Completion prompts.
    """

    _field: str = "text"

    def __init__(self, *args, max_length=None, align_samples=False, **kwargs):
        super().__init__(*args, **kwargs)
        if max_length is not None:
            self.max_length = max_length
        self.align_samples = align_samples
        self.min_sample_len = 1
        self.overlap_len = 0
        self.discard_portion = 0.0  # 1.0 means discard everything, 0.5 means discard half of the samples, etc.

    @property
    def supports_batched(self):
        return True

    @property
    def field(self) -> str:
        return self._field

    @field.setter
    def field(self, new_field: str):
        self._field = new_field

    def parse_instruction_fields(self, prompt) -> Tuple[str, str, str]:
        return (
            prompt[self.field],
            "",
            "",
        )

    def tokenize_prompt(self, prompt):
        res = defaultdict(lambda: [])
        feature_names = list(prompt.keys())
        for row in zip(*prompt.values()):
            prompt_row = dict(zip(feature_names, row))
            (
                instruction,
                _,
                _,
            ) = self.parse_instruction_fields(prompt_row)

            full_prompt = self._build_full_prompt(instruction, None, None)
            #  TODO: make add_eos_token an option
            tokenized_full_prompt = self._tokenize(full_prompt, add_eos_token=False)
            steps = self.sequence_len - self.overlap_len
            if steps < 1:
                raise ValueError("Sequence length must be greater than overlap length")

            # The case of a completion task given a smaller initial text blurb is common,
            # e.g. when tasked to write the starting point of a text, whereas a completion
            # task given the last few tokens of a text (followed by padding) is not.
            # Ideally, we want to align each text so that it begins with (as necessary)
            # right-padded tokens in the first sample, with no more padding required.
            if self.align_samples:
                for key, val in tokenized_full_prompt.items():
                    valsteps = (len(val) - self.sequence_len) // steps
                    left_padding = (len(val) - self.sequence_len) - valsteps * steps
                    if left_padding > 0:
                        res[key].append(val[0 : max(self.min_sample_len, left_padding)])
                    for i in range(
                        left_padding, len(val) + 1 - self.sequence_len, steps
                    ):
                        res[key].append(val[i : i + self.sequence_len])
            else:
                for key, val in tokenized_full_prompt.items():
                    for i in range(0, len(val), steps):
                        res[key].append(val[i : i + self.sequence_len])

        # # Discard a portion of the samples
        # if self.discard_portion > 0.0:
        #     # Shuffle samples
        #     random.shuffle(res[key])

        #     for key, val in res.items():
        #         res[key] = val[:int(len(val) * (1.0 - self.discard_portion))]

        return dict(res)

    def _build_full_prompt(
        self, instruction, input, response
    ):  # pylint: disable=redefined-builtin
        return next(iter(self.prompter.build_prompt(instruction, input, response)))


class CompletionPrompter:
    """
    Prompter for completion
    """

    def build_prompt(
        self,
        instruction: str,
        input=None,  # pylint: disable=redefined-builtin, unused-argument
        output=None,  # pylint: disable=unused-argument
    ) -> Generator[str, None, None]:
        yield instruction


def load(tokenizer, cfg, ds_cfg: Optional[Dict[str, Any]] = None):
    if tokenizer.padding_side != "left":
        logging.warning(
            f"warning: tokenizer padding side is {tokenizer.padding_side}; use 'left' padding to enable completion alignment"
        )

    strat = CompletionPromptTokenizingStrategy(
        CompletionPrompter(),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
        max_length=cfg.sequence_len * 64,
        align_samples=tokenizer.padding_side == "left",
    )
    if ds_cfg:
        if "field" in ds_cfg:
            strat.field = ds_cfg["field"]
        if "overlap_len" in ds_cfg:
            strat.overlap_len = ds_cfg["overlap_len"]
        if "min_sample_len" in ds_cfg:
            strat.min_sample_len = ds_cfg["min_sample_len"]
        if "discard_portion" in ds_cfg:
            strat.discard_portion = ds_cfg["discard_portion"]

    return strat
