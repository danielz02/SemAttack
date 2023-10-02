import os.path
from copy import deepcopy
from typing import List, Union, Optional, Dict

import numpy as np
import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizer, AutoTokenizer

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any

from attacks.tokenization import LABEL_CANDIDATE, GLUE_TASK_TO_KEYS, TASK_DESCRIPTION

# Updated based on https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md More specific
# instructions here: https://github.com/lm-sys/FastChat/blob/00d9e6675bdff60be6603ffff9313b1d797d2e3e/fastchat
# /conversation.py#L115-L124 And here:
# https://github.com/lm-sys/FastChat/blob/00d9e6675bdff60be6603ffff9313b1d797d2e3e/fastchat/conversation.py#L36


IGNORE_INDEX = -100


class SeparatorStyle(Enum):
    """Different separator style."""
    LLAMA2 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The system prompt
    system: Union[str, None]
    # Two roles
    roles: List[str]
    # All messages. Each item is (role, message).
    messages: List[List[str]]
    # The number of few shot examples
    offset: int
    # Separators
    sep_style: SeparatorStyle
    sep: str
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: str = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        seps = [self.sep, self.sep2]
        ret = ""
        for i, (role, message) in enumerate(self.messages):
            if message:
                if i == 0:
                    ret += self.system + message
                else:
                    ret += role + " " + message + seps[i % 2]
            else:
                ret += role
        return ret

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])


def get_preprocess_function(task_name: str, tokenizer: PreTrainedTokenizer, ):
    assert task_name in LABEL_CANDIDATE
    sentence1_key, sentence2_key = GLUE_TASK_TO_KEYS[task_name]
    # FIXME: New special tokens assigned id 0?
    tokenizer.add_special_tokens({"additional_special_tokens": ["<l>", "<i>", "</i>", "<j>", "<k>"]})

    def preprocess_function(example):
        for i, label in enumerate(LABEL_CANDIDATE[task_name]):
            sentence1 = example[sentence1_key]
            message = f"{TASK_DESCRIPTION[task_name]} {sentence1_key}: {sentence1}"
            if sentence2_key:
                sentence2 = example[sentence2_key]
                message = f"{message}<j> {sentence2_key}: <k>{sentence2}"
            message = f"{message}".replace('sentence1', 'premise').replace('sentence2', 'hypothesis')
            conversation = Conversation(
                name="llama-2",
                system="<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
                "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
                "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
                "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
                "If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n",
                roles=("[INST]", "[/INST]"),
                messages=(["[INST]", f"<i>{message}</i>"], ["[/INST]", f"<l>{label}"]),
                offset=0,
                sep_style=SeparatorStyle.LLAMA2,
                sep=" ",
                sep2=" </s><s>",
                stop_token_ids=[2],
            )
            tokens = tokenizer.tokenize(conversation.get_prompt())
            # TODO: Double check tokenizer. Currently using a temporary fix
            if "<s>" in tokens[0]:
                tokens[0] = tokens[0].strip("▁")
            if "</s>" in tokens[-1]:
                tokens[-1] = tokens[-1].strip("▁")

            input_start_idx = tokens.index("<i>")
            tokens.remove("<i>")
            if sentence2_key:
                input1_end_idx = tokens.index("<j>")
                tokens.remove("<j>")
                input2_start_idx = tokens.index("<k>")
                tokens.remove("<k>")
            else:
                input1_end_idx = None
                input2_start_idx = torch.nan
            input_end_idx = tokens.index("</i>")
            tokens.remove("</i>")
            label_start_idx = tokens.index("<l>")
            tokens.remove("<l>")
            token_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens), dtype=torch.long)

            instruction_token_ids = token_ids[:input_start_idx].clone()
            input_token_ids = token_ids[input_start_idx:input_end_idx].clone()
            response_header_token_ids = token_ids[input_end_idx:label_start_idx].clone()
            response_token_ids = token_ids[label_start_idx:].clone()

            if i == example["label"]:
                example["input_ids"] = token_ids
            example[f"instruction_token_ids"] = instruction_token_ids
            example[f"input_token_ids"] = input_token_ids
            example[f"response_header_token_ids"] = response_header_token_ids
            example[f"{label}_response_token_ids"] = response_token_ids
            example["label_names"] = LABEL_CANDIDATE[task_name]

            example["input_start_idx"] = input_start_idx
            example["input1_end_idx"] = input1_end_idx if input1_end_idx else input_end_idx
            example["input2_start_idx"] = input2_start_idx
            example["input_end_idx"] = input_end_idx
            example["label_start_idx"] = label_start_idx

        example["target"] = get_attack_target(example, task_name)

        return example

    return preprocess_function


def get_attack_target(x, task):
    labels = LABEL_CANDIDATE[task]

    if len(labels) == 3:
        if x["label"] == 2:
            target = 0
        elif x["label"] == 0:
            target = 2
        else:
            if np.random.uniform() < 0.5:
                target = 0
            else:
                target = 2
    elif len(labels) == 2:
        if x["label"] == 0:
            target = 1
        else:
            target = 0
    else:
        raise Exception('Unknown number of labels.')

    return target


def main():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", cache_dir="./.cache")

    for task in TASK_DESCRIPTION.keys():
        if task == 'mnli':
            split = 'validation_matched'
        elif task == 'mnli-mm':
            split = 'validation_mismatched'
        else:
            split = "validation"
        test_data = load_dataset("glue", task.replace("-mm", ""), cache_dir="./.cache/", split=split)
        save_dir = f"./.cache/glue-preprocessed-benign/meta-llama/Llama-2-7b-chat-hf/{task}/"
        if os.path.exists(save_dir):
            print("Loading preprocessed results")
            test_data = test_data.load_from_disk(save_dir)
        else:
            print("Preprocessing results")
            test_data = test_data.map(get_preprocess_function(task, tokenizer), num_proc=16)
            test_data.save_to_disk(save_dir)

        print(test_data[0])


if __name__ == "__main__":
    main()
