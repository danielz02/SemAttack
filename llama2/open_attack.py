'''
This example code shows how to use the PWWS attack model to attack a customized sentiment analysis model.
'''
import os
import OpenAttack
import numpy as np
import datasets
from datasets import load_dataset

import json
import torch
from transformers import AutoTokenizer
from attacks.tokenization import LABEL_CANDIDATE, TASK_DESCRIPTION, GLUE_TASK_TO_KEYS, get_attack_target
import attacks.util as util
from attacks.open_attack import ZeroShotLlamaClassifier as AlpacaClassifier
from attacks.open_attack import get_dataset_mapping, get_dataset
from attacks.model import ZeroShotLlamaForSemAttack
from tokenization_llama2 import SeparatorStyle, Conversation
import ssl


ssl._create_default_https_context = ssl._create_unverified_context


class ZeroShotLlamaClassifier(AlpacaClassifier):
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.model = ZeroShotLlamaForSemAttack(args.model, args.cache_dir).to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<l>", "<i>", "</i>", "<j>", "<k>"]})
        self.fixed_sentence = None
        self.fixed_idx = self.args.fix_sentence

    def preprocess_function(self, sent):
        task_name = self.args.task
        assert task_name in LABEL_CANDIDATE
        sentence1_key, sentence2_key = GLUE_TASK_TO_KEYS[task_name]
        example = {}
        for i, label in enumerate(LABEL_CANDIDATE[task_name]):
            sentence1 = self.fixed_sentence if self.fixed_idx == 0 else sent
            message = f"{TASK_DESCRIPTION[task_name]} {sentence1_key}: {sentence1}"
            if sentence2_key:
                sentence2 = sent if self.fixed_idx == 0 else self.fixed_sentence
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
            tokens = self.tokenizer.tokenize(conversation.get_prompt())
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
            token_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens), dtype=torch.long).to(self.device)

            instruction_token_ids = token_ids[:input_start_idx].clone()
            input_token_ids = token_ids[input_start_idx:input_end_idx].clone()
            response_header_token_ids = token_ids[input_end_idx:label_start_idx].clone()
            response_token_ids = token_ids[label_start_idx:].clone()

            example[f"instruction_token_ids"] = instruction_token_ids
            example[f"input_token_ids"] = input_token_ids
            example[f"response_header_token_ids"] = response_header_token_ids
            example[f"{label}_response_token_ids"] = response_token_ids
            example["label_names"] = LABEL_CANDIDATE[task_name]

            example["input_start_idx"] = input_start_idx
            example["input_end_idx"] = input_end_idx
            example["label_start_idx"] = label_start_idx

        return example


def main():
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        for dev in physical_devices:
            tf.config.experimental.set_memory_growth(dev, True)
    except RuntimeError:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    args = util.get_args()
    args.output_dir = os.path.join(args.output_dir, 'openattack', args.attack, args.task, str(args.fix_sentence))
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda:0")
    victim = ZeroShotLlamaClassifier(args, device)

    dataset = get_dataset(args, model=victim)

    algorithm_to_attacker = {
        'textbugger': OpenAttack.attackers.TextBuggerAttacker,
        'textfooler': OpenAttack.attackers.TextFoolerAttacker,
        'sememepso': OpenAttack.attackers.PSOAttacker,
        'bertattack': OpenAttack.attackers.BERTAttacker,
        # 'bae': OpenAttack.attackers.BAEAttacker,
        # 'genetic': OpenAttack.attackers.GeneticAttacker,
        # 'pwws': OpenAttack.attackers.PWWSAttacker,
        # 'deepwordbug': OpenAttack.attackers.DeepWordBugAttacker,
    }
    print('Attack using', args.attack)
    attacker = algorithm_to_attacker[args.attack]()

    attack_eval = OpenAttack.AttackEval(attacker, victim)

    summary, results = attack_eval.eval(dataset, visualize=False, progress_bar=True)
    # attack_eval.eval(dataset, visualize=True, num_workers=16)  # TypeError: cannot pickle '_io.BufferedReader' object

    print('Saving results to {}'.format(args.output_dir))
    with open(os.path.join(args.output_dir, f"{args.attack}_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    with open(os.path.join(args.output_dir, f"{args.attack}_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)


if __name__ == "__main__":
    main()
