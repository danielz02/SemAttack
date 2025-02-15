import json
import os
import util
import copy
import torch
import joblib
import numpy as np
from tqdm import tqdm
from attacks.CW_attack import CarliniL2
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from model import ZeroShotLlamaForSemAttack
from tokenization import GLUE_TASK_TO_KEYS, LABEL_CANDIDATE


def transform(seq, tokenizer, unk_words_dict=None):
    if unk_words_dict is None:
        unk_words_dict = {}
    if not isinstance(seq, list):
        seq = seq.squeeze().cpu().numpy().tolist()
    unk_count = 0
    for x in seq:
        if x == util.unk_id:
            unk_count += 1
    if unk_count == 0:
        return tokenizer.convert_tokens_to_string([tokenizer._convert_id_to_token(x) for x in seq if x not in [1, 2]])
    else:
        tokens_lists = [[]]
        for idx, x in enumerate(seq):
            if x in [util.bos_id, util.eos_id]:
                continue
            if x == util.unk_id:
                unk_words = unk_words_dict[idx]
                cur_size = len(tokens_lists)
                size = len(unk_words)
                new_tokens_lists = []
                for copy_time in range(size):
                    if len(new_tokens_lists) > 100:
                        continue
                    new_tokens_lists += copy.deepcopy(tokens_lists)
                tokens_lists = new_tokens_lists
                for unk_idx in range(size):
                    for i in range(cur_size):
                        full_idx = unk_idx * cur_size + i
                        if full_idx < len(tokens_lists):
                            tokens_lists[full_idx].append(unk_words[unk_idx])
            else:
                for tokens_idx in range(len(tokens_lists)):
                    tokens_lists[tokens_idx].append(tokenizer._convert_id_to_token(x))
        return [tokenizer.convert_tokens_to_string(tokens) for tokens in tokens_lists]


def init_dict():
    # {util.bos_id: [util.bos_id], util.eos_id: [util.eos_id]} -> not needed as perturbed region is only part of
    # the prompt
    return dict()


def get_word_from_token(token):
    return token.lower()


def transform_token(orig_token, new_word_list):
    if orig_token.lower().capitalize() == orig_token:
        return [new_word.capitalize() for new_word in new_word_list]
    else:
        return new_word_list


def difference(a, b):
    tot = 0
    for x, y in zip(a, b):
        if x != y:
            tot += 1

    return tot


def get_cluster_dict(input_cluster_dict, input_ids, tokenizer):
    cluster_dict = init_dict()
    input_ids = input_ids.squeeze().cpu().numpy().tolist()
    token_list = [tokenizer._convert_id_to_token(x) for x in input_ids]
    for i in range(len(token_list)):
        if input_ids[i] in cluster_dict:
            continue
        word = get_word_from_token(token_list[i])
        if word not in input_cluster_dict:
            cluster_dict[input_ids[i]] = [input_ids[i]]
            continue
        candidates = input_cluster_dict[word]
        candidates = [x[0] for x in candidates]
        candidates = transform_token(token_list[i], candidates)
        candidates = [tokenizer.convert_tokens_to_ids(x) for x in candidates]
        if input_ids[i] not in candidates:
            candidates.append(input_ids[i])
        while util.unk_id in candidates:
            candidates.remove(util.unk_id)
        cluster_dict[input_ids[i]] = candidates

    return cluster_dict


def get_knowledge_dict(input_knowledge_dict, input_ids, tokenizer):
    knowledge_dict = init_dict()
    input_ids = input_ids.squeeze().cpu().numpy().tolist()
    token_list = [tokenizer._convert_id_to_token(x) for x in input_ids]
    for i in range(len(token_list)):
        if input_ids[i] in knowledge_dict:
            continue
        word = get_word_from_token(token_list[i])
        if word not in input_knowledge_dict:
            knowledge_dict[input_ids[i]] = [input_ids[i]]
            continue
        candidates = input_knowledge_dict[word]
        candidates = [x[0] for x in candidates]
        candidates = transform_token(token_list[i], candidates)
        candidates = [tokenizer.convert_tokens_to_ids(x) for x in candidates]
        if input_ids[i] not in candidates:
            candidates.append(input_ids[i])
        while util.unk_id in candidates:
            candidates.remove(util.unk_id)
        knowledge_dict[input_ids[i]] = candidates

    return knowledge_dict


def get_typo_dict(input_typo_dict, input_ids, tokenizer):
    typo_dict = init_dict()
    input_ids = input_ids.squeeze().cpu().numpy().tolist()
    token_list = [tokenizer._convert_id_to_token(x) for x in input_ids]
    unk_words_dict = {}
    for i in range(len(token_list)):
        if input_ids[i] in typo_dict:
            for j in range(len(token_list)):
                if input_ids[i] == input_ids[j]:
                    if j in unk_words_dict:
                        unk_words_dict[i] = unk_words_dict[j]
                    break
            continue
        word = get_word_from_token(token_list[i])
        if word not in input_typo_dict:
            typo_dict[input_ids[i]] = [input_ids[i]]
            continue
        candidates = input_typo_dict[word]
        candidates = [x[0] for x in candidates]
        candidates = transform_token(token_list[i], candidates)
        unk_words_dict[i] = [x for x in candidates if tokenizer.convert_tokens_to_ids(x) == util.unk_id]
        candidates = [tokenizer.convert_tokens_to_ids(x) for x in candidates]
        if input_ids[i] not in candidates:
            candidates.append(input_ids[i])
        typo_dict[input_ids[i]] = candidates

    return typo_dict, unk_words_dict


def cw_word_attack(data_val, args, model, tokenizer, device, logger):
    logger.info("Begin Attack")
    logger.info(("const confidence lr:", args.const, args.confidence, args.lr))

    orig_failures = 0
    adv_correct = 0
    orig_correct = 0
    tot = 0
    tot_diff = 0
    tot_len = 0
    changed_rates = []
    nums_changed = []
    orig_texts = []
    adv_texts = []
    true_labels = []
    new_labels = []
    text_len = []

    ori_labels = []
    ori_preds = []
    preds = []

    test_batch = DataLoader(data_val, batch_size=1, shuffle=False)
    cw = CarliniL2(
        args, logger, debug=True, targeted=True, device=device,
        num_classes=len(LABEL_CANDIDATE[args.task]), decode=args.decode_adv
    )
    for batch_index, batch in enumerate(tqdm(test_batch)):
        inputs = batch
        batch_add_start = batch['add_start'] = []
        batch_add_end = batch['add_end'] = []
        batch['seq_len'] = []
        for i, sentence in enumerate(batch[GLUE_TASK_TO_KEYS[args.task][0]]):
            # input_x = sentence2 if args.fix_sentence == 0 else sentence1
            if args.fix_sentence == 0:  # Perturb Sentence 2
                batch['add_start'].append(batch["input2_start_idx"][i] - batch["input_start_idx"][i])
                batch['add_end'].append(batch["input_end_idx"][i] - batch["input_start_idx"][i])
            else:  # Perturb Sentence 1
                batch['add_start'].append(2)  # first two tokens: '▁sentence', ':'
                batch['add_end'].append(batch["input1_end_idx"][i] - batch["input_start_idx"][i])
            batch['seq_len'].append(len(inputs['input_token_ids'][i]))
            # print(inputs['input_token_ids'][i][batch['add_start'][i]:batch['add_end'][i]])
            # print(len(inputs['input_token_ids'][i]), batch['add_start'][i], batch['add_end'][i])
            # print(inputs['input_token_ids'][i])
        tot += len(batch['label'])

        # FIXME: Handle batches later
        inputs = {k: v[0].to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        inputs["label_names"] = [x[0] for x in inputs["label_names"]]

        label = batch['label'] = batch['label'].to(device)
        attack_targets = torch.full_like(batch['label'], batch['target'].item()).to(device).long()

        # test original acc
        out = model(inputs)["logits"]
        prediction = torch.max(out, 1)[1]
        ori_prediction = prediction
        batch['orig_correct'] = torch.sum((prediction == label).float())
        if prediction.item() != label.item():
            orig_failures += 1
            continue

        # prepare attack
        input_embedding = model.get_input_embedding_vector(inputs['input_token_ids'])
        cw_mask = np.zeros(input_embedding.shape).astype(np.float32)
        cw_mask = torch.from_numpy(cw_mask).float().to(device)
        for i, sentence in enumerate(batch[GLUE_TASK_TO_KEYS[args.task][0]]):
            cw_mask[batch['add_start'][i]:batch['add_end'][i]] = 1

        # FIXME: Batched processing
        cluster_char_dict = get_cluster_dict(json.loads(batch['similar_dict'][0]), inputs['input_token_ids'], tokenizer)
        typo_dict, unk_words_dict = get_typo_dict(
            json.loads(batch['bug_dict'][0]), inputs['input_token_ids'], tokenizer
        )
        knowledge_dict = get_knowledge_dict(
            json.loads(batch['knowledge_dict'][0]), inputs['input_token_ids'], tokenizer
        )

        for k, v in cluster_char_dict.items():
            synset = list(set(v + knowledge_dict[k]))
            knowledge_dict[k] = synset

        for k, v in typo_dict.items():
            synset = list(set(v + knowledge_dict[k]))
            knowledge_dict[k] = synset

        # print(knowledge_dict)
        cw.wv = knowledge_dict
        cw.mask = cw_mask
        cw.seq = inputs['input_token_ids']
        cw.batch_info = batch
        cw.tokenizer = tokenizer

        # attack
        adv_data = cw.run(model, input_embedding, attack_targets, inputs)
        # retest
        adv_seq = inputs['input_token_ids'].clone().detach().to(device)
        if not cw.o_best_sent:
            continue
        for i in range(batch_add_start[0], batch_add_end[0]):
            print(
                "adv_seq[i]", adv_seq[i], "knowledge_dict[adv_seq[i].item()]", knowledge_dict[adv_seq[i].item()],
                "cw.o_best_sent", cw.o_best_sent, "i - batch_add_start[0]", i - batch_add_start[0]
            )
            adv_seq[i] = knowledge_dict[adv_seq[i].item()][cw.o_best_sent[i - batch_add_start[0]]]
        adv_inputs = copy.deepcopy(inputs)
        adv_inputs['input_token_ids'] = adv_seq
        print("Sentence", tokenizer.decode(inputs['input_token_ids']))
        print("Adv Sentence", tokenizer.decode(adv_inputs['input_token_ids']))

        out = model(input_dict=adv_inputs)["logits"]
        prediction = torch.max(out, 1)[1]
        orig_correct += batch['orig_correct'].item()
        adv_correct += torch.sum((prediction == label).float()).item()

        diff = difference(adv_seq, inputs['input_token_ids'])
        tot_diff += diff
        tot_len += batch['seq_len'][0]  # TODO: Change later
        changed_rate = 1.0 * diff / batch['seq_len'][0]
        if ori_prediction.item() == label.item() and prediction.item() == attack_targets.item():
            changed_rates.append(changed_rate)
            nums_changed.append(diff)
            orig_texts.append(transform(inputs['input_token_ids'], tokenizer=tokenizer))
            adv_texts.append(transform(adv_seq, tokenizer, unk_words_dict))
            true_labels.append(label.item())
            new_labels.append(prediction.item())
            text_len.append(batch['seq_len'])

        ori_labels.append(label.item())
        ori_preds.append(ori_prediction.item())
        preds.append(prediction.item())

        results = []
        for i in range(len(changed_rates)):
            save_dict = {'orig_text': orig_texts[i], 'orig_y': true_labels[i], 'pred_y': new_labels[i],
                         'diff': nums_changed[i], 'diff_ratio': changed_rates[i], 'seq_len': text_len[i]}
            if isinstance(adv_texts[i], str):
                save_dict['adv_text'] = adv_texts[i]
                results.append(save_dict)
            else:
                for t in adv_texts[i]:
                    new_save_dict = copy.deepcopy(save_dict)
                    new_save_dict['adv_text'] = t
                    results.append(new_save_dict)
        joblib.dump(results, os.path.join(args.output_dir, 'attack_results.pkl'))

        message = 'For target model {}:\noriginal accuracy: {:.2f}%,\nadv accuracy: {:.2f}%,\n' \
                  'attack success rates: {:.2f},\navg changed rate: {:.02f}%\n'
        message = message.format(
            args.model, (1 - orig_failures / len(test_batch)) * 100, (adv_correct / len(test_batch)) * 100,
            len(adv_texts) / (len(test_batch) - orig_failures) * 100, np.mean(changed_rates) * 100
        )
        logger.info(message)

        joblib.dump(
            {'original_labels': ori_labels, 'original_predictions': ori_preds, 'predictions': preds},
            os.path.join(args.output_dir, 'labels.pkl')
        )


def main():
    args = util.get_args()
    if args.task == "sst2":
        assert args.fix_sentence == 1

    args.output_dir = os.path.join(
        args.output_dir, args.model, args.task, args.split, "l1" if args.l1 else "l2",
        f"fix_{args.fix_sentence}", str(args.shard)
    )
    print(f"saving to {args.output_dir}")
    if args.tf32:
        from torch.backends import cuda
        torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.dataset_dir, exist_ok=True)
    logger = util.init_logger(args.output_dir)

    device = torch.device("cuda:0")

    # FIXME: Hard code for now
    model = ZeroShotLlamaForSemAttack(args.model, args.cache_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    if args.bf16:
        assert not args.tf32
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model.to(device)
    model.eval()

    # Set the random seed manually for reproducibility.
    util.set_seed(args.seed)
    if args.split == "train":
        split = "train"
    else:
        if args.task == 'mnli':
            split = 'validation_matched'
        elif args.task == 'mnli-mm':
            split = 'validation_mismatched'
        else:
            split = "validation"

    test_data = load_dataset("glue", args.task.replace("-mm", ""), cache_dir=args.cache_dir, split=split)
    test_data = test_data.load_from_disk(os.path.join("./adv-glue/", args.model, args.task, args.split, "FC_FT_FK"))
    test_data.set_format("pt")

    if args.shard != -1:
        test_data = test_data.shard(num_shards=args.num_shard, index=args.shard)

    cw_word_attack(test_data, args, model, tokenizer, device, logger)


if __name__ == '__main__':
    main()
