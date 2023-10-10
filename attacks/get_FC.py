import json
import os

import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

import faiss
from util import get_args
from collections import Counter
from datasets import load_from_disk
from multiprocess import set_start_method
from tokenization import GLUE_TASK_TO_KEYS
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer


try:
    set_start_method("spawn")
except RuntimeError:
    pass


def get_topk_words(indices):
    count = Counter(word_list[indices])
    sorted_words = sorted(count.items(), key=lambda x: x[1], reverse=True)
    return sorted_words


def filter_words(words, neighbors):
    words = [item[0] for item in words if item[1] >= neighbors]
    return words


def get_similar_dict(data):
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % torch.cuda.device_count())

    similar_char_dict = {}

    indexed_tokens = data["input_ids"]
    input_embeddings = data["input_embeddings"][data["input_start_idx"]:data["input_end_idx"]]
    # TODO: How to deal with special character ▁?
    tokenized_words = [tokenizer._convert_id_to_token(x) for x in indexed_tokens]
    knn_dist, knn_indices = gpu_index.search(input_embeddings, 700)

    for i in range(data["input_start_idx"], data["input_end_idx"]):
        if tokenized_words[i].strip("▁") in word_list_set:
            words = get_topk_words(knn_indices[i - data["input_start_idx"]])
            words = filter_words(words, 8)
        else:
            words = []
        if len(words) >= 1:
            similar_char_dict[tokenized_words[i]] = [
                tokenizer.tokenize(word, add_special_tokens=False) for word in words
            ]
        else:
            similar_char_dict[tokenized_words[i]] = [tokenized_words[i]]

    data["similar_dict"] = json.dumps(similar_char_dict)
    return data


def get_input_embedding(batch):
    tokens_tensor = pad_sequence(batch["input_ids"], batch_first=True).to(device)
    masks_tensor = pad_sequence([torch.ones_like(x) for x in batch["input_ids"]], batch_first=True).to(device)
    with torch.no_grad():
        encoded_layers = model.model(input_ids=tokens_tensor, attention_mask=masks_tensor).last_hidden_state
    batch["input_embeddings"] = encoded_layers.float().cpu().numpy()

    return batch


def gather_embedding(dirname):
    import os
    import json
    import numpy as np
    from glob import glob
    from tqdm import tqdm

    fs = sorted(glob(f"{dirname}/pickles/*.npz"))

    pointer = 0
    word_list_selected = []
    len_list = [0, ]
    s = np.zeros((13928506, 4096), dtype=np.float32)
    print(len(s))

    for f in tqdm(fs):
        word, _ = os.path.basename(f).split(".")
        try:
            locs_and_data = np.load(f)
            if np.all(locs_and_data['points'] == 0):
                continue
            mask = ~np.all(locs_and_data['points'] == 0, axis=1)
            cur_len = locs_and_data['points'][mask].shape[0]
            s[pointer:pointer + cur_len] = locs_and_data["points"][mask]
        except ValueError as e:
            print(e)
            continue
        except IndexError as e:
            print(e)
            continue
        word_list_selected += [word] * cur_len
        pointer = pointer + cur_len
        len_list.append(pointer)

    s = s[:pointer]
    len_list = np.array(len_list)
    word_list_selected = np.array(word_list_selected)
    print(len(s), pointer)
    np.save(f'{dirname}/word_list.npy', word_list_selected)
    np.save(f'{dirname}/len_list.npy', len_list)

    filtered_words = []
    for word in os.listdir(f'{dirname}/pickles/'):
        word, _ = word.split('.')
        filtered_words.append(word)

    with open(f'{dirname}/filtered_words.json', 'w') as outfile:
        json.dump(filtered_words, outfile)

    return s, word_list_selected


if __name__ == '__main__':
    # Set the random seed manually for reproducibility.

    args = get_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, torch_dtype=torch.bfloat16)

    m = 8  # number of centroid IDs in final compressed vectors
    d = 4096
    nlist = 50
    bits = 8  # number of bits in each centroid
    res = faiss.StandardGpuResources()
    index_path = f"{args.embedding_space}.faiss"
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        word_list = np.load(args.word_list).reshape(-1)
    else:
        if not os.path.exists(args.embedding_space):
            embeddings, word_list = gather_embedding(os.path.join("./static", args.model))
        else:
            embeddings = np.load(args.embedding_space)
            word_list = np.load(args.word_list).reshape(-1)

        embedding_space = torch.from_numpy(embeddings)
        quantizer = faiss.IndexFlatL2(d)  # we keep the same L2 distance flat index
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index.train(embedding_space)
        gpu_index.add(embedding_space)
        faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), index_path)

    word_list_set = set(word_list)
    data_path = os.path.join(args.cache_dir, "glue-preprocessed-benign", args.model, args.task, args.split)
    embedding_path = os.path.join(args.cache_dir, "glue-embedding-benign", args.model, args.task, args.split)

    if os.path.exists(embedding_path):
        test_data = load_from_disk(embedding_path)
        test_data.set_format("pt")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model)
        model.eval()
        model = model.to(device)

        test_data = load_from_disk(data_path)
        test_data.set_format("pt")
        if len(test_data) > args.max_length:
            test_data = test_data.train_test_split(test_size=args.max_length, seed=args.seed)["test"]
        test_data = test_data.map(get_input_embedding, num_proc=1, with_rank=False, batched=True, batch_size=16)
        # test_data.save_to_disk(embedding_path)

    test_data = test_data.map(get_similar_dict, num_proc=1, with_rank=False)
    if "input_embeddings" in test_data.column_names:
        test_data = test_data.remove_columns(["input_embeddings"])
    test_data.save_to_disk(os.path.join("./adv-glue", args.model, args.task, args.split, "FC"))
    print(f'Saving to {os.path.join("./adv-glue", args.model, args.task, args.split, "FC")}')
