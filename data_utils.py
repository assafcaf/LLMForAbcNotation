import random
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


USEABLE_KEYS = [i+":" for i in "BCDFGHIKLMmNOPQRrSsTUVWwXZ"]


def split_data(data, eval_ratio=0.1, seed=421):
    random.seed(seed)
    data = clean_data(data)
    random.shuffle(data)
    split_idx = int(len(data)*eval_ratio)
    eval_set = data[:split_idx]
    train_set = data[split_idx:]
    return train_set, eval_set


def clean_data(data):
    new_data = []
    for item in tqdm(data, desc="cleaning data", total=len(data)):
        d_split = item["abc notation"].split("\n")
        idx = [i+1 for i, x in enumerate(item["abc notation"].split("\n")) if x.startswith("K")][0]
        keys = d_split[:idx]
        notes = "".join(d_split[idx:])
        new_data.append({"keys": keys, "notes": notes})
    return new_data


def read_abc(path):
    keys = []
    notes = []
    with open(path) as rf:
        for line in rf:
            line = line.strip()
            if line.startswith("%"):
                continue

            if any([line.startswith(key) for key in USEABLE_KEYS]):
                keys.append(line)
            else:
                notes.append(line)

    keys = " ".join(keys)
    notes = "".join(notes).strip()
    notes = notes.replace(" ", "")

    if notes.endswith("|"):
        notes = notes[:-1]

    notes = notes.replace("[", " [")
    notes = notes.replace("]", "] ")
    notes = notes.replace("(", " (")
    notes = notes.replace(")", ") ")
    notes = notes.replace("|", " | ")
    notes = notes.strip()
    notes = " ".join(notes.split(" "))

    if not keys or not notes:
        return None, None

    return keys, notes


def collate_function(batch):
    features = [i["input_ids"] for i in batch]
    target = [i["labels"] for i in batch]
    masks = [i["attention_mask"] for i in batch]
    max_len = 1024

    seq = torch.zeros(max_len).long()
    features.append(seq)
    target.append(seq)
    masks.append(seq)

    features_padded = pad_sequence(features, batch_first=True)
    target_padded = pad_sequence(target, batch_first=True)
    masks = pad_sequence(masks, batch_first=True)

    return {"input_ids": features_padded[:-1, :],
            "labels": target_padded[:-1, :],
            "attention_mask": masks[:-1, :],
            }


class CollateFunction:
    def __init__(self, max_length):
        self.max_length = max_length

    def collate_function(self, batch):
        # features = [torch.nn.functional.pad(i["input_ids"][0], (self.max_length - len(i["input_ids"][0]), 0), 'constant', 0) for i in batch]
        # masks = [torch.nn.functional.pad(i["attention_mask"][0], (self.max_length - len(i["attention_mask"][0]), 0), 'constant', 0) for i in batch]
        # target = [torch.nn.functional.pad(i["labels"][0], (0, self.max_length - len(i["labels"][0])), 'constant', 0) for i in batch]
        features = [i["input_ids"] for i in batch]
        target = [i["labels"] for i in batch]
        masks = [i["attention_mask"] for i in batch]


        features_padded = pad_sequence(features, batch_first=True)
        target_padded = pad_sequence(target, batch_first=True)
        masks = pad_sequence(masks, batch_first=True)
        return {"input_ids": features_padded,
                "labels": target_padded,
                "attention_mask": masks}


def parse_notation(sample):
    abc_notation = sample["abc notation"]
    for i, field in enumerate(abc_notation.split("\n")):
        if sum([field.startswith(f) for f in USEABLE_KEYS]):
            continue
        break
    keys = "\n".join(abc_notation.split("\n")[:i])
    abc = "\n".join(abc_notation.split("\n")[i:])
    return keys, abc


def read_file(path):
    keys, notes = [], []
    meta, abc = read_abc(path)
    keys.append(meta)
    notes.append(abc.split("|"))
    return notes, keys

