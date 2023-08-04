from tqdm import tqdm
from data_utils import read_abc
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class ABCDataset(Dataset):
    def __init__(self, paths, tokenizer,
                 context_bars_num=8,
                 target_bars_num=8,
                 is_test=False,
                 max_len=1024,):
        super().__init__(ABCDataset)
        self.paths = paths
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.notes, self.keys = self.read_files(paths)
        self.column_names = ['input_ids', 'attention_mask', 'labels']
        self. __version__ = torch.__version__
        self.context_bars_num = context_bars_num
        self.target_bars_num = target_bars_num
        self.bos = '[BOS]'
        self.eos = '[EOS]'
        self.is_test = is_test

    def __len__(self):
        return len(self.paths)

    def read_files(self, paths):
        keys, notes = [], []
        for path in tqdm(paths, desc="Reading files", total=len(paths)):
            meta, abc = read_abc(path)
            keys.append(meta)
            notes.append(abc.split("|"))
        return notes, keys

    def __getitem__(self, idx):
        notes = self.notes[idx]
        keys = self.keys[idx]

        if not self.is_test:
            split_indx = 8

            # split notes to context (input for network) and target (that model must to generate)
            context_notes = notes[split_indx - self.context_bars_num: split_indx]
            target_notes = notes[split_indx: split_indx + self.target_bars_num]
        else:
            context_notes = notes
            target_notes = []

        context = self.bos + "".join(keys) + "".join(context_notes) + self.eos
        target = self.bos + "".join(target_notes) + self.eos

        context_tokenized = self.tokenizer.encode(context)
        labels = self.tokenizer.encode(target).ids
        return {'input_ids': torch.tensor(context_tokenized.ids, dtype=torch.long),
                'attention_mask': torch.tensor(context_tokenized.attention_mask, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long)}


class ABCDataset2(Dataset):
    def __init__(self, paths, tokenizer,
                 context_bars_num=8,
                 target_bars_num=8,
                 is_test=False,
                 max_len=64):

        super().__init__()
        self.paths = paths
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.context_bars_num = context_bars_num
        self.target_bars_num = target_bars_num
        self.bos = '<s>'
        self.eos = '</s>'
        self.is_test = is_test
        self.max_length = max_len

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        keys, notes = read_abc(self.paths[idx])
        notes = notes.split("|")
        if not self.is_test:

            # split notes to context (input for network) and target (that model must to generate)
            context_notes = notes[:self.context_bars_num]
            target_notes = notes[self.context_bars_num: self.context_bars_num + self.target_bars_num]
        else:
            context_notes = notes
            target_notes = []

        context = self.tokenizer.bos_token + "".join(keys) + "|".join(context_notes) + self.tokenizer.eos_token
        target = self.tokenizer.bos_token + "|".join(target_notes) + self.tokenizer.eos_token

        context_tokenized = self.tokenizer(context,
                                           padding='max_length',
                                           truncation=True,
                                           max_length=self.max_length,
                                           return_overflowing_tokens=True,
                                           return_length=True)

        labels = self.tokenizer(target,
                                padding='max_length',
                                truncation=True,
                                max_length=self.max_length,
                                return_overflowing_tokens=True,
                                return_length=True).input_ids

        return {'input_ids': torch.tensor(context_tokenized.input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(context_tokenized.attention_mask, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long)
                }


class ABCDataset_old(Dataset):
    def __init__(self, data,
                 context_bars_num=8,
                 target_bars_num=8,
                 bos_id=2,
                 eos_id=3,
                 is_test=False):

        self.notes = []
        self.keys = []

        for (keys, notes) in data:
            if notes is None:
                continue

            self.keys.append(keys)
            self.notes.append(notes)

        self.context_bars_num = context_bars_num
        self.target_bars_num = target_bars_num
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.is_test = is_test

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        notes = self.notes[idx]
        keys = self.keys[idx]

        if not self.is_test:
            split_indx = 8

            # split notes to context (input for network) and target (that model must to generate)
            context_notes = notes[split_indx - self.context_bars_num: split_indx]
            target_notes = notes[split_indx: split_indx + self.target_bars_num]
        else:
            context_notes = notes
            target_notes = []

        context_tokens = [self.bos_id] + keys
        target_tokens = [self.bos_id]

        for bar in context_notes:
            context_tokens += bar

        for bar in target_notes:
            target_tokens += bar

        context_tokens += [self.eos_id]
        target_tokens += [self.eos_id]

        context_tokens = torch.tensor(context_tokens, dtype=torch.long)
        target_tokens = torch.tensor(target_tokens, dtype=torch.long)

        return {"input_ids": context_tokens, "label_ids": target_tokens}

    def __getitems__(self, idx):
        return [self.__getitem__(i) for i in idx]


class Dataset(torch.utils.data.Dataset):
    """
    This class loads and preprocesses the given text data
    """

    def __init__(self, paths, tokenizer):
        """
        This function initialises the object. It takes the given paths and tokeniser.
        """
        # the last file might not have 10000 samples, which makes it difficult to get the total length of the ds
        self.paths = paths[:len(paths) - 1]
        self.tokenizer = tokenizer
        self.data = self.read_file(self.paths[0])
        self.current_file = 1
        self.remaining = len(self.data)
        self.encodings = self.get_encodings(self.data)

    def __len__(self):
        """
        returns the lenght of the ds
        """
        return 10000 * len(self.paths)

    def read_file(self, path):
        """
        reads a given file
        """
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        return lines

    def get_encodings(self, lines_all):
        """
        Creates encodings for a given text input
        """
        # tokenise all text
        batch = self.tokenizer(lines_all, max_length=512, padding='max_length', truncation=True)

        # Ground Truth
        labels = torch.tensor(batch['input_ids'])
        # Attention Masks
        mask = torch.tensor(batch['attention_mask'])

        # Input to be masked
        input_ids = labels.detach().clone()
        rand = torch.rand(input_ids.shape)

        # with a probability of 15%, mask a given word, leave out CLS, SEP and PAD
        mask_arr = (rand < .15) * (input_ids != 0) * (input_ids != 2) * (input_ids != 3)
        # assign token 4 (=MASK)
        input_ids[mask_arr] = 4

        return {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}

    def __getitem__(self, i):
        """
        returns item i
        Note: do not use shuffling for this dataset
        """
        # if we have looked at all items in the file - take next
        if self.remaining == 0:
            self.data = self.read_file(self.paths[self.current_file])
            self.current_file += 1
            self.remaining = len(self.data)
            self.encodings = self.get_encodings(self.data)

        # if we are at the end of the dataset, start over again
        if self.current_file == len(self.paths):
            self.current_file = 0

        self.remaining -= 1
        return {key: tensor[i % 10000] for key, tensor in self.encodings.items()}