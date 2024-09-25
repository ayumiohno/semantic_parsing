import json
import random

import numpy as np
import torch

# what state border s0	( lambda $0 e ( and ( state:t $0 ) ( next_to:t $0 s0 ) ) )


class DataLoader:
    def __init__(self, mode):
        self.vocabs_q = []
        self.vocabs_f = []
        self.vocabs_f_by_db = {}
        self.data = []
        self.batch_size = 48
        self.sql_words = json.load(open("sql_words.json"))
        self.load_train(mode)

    def get_default_set(self):
        res = set(
            [self.get_start_idx(), self.get_end_idx(), self.vocabs_f.index("<unk>")]
        )
        for word in self.sql_words:
            if word in self.vocabs_f:
                res.add(self.vocabs_f.index(word))
        return res

    def load_train(self, mode):
        data = json.load(open("train_spider.json"))
        vocabs_q = {}
        vocabs_f = {}
        for idx, row in enumerate(data):
            if (idx % 10) in [3, 6, 9]:
                continue
            logic = row["query_toks_no_value"]
            row["query"]
            lang = row["question_toks"]
            db = row["db_id"]
            for tok in logic:
                if tok not in vocabs_f.keys():
                    vocabs_f[tok] = 1
                else:
                    vocabs_f[tok] += 1
            for tok in lang:
                if tok not in vocabs_q.keys():
                    vocabs_q[tok] = 1
                else:
                    vocabs_q[tok] += 1
        for k, v in vocabs_f.items():
            # if v > 1:
            self.vocabs_f.append(k)
        for k, v in vocabs_q.items():
            if v > 1:
                self.vocabs_q.append(k)
        json.dump(self.vocabs_f, open("vocabs_f.json", "w"))
        json.dump(self.vocabs_q, open("vocabs_q.json", "w"))
        self.vocabs_f.append("<unk>")
        self.vocabs_f.append("<s>")
        self.vocabs_f.append("<n>")
        self.vocabs_q.append("<unk>")
        default_set = self.get_default_set()
        print(len(self.vocabs_q), len(self.vocabs_f))
        for idx, row in enumerate(data):
            if mode == "train":
                if (idx % 10) in [3, 6, 9]:
                    continue
            else:
                if not ((idx % 10) in [3, 6, 9]):
                    continue
            logic = row["query_toks_no_value"]
            lang = row["question_toks"]
            db = row["db_id"]
            logic_idx = [self.get_start_idx()]
            lang_idx = []
            if db not in self.vocabs_f_by_db.keys():
                self.vocabs_f_by_db[db] = default_set.copy()
            for tok in logic:
                if tok in self.vocabs_f:
                    logic_idx.append(self.vocabs_f.index(tok))
                    self.vocabs_f_by_db[db].add(self.vocabs_f.index(tok))
                else:
                    logic_idx.append(self.vocabs_f.index("<unk>"))
            for tok in lang:
                if tok in self.vocabs_q:
                    lang_idx.append(self.vocabs_q.index(tok))
                else:
                    lang_idx.append(self.vocabs_q.index("<unk>"))
            logic_idx.append(self.vocabs_f.index("<n>"))
            if mode == "train":
                self.data.append((torch.tensor(lang_idx), torch.tensor(logic_idx)))
            else:
                self.data.append((torch.tensor(lang_idx), torch.tensor(logic_idx), db))
        if mode == "train":
            validation_idx = np.random.choice(len(self.data), 100, replace=False)
            self.validation_data = [self.data[i] for i in validation_idx]
            self.data = [
                self.data[i] for i in range(len(self.data)) if i not in validation_idx
            ]

    def get_end_idx(self):
        return len(self.vocabs_f) - 1

    def get_start_idx(self):
        return len(self.vocabs_f) - 2

    def random_batch(self):
        res = []
        size = len(self.data) // self.batch_size
        for i in range(self.batch_size):
            res.append(self.data[i * size + random.randint(0, size - 1)])
        return res
