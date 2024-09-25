import json
import random

import numpy as np
import torch

# what state border s0	( lambda $0 e ( and ( state:t $0 ) ( next_to:t $0 s0 ) ) )


class DataLoader:
    def __init__(self, mode):
        self.table_info = json.load(open("tables.json"))
        self.vocabs_q = []
        self.vocabs_f = []
        self.data = []
        self.batch_size = 48
        self.sql_words = json.load(open("sql_words.json"))
        self.load_train(mode)

    def load_train(self, mode):
        data = json.load(open("train_spider.json"))
        vocabs_q = {}
        vocabs_f = {}
        for idx, row in enumerate(data):
            if (idx % 10) in [3, 6, 9]:
                continue
            for tok in row["query_toks_no_value"]:
                if tok not in vocabs_f.keys():
                    vocabs_f[tok] = 1
                else:
                    vocabs_f[tok] += 1
            for tok in row["question_toks"]:
                if tok not in vocabs_q.keys():
                    vocabs_q[tok] = 1
                else:
                    vocabs_q[tok] += 1
        for k, v in vocabs_f.items():
            self.vocabs_f.append(k)
        for k, v in vocabs_q.items():
            if v > 2:
                self.vocabs_q.append(k)
        self.vocabs_f.append("<unk>")
        self.vocabs_f.append("<s>")
        self.vocabs_f.append("<n>")
        self.vocabs_q.append("<unk>")
        self.vocabs_q.append("|")
        self.vocabs_q.append(":")
        self.vocabs_f.append("|")
        for word in self.sql_words:
            if word not in self.vocabs_f:
                self.vocabs_f.append(word)
        for _, info in self.table_info.items():
            for tb, cols in info.items():
                if tb not in self.vocabs_q:
                    self.vocabs_q.append(tb)
                for col in cols:
                    if not col in self.vocabs_q:
                        self.vocabs_q.append(col)
                    if not col in self.vocabs_f:
                        self.vocabs_f.append(col)
        print(len(self.vocabs_f), len(self.vocabs_q))
        # self.vocabs_q = list(set(self.vocabs_q))
        # self.vocabs_f = list(set(self.vocabs_f))
        json.dump(self.vocabs_f, open("vocabs_f.json", "w"))
        json.dump(self.vocabs_q, open("vocabs_q.json", "w"))

        if mode == "train":
            validation_idx = np.random.choice(4900, 100, replace=False)

        for idx, row in enumerate(data):
            if mode == "train":
                if (idx % 10) in [3, 6, 9]:
                    continue
            else:
                if not ((idx % 10) in [3, 6, 9]):
                    continue
            logic = row["query_toks_no_value"]
            lang = row["question_toks"]
            logic_idx = [self.get_start_idx()]
            lang_idx = []
            if mode == "train":  # and not idx in validation_idx:
                # Add Skeleton
                for tok in logic:
                    if tok in self.sql_words:
                        logic_idx.append(self.vocabs_f.index(tok))
                    else:
                        if logic_idx[-1] != self.vocabs_f.index("_"):
                            logic_idx.append(self.vocabs_f.index("_"))
                logic_idx.append(self.vocabs_f.index("|"))
            # Add SQL Query
            for tok in logic:
                if tok in self.vocabs_f:
                    logic_idx.append(self.vocabs_f.index(tok))
                else:
                    logic_idx.append(self.vocabs_f.index("<unk>"))

            db = row["db_id"]
            for tb, cols in self.table_info[db].items():
                lang_idx.append(self.vocabs_q.index(tb))
                lang_idx.append(self.vocabs_q.index(":"))
                for col in cols:
                    lang_idx.append(self.vocabs_q.index(col))
                lang_idx.append(self.vocabs_q.index("|"))
            for tok in lang:
                if tok in self.vocabs_q:
                    lang_idx.append(self.vocabs_q.index(tok))
                else:
                    lang_idx.append(self.vocabs_q.index("<unk>"))
            logic_idx.append(self.vocabs_f.index("<n>"))
            if mode == "train":
                self.data.append((torch.tensor(lang_idx), torch.tensor(logic_idx)))
            else:
                self.data.append((torch.tensor(lang_idx), torch.tensor(logic_idx)))

        if mode == "train":
            self.validation_data = [self.data[i] for i in validation_idx]
            self.data = [
                self.data[i] for i in range(len(self.data)) if i not in validation_idx
            ]

    def get_end_idx(self):
        return self.vocabs_f.index("<n>")

    def get_start_idx(self):
        return self.vocabs_f.index("<s>")

    def random_batch(self):
        res = []
        size = len(self.data) // self.batch_size
        for i in range(self.batch_size):
            res.append(self.data[i * size + random.randint(0, size - 1)])
        return res
