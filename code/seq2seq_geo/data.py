import random

import torch

# what state border s0	( lambda $0 e ( and ( state:t $0 ) ( next_to:t $0 s0 ) ) )


class DataLoader:
    def __init__(self):
        self.vocabs_q = self.load_vocabs_q("seq2seq_geoqueries/vocab.q.txt")
        self.vocabs_f = self.load_vocabs_f("seq2seq_geoqueries/vocab.f.txt")
        self.batch_size = 20

    def load_vocabs_q(self, filename):
        vocabs = []
        with open(filename) as file:
            for line in file:
                word, cnt = line.strip().split("\t")
                vocabs.append(word)
        vocabs.append("<unk>")
        return vocabs

    def load_vocabs_f(self, filename):
        vocabs = []
        with open(filename) as file:
            for line in file:
                word, idx = line.strip().split("\t")
                vocabs.append(word)
        vocabs.append("<unk>")
        vocabs.append("<s>")
        vocabs.append("<n>")
        return vocabs

    def get_end_idx(self):
        return len(self.vocabs_f) - 1

    def get_start_idx(self):
        return len(self.vocabs_f) - 2

    def load_logic(self, line):
        words = line.strip().split(" ")
        logic = [self.get_start_idx()]
        for word in words:
            if word in self.vocabs_f:
                logic.append(self.vocabs_f.index(word))
            else:
                logic.append(self.vocabs_f.index("<unk>"))
        logic.append(self.get_end_idx())
        return torch.tensor(logic)

    def load_lang(self, line):
        words = line.strip().split(" ")
        lang = []
        for word in words:
            if word in self.vocabs_q:
                lang.append(self.vocabs_q.index(word))
            else:
                lang.append(self.vocabs_q.index("<unk>"))
        return torch.tensor(lang)

    def load_data(self, filename):
        data = []
        with open(filename) as file:
            for line in file:
                lang, logic = line.strip().split("\t")
                data.append((self.load_lang(lang), self.load_logic(logic)))
        self.data = data
        return data

    def random_batch(self):
        res = []
        size = len(self.data) // self.batch_size
        for i in range(self.batch_size):
            res.append(self.data[i * size + random.randint(0, size - 1)])
        return res
