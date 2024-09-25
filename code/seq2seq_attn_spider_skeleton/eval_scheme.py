import json

sql_words = json.load(open("sql_words.json", "r"))


def get_scheme(tokens):
    res = []
    for token in tokens:
        if token in sql_words:
            res.append(token)
        # else:
        #     res.append('_')
    return res


def get_scheme_inf(tokens):
    res = []
    for token in tokens:
        if token == "_":
            continue
        res.append(token)
        # else:
        #     res.append('_')
    return res


test_file = open("train_spider.json", "r")
inf_file = open("result.txt", "r")

test_lines = json.load(test_file)
inf_lines = inf_file.readlines()

test_file.close()
inf_file.close()

exps = []
for idx, test in enumerate(test_lines):
    if not (idx % 10 in [3, 6, 9]):
        continue
    exps.append(test["query_toks_no_value"])

print(len(exps))
print(len(inf_lines))
assert len(exps) == len(inf_lines)

cnt = 0
cnt_2 = 0
for exp, inf in zip(exps, inf_lines):
    inf = inf.rstrip().rstrip("<n>")
    inf = inf.split(" ")
    exp = [x for x in exp if x != ""]
    inf = [x for x in inf if x != ""]
    exp = get_scheme(exp)
    inf = get_scheme_inf(inf)
    if exp == inf:
        cnt += 1
        print(" ".join(inf))
    # else:
    # print(' '.join(exp), "|||||", ' '.join(inf))
print(cnt / len(inf_lines))
