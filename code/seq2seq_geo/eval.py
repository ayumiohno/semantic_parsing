test_file = open("seq2seq_geoqueries/test.txt", "r")
inf_file = open("best_ep95_bactch30/result.txt", "r")

test_lines = test_file.readlines()
inf_lines = inf_file.readlines()

test_file.close()
inf_file.close()

cnt = 0
for test, inf in zip(test_lines, inf_lines):
    _, exp = test.strip().split("\t")
    inf = inf.rstrip().rstrip("<n>")
    exp = exp.split(" ")
    inf = inf.split(" ")
    exp = [x for x in exp if x != ""]
    inf = [x for x in inf if x != ""]
    if exp == inf:
        cnt += 1

print(cnt / len(test_lines))
