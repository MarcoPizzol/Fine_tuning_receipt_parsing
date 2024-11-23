from datasets import load_dataset
import json

dataset = load_dataset("mp-02/cord2")
labels = dataset["train"].features["ner_tags"].feature.names
id2label = dict(enumerate(labels))
label2id = {v: k for k, v in enumerate(labels)}

labels_set = [[], []]
for elem in dataset["train"]:
    for item in elem["ner_tags"]:
        if id2label[item][:2] != "B-":
            continue
        elif id2label[item][2:] not in labels_set[0]:
            labels_set[0].append(id2label[item][2:])
            labels_set[1].append(1)
        else:
            labels_set[1][labels_set[0].index(id2label[item][2:])] = 1 + labels_set[1][labels_set[0].index(id2label[item][2:])]

for i in range(len(labels_set[0])):
    print(labels_set[0][i], ": ", labels_set[1][i], "\n")

labels_list = list(labels_set[0])
labels_list.sort()
print(len(labels_list))
