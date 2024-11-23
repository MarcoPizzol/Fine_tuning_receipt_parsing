from datasets import load_dataset
import json

dataset = load_dataset("naver-clova-ix/cord-v2")

labels_set = [[], []]
for elem in dataset["train"]["ground_truth"]:
    k = json.loads(elem)
    for item in k["valid_line"]:
        if item["category"] not in labels_set[0]:
            labels_set[0].append(item["category"])
            labels_set[1].append(1)
        else:
            labels_set[1][labels_set[0].index(item["category"])] = 1 + labels_set[1][labels_set[0].index(item["category"])]

print(labels_set[0], "\n", labels_set[1])

labels_to_remove = []
for i, j in enumerate(labels_set[0]):
    if labels_set[1][i] < 50:
        labels_to_remove.append(j)

print(labels_to_remove)

for l in labels_to_remove:
   labels_set[0].pop(labels_set[0].index(l))

labels_list = list(labels_set[0])
labels_list.sort()
for i, elem in enumerate(labels_list):
    if elem[:2] == "I-":
        continue
    labels_list.append("I-" + elem.upper())
    labels_list[i] = "B-" + elem.upper()
labels_list.append("O")
print(labels_list)
print(len(labels_list))