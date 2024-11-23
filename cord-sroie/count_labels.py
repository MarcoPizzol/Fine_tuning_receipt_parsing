from datasets import load_dataset
import json

dataset = load_dataset("mp-02/cord-sroie")
id2label = {0: 'O', 1: 'B-MENU.CNT', 2: 'B-MENU.DISCOUNTPRICE', 3: 'B-MENU.NM', 4: 'B-MENU.NUM', 5: 'B-MENU.PRICE', 6: 'B-MENU.SUB.CNT', 7: 'B-MENU.SUB.NM', 8: 'B-MENU.SUB.PRICE', 9: 'B-MENU.UNITPRICE', 10: 'B-SUB_TOTAL.DISCOUNT_PRICE', 11: 'B-SUB_TOTAL.ETC', 12: 'B-SUB_TOTAL.SERVICE_PRICE', 13: 'B-SUB_TOTAL.SUBTOTAL_PRICE', 14: 'B-SUB_TOTAL.TAX_PRICE', 15: 'B-TOTAL.CASHPRICE', 16: 'B-TOTAL.CHANGEPRICE', 17: 'B-TOTAL.CREDITCARDPRICE', 18: 'B-TOTAL.MENUQTY_CNT', 19: 'B-TOTAL.TOTAL_PRICE', 20: 'I-MENU.CNT', 21: 'I-MENU.DISCOUNTPRICE', 22: 'I-MENU.NM', 23: 'I-MENU.NUM', 24: 'I-MENU.PRICE', 25: 'I-MENU.SUB.CNT', 26: 'I-MENU.SUB.NM', 27: 'I-MENU.SUB.PRICE', 28: 'I-MENU.UNITPRICE', 29: 'I-SUB_TOTAL.DISCOUNT_PRICE', 30: 'I-SUB_TOTAL.ETC', 31: 'I-SUB_TOTAL.SERVICE_PRICE', 32: 'I-SUB_TOTAL.SUBTOTAL_PRICE', 33: 'I-SUB_TOTAL.TAX_PRICE', 34: 'I-TOTAL.CASHPRICE', 35: 'I-TOTAL.CHANGEPRICE', 36: 'I-TOTAL.CREDITCARDPRICE', 37: 'I-TOTAL.MENUQTY_CNT', 38: 'I-TOTAL.TOTAL_PRICE', 39: 'S-COMPANY', 40: 'S-DATE', 41: 'S-ADDRESS'}
labels_set = [[], []]
for elem in dataset["test"]:
    k = elem["ner_tags"]
    for i in k:
        if id2label[i] not in labels_set[0]:
            labels_set[0].append(id2label[i])
            labels_set[1].append(1)
        else:
            labels_set[1][labels_set[0].index(id2label[i])] = 1 + labels_set[1][labels_set[0].index(id2label[i])]

for i,j in zip(labels_set[0], labels_set[1]):
    print(i,":", j, "\n")

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