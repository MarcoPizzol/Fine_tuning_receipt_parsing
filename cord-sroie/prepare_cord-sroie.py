from datasets import load_dataset, Dataset, Features, Value, Sequence, ClassLabel, Image
import pandas as pd
import datasets
from tqdm import tqdm
cord = load_dataset("mp-02/cord")
sroie = load_dataset("mp-02/sroie")

# print(cord['train'].features['ner_tags'].feature.names)
#  ['O', 'B-MENU.CNT', 'B-MENU.DISCOUNTPRICE', 'B-MENU.NM', 'B-MENU.NUM', 'B-MENU.PRICE', 'B-MENU.SUB.CNT', 'B-MENU.SUB.NM', 'B-MENU.SUB.PRICE', 'B-MENU.UNITPRICE', 'B-SUB_TOTAL.DISCOUNT_PRICE', 'B-SUB_TOTAL.ETC', 'B-SUB_TOTAL.SERVICE_PRICE', 'B-SUB_TOTAL.SUBTOTAL_PRICE', 'B-SUB_TOTAL.TAX_PRICE', 'B-TOTAL.CASHPRICE', 'B-TOTAL.CHANGEPRICE', 'B-TOTAL.CREDITCARDPRICE', 'B-TOTAL.MENUQTY_CNT', 'B-TOTAL.TOTAL_PRICE', 'I-MENU.CNT', 'I-MENU.DISCOUNTPRICE', 'I-MENU.NM', 'I-MENU.NUM', 'I-MENU.PRICE', 'I-MENU.SUB.CNT', 'I-MENU.SUB.NM', 'I-MENU.SUB.PRICE', 'I-MENU.UNITPRICE', 'I-SUB_TOTAL.DISCOUNT_PRICE', 'I-SUB_TOTAL.ETC', 'I-SUB_TOTAL.SERVICE_PRICE', 'I-SUB_TOTAL.SUBTOTAL_PRICE', 'I-SUB_TOTAL.TAX_PRICE', 'I-TOTAL.CASHPRICE', 'I-TOTAL.CHANGEPRICE', 'I-TOTAL.CREDITCARDPRICE', 'I-TOTAL.MENUQTY_CNT', 'I-TOTAL.TOTAL_PRICE']

# print(sroie['train'].features['ner_tags'].feature.names)
#  ['S-COMPANY', 'S-DATE', 'S-ADDRESS', 'S-TOTAL', 'O']

new_tags = cord['train'].features['ner_tags'].feature.names + sroie['train'].features['ner_tags'].feature.names[:3]
#  ['O', 'B-MENU.CNT', 'B-MENU.DISCOUNTPRICE', 'B-MENU.NM', 'B-MENU.NUM', 'B-MENU.PRICE', 'B-MENU.SUB.CNT', 'B-MENU.SUB.NM', 'B-MENU.SUB.PRICE', 'B-MENU.UNITPRICE', 'B-SUB_TOTAL.DISCOUNT_PRICE', 'B-SUB_TOTAL.ETC', 'B-SUB_TOTAL.SERVICE_PRICE', 'B-SUB_TOTAL.SUBTOTAL_PRICE', 'B-SUB_TOTAL.TAX_PRICE', 'B-TOTAL.CASHPRICE', 'B-TOTAL.CHANGEPRICE', 'B-TOTAL.CREDITCARDPRICE', 'B-TOTAL.MENUQTY_CNT', 'B-TOTAL.TOTAL_PRICE', 'I-MENU.CNT', 'I-MENU.DISCOUNTPRICE', 'I-MENU.NM', 'I-MENU.NUM', 'I-MENU.PRICE', 'I-MENU.SUB.CNT', 'I-MENU.SUB.NM', 'I-MENU.SUB.PRICE', 'I-MENU.UNITPRICE', 'I-SUB_TOTAL.DISCOUNT_PRICE', 'I-SUB_TOTAL.ETC', 'I-SUB_TOTAL.SERVICE_PRICE', 'I-SUB_TOTAL.SUBTOTAL_PRICE', 'I-SUB_TOTAL.TAX_PRICE', 'I-TOTAL.CASHPRICE', 'I-TOTAL.CHANGEPRICE', 'I-TOTAL.CREDITCARDPRICE', 'I-TOTAL.MENUQTY_CNT', 'I-TOTAL.TOTAL_PRICE', 'S-COMPANY', 'S-DATE', 'S-ADDRESS']


#before changing tags we need to increse the number of tags in sroie
sroie = sroie.cast_column("ner_tags", Sequence(feature=ClassLabel(names=new_tags, id=None), length=-1, id=None))

def change_tags(example):
    for i in range(len(example["ner_tags"])):
        #if the tag is O for SROIE we make it O for the new tags
        if example["ner_tags"][i] == 4:
            example["ner_tags"][i] = 0
        #if the tag is TOTAL for SROIE we make it B-TOTAL.TOTAL_PRICE for the new tags (we don't want labels overlapping)
        elif example["ner_tags"][i] == 3:
            example["ner_tags"][i] = 19
        else:
            example["ner_tags"][i] += 39
    return example

sroie = sroie.map(change_tags)


features = Features({
 'image': Image(decode=True, id=None),
 'words': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
 'bboxes': Sequence(feature=Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None), length=-1, id=None),
 'ner_tags': Sequence(feature=ClassLabel(names=new_tags, id=None), length=-1, id=None),
})

train_images = cord['train']['image'] + sroie['train']['image'][:len(sroie['train']['image'])//3]
for i in range(0, len(train_images)):
    train_images[i] = datasets.Image().encode_example(train_images[i])
train_words = cord['train']['words'] + sroie['train']['words'][:len(sroie['train']['image'])//3]
train_bboxes = cord['train']['bboxes'] + sroie['train']['bboxes'][:len(sroie['train']['image'])//3]
train_ner_tags = cord['train']['ner_tags'] + sroie['train']['ner_tags'][:len(sroie['train']['image'])//3]


df = pd.DataFrame({'image': train_images, 'words': train_words, 'ner_tags': train_ner_tags, 'bboxes': train_bboxes})
df_train = Dataset.from_pandas(df, features=features)
df_train = df_train.shuffle()


test_images = cord['test']['image'] + sroie['test']['image'][:len(sroie['test']['image'])//2]
test_words = cord['test']['words'] + sroie['test']['words'][:len(sroie['test']['image'])//2]
test_bboxes = cord['test']['bboxes'] + sroie['test']['bboxes'][:len(sroie['test']['image'])//2]
test_ner_tags = cord['test']['ner_tags'] + sroie['test']['ner_tags'][:len(sroie['test']['image'])//2]
df_test = Dataset.from_dict({'image': test_images, 'words': test_words, 'ner_tags': test_ner_tags, 'bboxes': test_bboxes}, features=features)
df_test = df_test.shuffle()


validation_images = cord['validation']['image'] + sroie['test']['image'][len(sroie['test']['image'])//2:]
validation_words = cord['validation']['words'] + sroie['test']['words'][len(sroie['test']['image'])//2:]
validation_bboxes = cord['validation']['bboxes'] + sroie['test']['bboxes'][len(sroie['test']['image'])//2:]
validation_ner_tags = cord['validation']['ner_tags'] + sroie['test']['ner_tags'][len(sroie['test']['image'])//2:]
df_validation = Dataset.from_dict({'image': validation_images, 'words': validation_words, 'ner_tags': validation_ner_tags, 'bboxes': validation_bboxes}, features=features)
df_validation = df_validation.shuffle()


df_train.push_to_hub("mp-02/cord-sroie", split="train")
df_validation.push_to_hub("mp-02/cord-sroie", split="validation")
df_test.push_to_hub("mp-02/cord-sroie", split="test")
