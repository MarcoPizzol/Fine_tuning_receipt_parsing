import json
import datasets
import pandas as pd


def load_image(img):

    image = img.convert("RGB")
    w, h = image.size
    return image, (w, h)


def normalize_bbox(bbox, size):

    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


def quad_to_box(quad):

    box = (
        max(0, quad["x1"]),
        max(0, quad["y1"]),
        quad["x3"],
        quad["y3"]
    )
    if box[3] < box[1]:
        bbox = list(box)
        tmp = bbox[3]
        bbox[3] = bbox[1]
        bbox[1] = tmp
        box = tuple(bbox)
    if box[2] < box[0]:
        bbox = list(box)
        tmp = bbox[2]
        bbox[2] = bbox[0]
        bbox[0] = tmp
        box = tuple(bbox)
    return box


def get_line_bbox(bboxs):

    x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
    y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]

    x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

    assert x1 >= x0 and y1 >= y0
    bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
    return bbox


def generate_examples(ds):

    examples_ids = []
    examples_words = []
    examples_bboxes = []
    examples_ner_tags = []
    examples_images = ds[0:ds.num_rows]["image"]
    ann_dir = ds[0:ds.num_rows]["ground_truth"]

    for guid, file in enumerate(ann_dir):
        words = []
        bboxes = []
        ner_tags = []
        data = json.loads(file)
        image, size = load_image(examples_images[guid])

        for item in data["valid_line"]:
            cur_line_bboxes = []
            line_words, label = item["words"], item["category"]
            line_words = [w for w in line_words if w["text"].strip() != ""]
            if len(line_words) == 0:
                continue
            #  sostituisco le label generiche (.etc) e quelle con poca rappresentazione con O
            if label in ['total.menutype_cnt', 'total.total_etc', 'total.emoneyprice', 'menu.sub.unitprice', 'void_menu.nm', 'void_menu.price', 'sub_total.othersvc_price', 'menu.vatyn', 'menu.itemsubtotal', 'menu.etc']:
                for w in line_words:
                    words.append(w["text"])
                    ner_tags.append("O")
                    cur_line_bboxes.append(normalize_bbox(quad_to_box(w["quad"]), size))
            else:
                words.append(line_words[0]["text"])
                ner_tags.append("B-" + label.upper())
                cur_line_bboxes.append(normalize_bbox(quad_to_box(line_words[0]["quad"]), size))
                for w in line_words[1:]:
                    words.append(w["text"])
                    ner_tags.append("I-" + label.upper())
                    cur_line_bboxes.append(normalize_bbox(quad_to_box(w["quad"]), size))
            # by default: --segment_level_layout 1
            # if do not want to use segment_level_layout, comment the following line
            cur_line_bboxes = get_line_bbox(cur_line_bboxes)
            bboxes.extend(cur_line_bboxes)

        examples_ids.append(str(guid))
        examples_words.append(words)
        examples_bboxes.append(bboxes)
        examples_ner_tags.append(ner_tags)

    return examples_ids, examples_words, examples_bboxes, examples_ner_tags, examples_images


dataset = datasets.load_dataset("naver-clova-ix/cord-v2")

ids_train, words_train, boxes_train, labels_train, images_train = generate_examples(dataset["train"])
ids_val, words_val, boxes_val, labels_val, images_val = generate_examples(dataset["validation"])
ids_test, words_test, boxes_test, labels_test, images_test = generate_examples(dataset["test"])

#using the images (PIL.PNG) directly in the parquet doesn't work, so we encode them first
#there's probably a more efficient way of doing this

for i in range(0, dataset['train'].num_rows):
    images_train[i] = datasets.Image().encode_example(images_train[i])

for i in range(0, dataset['validation'].num_rows):
    images_val[i] = datasets.Image().encode_example(images_val[i])

for i in range(0, dataset['test'].num_rows):
    images_test[i] = datasets.Image().encode_example(images_test[i])


#in order to upload the prepared dataset on Hugginface, we transform it into a pandas dataframe
df_train = pd.DataFrame({'id': ids_train, 'image': images_train, 'words': words_train, 'ner_tags': labels_train, 'bboxes': boxes_train})
df_validation = pd.DataFrame({'id': ids_val, 'image': images_val, 'words': words_val, 'ner_tags': labels_val, 'bboxes': boxes_val})
df_test = pd.DataFrame({'id': ids_test, 'image': images_test, 'words': words_test, 'ner_tags': labels_test, 'bboxes': boxes_test})


features = datasets.Features({
                        "id": datasets.Value("string"),
                        "words": datasets.Sequence(datasets.Value("string")),
                        "ner_tags": datasets.Sequence(
                            datasets.features.ClassLabel(
                                names=['O', 'B-MENU.CNT', 'B-MENU.DISCOUNTPRICE', 'B-MENU.NM', 'B-MENU.NUM', 'B-MENU.PRICE', 'B-MENU.SUB.CNT', 'B-MENU.SUB.NM', 'B-MENU.SUB.PRICE', 'B-MENU.UNITPRICE', 'B-SUB_TOTAL.DISCOUNT_PRICE', 'B-SUB_TOTAL.ETC', 'B-SUB_TOTAL.SERVICE_PRICE', 'B-SUB_TOTAL.SUBTOTAL_PRICE', 'B-SUB_TOTAL.TAX_PRICE', 'B-TOTAL.CASHPRICE', 'B-TOTAL.CHANGEPRICE', 'B-TOTAL.CREDITCARDPRICE', 'B-TOTAL.MENUQTY_CNT', 'B-TOTAL.TOTAL_PRICE', 'I-MENU.CNT', 'I-MENU.DISCOUNTPRICE', 'I-MENU.NM', 'I-MENU.NUM', 'I-MENU.PRICE', 'I-MENU.SUB.CNT', 'I-MENU.SUB.NM', 'I-MENU.SUB.PRICE', 'I-MENU.UNITPRICE', 'I-SUB_TOTAL.DISCOUNT_PRICE', 'I-SUB_TOTAL.ETC', 'I-SUB_TOTAL.SERVICE_PRICE', 'I-SUB_TOTAL.SUBTOTAL_PRICE', 'I-SUB_TOTAL.TAX_PRICE', 'I-TOTAL.CASHPRICE', 'I-TOTAL.CHANGEPRICE', 'I-TOTAL.CREDITCARDPRICE', 'I-TOTAL.MENUQTY_CNT', 'I-TOTAL.TOTAL_PRICE']
                            )
                        ),
                        "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                        "image": datasets.features.Image(),
                        })

#and then we cast it to Dataset
train_dataset = datasets.Dataset.from_pandas(df_train, features=features)
validation_dataset = datasets.Dataset.from_pandas(df_validation, features=features)
test_dataset = datasets.Dataset.from_pandas(df_test, features=features)

train_dataset.push_to_hub("mp-02/cord", split="train")
validation_dataset.push_to_hub("mp-02/cord", split="validation")
test_dataset.push_to_hub("mp-02/cord", split="test")