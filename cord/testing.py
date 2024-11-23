import torch
import json
from transformers import (
    AutoModelForTokenClassification,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)
import numpy as np
from datasets import load_dataset
from evaluate import load as load_metric
from seqeval.metrics import classification_report

dataset = load_dataset("mp-02/cord")

processor = AutoProcessor.from_pretrained(
    "microsoft/layoutlmv3-base",
    apply_ocr=False,
)

label_list = dataset["test"].features["ner_tags"].feature.names
id2label = dict(enumerate(label_list))
label2id = {v: k for k, v in enumerate(label_list)}

num_labels = len(label_list)

def prepare_examples(examples):
  images = examples["image"]
  words = examples["words"]
  boxes = examples["bboxes"]
  word_labels = examples["ner_tags"]

  encoding = processor(images, words, boxes=boxes, word_labels=word_labels,
                       truncation=True, padding="max_length")

  return encoding


# Metrics
metric = load_metric("seqeval")
return_entity_level_metrics = True


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    print (classification_report(true_labels, true_predictions))


val_dataset = dataset['test'].map(prepare_examples,
                                      batched=True,
                                      remove_columns=dataset['test'].column_names
                                      )

test_args = TrainingArguments(
    output_dir="layoutlmv3-finetuned-cord_test",
    do_train=False,
    do_predict=True,
    per_device_eval_batch_size=5,
    dataloader_drop_last=False,
)

model = AutoModelForTokenClassification.from_pretrained("mp-02/layoutlmv3-base-cord")

# init trainer
trainer = Trainer(
              model=model,
              args=test_args,
              compute_metrics=compute_metrics)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

trainer.predict(val_dataset)
