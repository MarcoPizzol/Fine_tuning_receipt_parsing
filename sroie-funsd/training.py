import os
import pandas as pd
from PIL import Image
import numpy as np
from datasets import load_dataset, load_metric, Dataset
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)
from transformers.data.data_collator import default_data_collator

dataset = load_dataset("mp-02/sroie-funsd")

column_names = dataset["train"].column_names
features = dataset["train"].features
remove_columns = column_names

label_list = features["ner_tags"].feature.names
id2label = dict(enumerate(label_list))
label2id = {v: k for k, v in enumerate(label_list)}
num_labels = len(label_list)


# Load pretrained model and processor
config = AutoConfig.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=num_labels,
)

processor = AutoProcessor.from_pretrained(
    "microsoft/layoutlmv3-base",
    use_fast=True,
    add_prefix_space=True,
    apply_ocr=False,
)

model = AutoModelForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    from_tf=False,
    config=config,
)

# Set the correspondences label/ID inside the model config
model.config.label2id = label2id
model.config.id2label = id2label


# Preprocessing the dataset
# The processor does everything for us (prepare the image using LayoutLMv3ImageProcessor
# and prepare the words, boxes and word-level labels using LayoutLMv3TokenizerFast)
def prepare_examples(examples):
    images = examples["image"]
    words = examples["words"]
    boxes = examples["bboxes"]
    word_labels = examples["ner_tags"]

    encoding = processor(
        images,
        words,
        boxes=boxes,
        word_labels=word_labels,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    return encoding


train_dataset = dataset["train"].map(
    prepare_examples,
    batched=True,
    remove_columns=remove_columns,
)

eval_ds = dataset["test"].select(range(dataset["test"].num_rows//2))
predict_ds = dataset["test"].select(range(dataset["test"].num_rows//2, dataset["test"].num_rows))

eval_dataset = eval_ds.map(
    prepare_examples,
    batched=True,
    remove_columns=remove_columns,
)

predict_dataset = predict_ds.map(
    prepare_examples,
    batched=True,
    remove_columns=remove_columns,
)

# Metrics
metric = load_metric("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

training_args = TrainingArguments(output_dir="layoutlmv3-finetuned-sroie-funsd",
                                  max_steps=2500,
                                  per_device_train_batch_size=4,
                                  per_device_eval_batch_size=4,
                                  push_to_hub=True,
                                  push_to_hub_model_id=f"layoutlmv3-finetuned-sroie-funsd",
                                  learning_rate=7e-6,
                                  evaluation_strategy="steps",
                                  eval_steps=250,
                                  load_best_model_at_end=True,
                                  metric_for_best_model="eval_f1")


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()

predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
predictions = np.argmax(predictions, axis=2)

# Remove ignored index (special tokens)
true_predictions = [
    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

trainer.log_metrics("predict", metrics)
trainer.save_metrics("predict", metrics)

# Save predictions
output_predictions_file = os.path.join(training_args.output_dir, "predictions.txt")
if trainer.is_world_process_zero():
    with open(output_predictions_file, "w") as writer:
        for prediction in true_predictions:
            writer.write(" ".join(prediction) + "\n")

kwargs = {"finetuned_from": "layoutlmv3", "tasks": "token-classification"}
kwargs["dataset_tags"] = "mp-02/sroie-funsd"
kwargs["dataset"] = "mp-02/sroie-funsd"

trainer.push_to_hub(**kwargs)