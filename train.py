import huggingface_hub
import wandb
from datasets import load_dataset, DatasetDict
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomResizedCrop,
    ToTensor,
    Resize,
)
import numpy as np
from evaluate import load
from sklearn.metrics import recall_score, precision_score, accuracy_score
import os

HF_KEY = os.environ["HF_KEY"]
WANDB_API_KEY = os.environ["WANDB_API_KEY"]

huggingface_hub.login(token=HF_KEY)

loaded_dataset = load_dataset("garythung/trashnet")
processor = ViTImageProcessor.from_pretrained("suramadu08/trash-classification-vit")
metric = load("accuracy")


def split_dataset(loaded_dataset):
    train_test = loaded_dataset["train"].train_test_split(test_size=0.3, stratify_by_column="label")
    val_test = train_test["test"].train_test_split(test_size=0.5, stratify_by_column="label")
    
    dataset = DatasetDict({
    "train" : train_test["train"],
    "validation" : val_test["train"],
    "test" : val_test["test"]

    })
    
    train_ds = dataset["train"]
    val_ds = dataset["validation"]
    test_ds = dataset["test"]
    
    return train_ds, val_ds, test_ds

def preprocess_label(train_ds):
    id2label = {id: label for id, label in enumerate(train_ds.features["label"].names)}
    label2id = {label: id for id, label in id2label.items()}
    
    return id2label, label2id

def load_model(id2label, label2id, model_name="suramadu08/trash-classification-vit"):
    model = ViTForImageClassification.from_pretrained(model_name, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
    
    return model

def apply_train_transforms(examples):
    image_mean, image_std = processor.image_mean, processor.image_std
    size = processor.size["height"]
    
    normalize = Normalize(mean=image_mean, std=image_std)
    train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    examples["pixel_values"] = [train_transforms(image.convert("RGB")) for image in examples["image"]]
    return examples


def apply_val_transforms(examples):
    image_mean, image_std = processor.image_mean, processor.image_std
    size = processor.size["height"]
    
    normalize = Normalize(mean=image_mean, std=image_std)
    val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )
    examples["pixel_values"] = [val_transforms(image.convert("RGB")) for image in examples["image"]]
    return examples


def apply_test_transforms(examples):
    image_mean, image_std = processor.image_mean, processor.image_std
    size = processor.size["height"]
    normalize = Normalize(mean=image_mean, std=image_std)
    test_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )
    
    examples["pixel_values"] = [test_transforms(image.convert("RGB")) for image in examples["image"]]
    return examples

def preprocess_dataset(train_ds, val_ds, test_ds):
    train_ds.set_transform(apply_train_transforms)
    val_ds.set_transform(apply_val_transforms)
    test_ds.set_transform(apply_test_transforms)
    return train_ds, val_ds, test_ds

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def data_loader(train_ds):
    train_dl = DataLoader(train_ds, collate_fn=collate_fn, batch_size=4)
    return train_dl

def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

def train(model, processor, train_ds, val_ds):
    train_args = TrainingArguments(
        output_dir="output-models",
        save_total_limit=2,
        report_to="wandb",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=15,
        per_device_eval_batch_size=10,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_dir="logs",
        remove_unused_columns=False,
        run_name = 'second-run'
    )

    trainer = Trainer(
        model,
        train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        tokenizer=processor,
        compute_metrics=compute_metrics,

    )
    
    return trainer

def predict(trainer, ds):
    outputs = trainer.predict(ds)
    return outputs

def evaluate(trainer, ds):
    outputs = predict(trainer, ds)
    y_true = outputs.label_ids
    y_pred = outputs.predictions.argmax(1)
    
    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    
    return recall, precision, accuracy

def main():    
    
    train_ds, val_ds, test_ds = split_dataset(loaded_dataset)
    id2label, label2id = preprocess_label(train_ds)
    
    model = load_model(id2label, label2id)
    
    train_ds, val_ds, test_ds = preprocess_dataset(train_ds, val_ds, test_ds)
    
    trainer = train(model, processor, train_ds, val_ds)
    
    recall, precision, accuracy = evaluate(trainer, test_ds)
    
    if precision < 0.8 or recall < 0.8 or accuracy < 0.8:
        wandb.login(key=WANDB_API_KEY)
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"
        wandb.init(project="vit-for-trash")
        trainer.train()
        model.push_to_hub("trash-classification-vit")
        wandb.finish()
    else:
        print("models is good enough")
    

if __name__ == "__main__":

    main()