# ViT-for-Trash

This project utilize Vision Transformer to clasiy trash image dataset. The reason why Vision Transformer is choosed over traditional CNNs because its can use a self-attention mechanism that allows them to attend to any part of the image, regardless of its location. Its also can be trained on smaller datasets since the dataset only have 5054 samples.

# Step To Reproduce This Project's Model
- install dependencies

  ```
  pip install -r requirements.txt
  ```
- load the dataset
  ```
  from datasets import load_dataset
  dataset = load_dataset("garythung/trashnet")
  ```
- split dataset into train, test, val
  ```
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
  ```
- preprocess label
  ```
  id2label = {id: label for id, label in enumerate(train_ds.features["label"].names)}
  label2id = {label: id for id, label in id2label.items()}
  ```
- load model & processor
  ```
  model_name = "google/vit-large-patch16-224"
  processor = ViTImageProcessor.from_pretrained(model_name)
  model = ViTForImageClassification.from_pretrained(model_name, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
  ```
- preprocess the dataset
  ```
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
  
  val_transforms = Compose(
      [
          Resize(size),
          CenterCrop(size),
          ToTensor(),
          normalize,
      ]
  )
  test_transforms = Compose(
      [
          Resize(size),
          ToTensor(),
          normalize,
      ]
  )
  ```
  
- appy the preprocess method
  ```
  train_ds.set_transform(apply_train_transforms)
  val_ds.set_transform(apply_val_transforms)
  test_ds.set_transform(apply_test_transforms)
  ```
  
- adding necessary function
  ```
  def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}
  
  def data_loader(train_ds):
    train_dl = DataLoader(train_ds, collate_fn=collate_fn, batch_size=4)
    return train_dl
  
  def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)
  ```
  
- initialize training arguments
  ```
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
  ```
  
- intialize trainer object
  ```
  trainer = Trainer(
      model,
      train_args,
      train_dataset=train_ds,
      eval_dataset=val_ds,
      data_collator=collate_fn,
      tokenizer=processor,
      compute_metrics=compute_metrics,

  )
  ```
- finally train the model
  ```
  trainer.train()
  ```
