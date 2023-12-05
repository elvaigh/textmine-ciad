import flash
import torch
from flash.core.data.utils import download_data
from flash.text import TextClassificationData, TextClassifier
from transformers import RobertaTokenizer, RobertaForSequenceClassification, CamembertForSequenceClassification, CamembertForTokenClassification,CamembertTokenizer, BertTokenizerFast,AlbertForTokenClassification, AlbertTokenizer, BertTokenizer, BertForTokenClassification, BertForSequenceClassification, AdamW
from torch.optim import AdamW   
import pandas as pd

# 1. Create the DataModule
# download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", "./data/")


datamodule = TextClassificationData.from_csv(
    "Token",
    "Label",
    train_file="train.csv",
    val_split=0.1,
    #val_file="data/imdb/valid.csv",
    batch_size=128,
)
n_classes = len(datamodule.labels)
print(datamodule.multi_label)
print(datamodule.labels)
# model = BertForSequenceClassification.from_pretrained('Evolett/rubert-tiny2-finetuned-ner', num_labels=n_classes, ignore_mismatched_sizes=True)
# 2. Build the task
model = TextClassifier(num_classes=n_classes, 
                       backbone="prajjwal1/bert-tiny",
                       multi_label=datamodule.multi_label,
                       #learning_rate=2e-5,
                        optimizer=AdamW,
                        #serializer=Probabilities(multi_label=True),  # Labels(multi_label=True),
                        learning_rate=5e-4,
                        lr_scheduler="constant_schedule"
                       )

# 3. Create the trainer and finetune the model
# trainer = flash.Trainer(max_epochs=10, gpus=torch.cuda.device_count())
# trainer.finetune(model, datamodule=datamodule, strategy='no_freeze')#('freeze_unfreeze', 1))

# 4. Classify the tokens
# trainer = flash.Trainer(max_epochs=1, gpus=torch.cuda.device_count())

model = TextClassifier.load_from_checkpoint(checkpoint_path="models/flash_tiny-bert", learning_rate=2e-6)
trainer = flash.Trainer(max_epochs=10, gpus=torch.cuda.device_count())
trainer.finetune(model, datamodule=datamodule, strategy='no_freeze')
trainer.save_checkpoint("models/flash_tiny-bert")

#trainer.freeze()
df = pd.read_csv("test.csv", sep=",")
# Texte à classer
sentence = list(df["Token"])
ids = list(df["Id"])
vdatamodule = TextClassificationData.from_lists(
    predict_data=sentence,
    batch_size=128,
)
predictions = trainer.predict(model, datamodule=vdatamodule, output="labels")
predicted = []
for p in predictions:
    predicted.extend(p)
predicted = [datamodule.labels[p].replace("-", " ")  for p in predicted ]
print(len(predicted), len(ids))
data = {"Id":ids, "Label":predicted }
dft = pd.DataFrame.from_dict(data)
dft.to_csv("submission_ciad_flash_.csv", sep=",", index=False)


