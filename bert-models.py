from cProfile import label
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, CamembertForSequenceClassification, CamembertForTokenClassification,CamembertTokenizer, BertTokenizerFast,AlbertForTokenClassification, AlbertTokenizer, BertTokenizer, BertForTokenClassification, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


# tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
# model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

# nlp = pipeline("ner", model=model, tokenizer=tokenizer)
# example = "My name is Wolfgang and I live in Berlin"

# ner_results = nlp(example)
# print(ner_results)

df = pd.read_csv("train.csv", sep=",")
# df2 = pd.read_csv("submission_ciad.csv", sep=",")
# df3 = pd.read_csv("test.csv", sep=",")
# Texte à classer
# texts  = list(df["Token"]) + list(df3["Token"])
# labels = list(df["Label"]) + list(df2["Label"])
texts  = list(df["Token"]) 
labels = list(df["Label"])
ul = list(set(labels))
dic = {ul[i]:i for i in range(len(ul))}
rdic = {i:ul[i] for i in range(len(ul))}

labels = [dic[l]for l in labels]
# labels = [dic[l]for l in labels]
print(dic, rdic)
num_classes=5
# Définir les paramètres d'entraînement
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
learning_rate = 2e-5

# tokenizer = CamembertTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")# dslim/bert-base-NER => 92%
# model = CamembertForSequenceClassification.from_pretrained('Jean-Baptiste/camembert-ner')#distilbert-base-uncased')
# Charger et tokenizer les données
tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')#Evolett/rubert-tiny2-finetuned-ner')#prajjwal1/bert-tiny') gagan3012/bert-tiny-finetuned-ner jcr987/camembert-finetuned-ner
model = BertForSequenceClassification.from_pretrained('prajjwal1/bert-tiny', num_labels=num_classes, ignore_mismatched_sizes=True)

# tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
# model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=num_classes)

# model = BertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=len(labels))
# tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

# tokenizer = CamembertTokenizer.from_pretrained('camembert/camembert-base-wikipedia-4gb')
# model = CamembertForTokenClassification.from_pretrained('camembert/camembert-base-wikipedia-4gb', num_labels=len(labels))
# Load CamemBERT tokenizer
# tokenizer = CamembertTokenizer.from_pretrained('camembert-base-ccnet')
# # Load CamemBERT model for token classification
# model = CamembertForTokenClassification.from_pretrained('camembert-base-ccnet', num_labels=len(labels))

# tokenizer = BertTokenizer.from_pretrained("dslim/bert-base-NER")
# model = BertForSequenceClassification.from_pretrained("dslim/bert-base-NER")

# model.to(device)


# Séparer les données en ensembles d'entraînement et de validation
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)
# dic = {"Token":train_texts, "Label":train_labels}
# dft = pd.DataFrame.from_dict(dic)
# dft.to_csv("train_90.csv", sep=",", index=False)
# dic = {"Token":val_texts, "Label":val_labels}
# dft = pd.DataFrame.from_dict(dic)
# dft.to_csv("val_10.csv", sep=",", index=False)

# df1 = pd.read_csv("test.csv", sep=",")
# df2 = pd.read_csv("submission_ciad_corrige_92.csv", sep=",")
# dic = {"Id":list(df1["Id"]), "Token":list(df1["Token"]), "Label":list(df2["Label"])}
# dft = pd.DataFrame.from_dict(dic)
# dft.to_csv("pred.csv", sep=",", index=False)
# exit()

# Créer une classe Dataset personnalisée
class NerDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = torch.tensor([self.labels[idx]])
        return text, label


def train(num_epochs=100):
    # Créer les instances des ensembles d'entraînement et de validation
    train_dataset = NerDataset(train_texts, train_labels)
    val_dataset = NerDataset(val_texts, val_labels)

    # Créer les DataLoaders pour l'entraînement et la validation
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Charger le modèle pré-entraîné


    # Définir l'optimiseur
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Boucle d'entraînement
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_accuracy = 0
        train_total = 0
        for batch in train_dataloader:
            batch_texts, batch_labels = batch
            
            encoded_inputs = tokenizer.batch_encode_plus(
                list(batch_texts),
                add_special_tokens=True,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            # print(encoded_inputs['input_ids'])
            input_ids = encoded_inputs['input_ids'].to(device)
            attention_mask = encoded_inputs['attention_mask'].to(device)
            # print(batch_labels)
            labels = batch_labels.to(device)
            # print(labels.shape, input_ids.shape )
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            _, predicted_labels = torch.max(logits, dim=1)
            
            train_accuracy += (predicted_labels == labels).sum().item()
            train_total += labels.size(0)
            loss = outputs.loss
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_accuracy = train_accuracy / train_total
        # Calculer l'exactitude sur l'ensemble de validation
        model.eval()
        val_accuracy = 0
        val_total = 0
        g_labels, p_labels = [], []
        with torch.no_grad():
            for batch in val_dataloader:
                batch_texts, batch_labels = batch
                encoded_inputs = tokenizer.batch_encode_plus(
                    batch_texts,
                    padding='longest',
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                )
                
                input_ids = encoded_inputs['input_ids'].to(device)
                attention_mask = encoded_inputs['attention_mask'].to(device)
                labels = batch_labels.to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                _, predicted_labels = torch.max(logits, dim=1)
                
                val_accuracy += (predicted_labels == labels).sum().item()
                val_total += labels.size(0)
                g_labels += [x.item() for x in labels]
                p_labels += [x.item() for x in predicted_labels]
        
        
        average_loss = total_loss / len(train_dataloader)
        val_accuracy = val_accuracy / val_total
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {average_loss:.4f} - train_ccuracy: {train_accuracy:.4f} - val_ccuracy: {val_accuracy:.4f}")
        # print(classification_report(g_labels, p_labels))
        data = {"true":g_labels, "pred":p_labels }
        dft = pd.DataFrame.from_dict(data)
        dft.to_csv("val_pred.csv", sep=",", index=False)
        if epoch>0 and epoch%10==0:
            test(epoch)

    # Save the finetuned model
    output_dir = "./tinybert_finetuned"

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Finetuned model saved to:", output_dir)
    

def test(epoch):
    # saved_model_dir = "./tinybert_finetuned"

    # tokenizer = BertTokenizer.from_pretrained(saved_model_dir)
    # model = BertForTokenClassification.from_pretrained(saved_model_dir)
    model.eval()

    # Example input
    df = pd.read_csv("test.csv", sep=",")
    # Texte à classer
    sentence = list(df["Token"])
    print(len(sentence))
    ids =  list(df["Id"])
    test_data = NerDataset(sentence, ids)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    predicted = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            batch_texts, _ = batch
            # print(batch_texts)
            encoded_inputs = tokenizer.batch_encode_plus(
                batch_texts,
                padding='longest',
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            input_ids = encoded_inputs['input_ids'].to(device)
            attention_mask = encoded_inputs['attention_mask'].to(device)
            # labels = batch_labels.to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            _, predicted_labels = torch.max(logits, dim=1)
            
            predicted += predicted_labels
    predicted = list(predicted)
    predicted = [rdic[l.item()].replace("B-","").replace("I-","").replace("-","") for l in predicted]
    data = {"Id":ids, "Label":predicted }
    dft = pd.DataFrame.from_dict(data)
    dft.to_csv("submission_ciad_{}.csv".format(epoch), sep=",", index=False)
    
def corriger(filename):
    df = pd.read_csv(filename, sep=",")
    df2 = pd.read_csv("test.csv", sep=",")
    labels = list(df["Label"])
    tokens = list(df2["Token"])
    glabels=['geogName', 'aucun', 'geogName name', 'geogFeat', 'geogFeat geogName']
    changed=0
    for i in range(1,len(labels)):
        if i<len(labels)-1 and labels[i-1]!="aucun" and labels[i+1]!="aucun" and labels[i]=="aucun":
            labels[i]="geogName name"
            changed+=1
        if labels[i] == "geogName":
            labels[i]="geogName name"
            changed+=1
        else:
            labels[i] = labels[i].replace('geogFeatgeogName', 'geogFeat geogName')
            labels[i] = labels[i].replace('geogNamename', 'geogName name')
            
    print(changed)
    df["Label"]=labels
    #df["Token"]=tokens
    df.to_csv(filename.replace(".csv","_corrige.csv"), sep=",", index=False)

epochs=20
train(num_epochs=epochs)
test(epochs)
#corriger("submission_camemebert_30_64.csv")
