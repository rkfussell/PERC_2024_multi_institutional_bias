import pandas as pd 
import numpy as np
import os 
import re
import string
from tqdm import tqdm
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model, model_selection
from sklearn.metrics import balanced_accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score

# bert imports
from transformers import BertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,WeightedRandomSampler
import torch.nn as nn
from transformers import BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F


import random
import time
from data_utils import text_preprocessing_simple


def preprocessing_for_bert(data, tokenizer, max_len):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing_simple(str(sent)),  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=max_len,                  # Max length to truncate/pad
            pad_to_max_length=True,         # Pad sentence to max length
            #return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True      # Return attention mask
            )
        
        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks
# Create the BertClassfier class
class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, 2

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits

def initialize_model(device, epochs=4, train_dataloader=None):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(freeze_bert=False)

    # Tell PyTorch to run the model on GPU
    bert_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=5e-5,    # Default learning rate
                      eps=1e-8    # Default epsilon value
                      )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler


def train_BERT(device, model, optimizer, scheduler, train_dataloader, val_dataloader=None, epochs=4, evaluation=False, weight=[1,1]):
    """Train the BertClassifier model.
        """
    # Start training loop
    print("Start training...\n")
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weight, dtype=torch.float).to(device))
    for epoch_i in range(epochs):
    # =======================================
    #               Training
    # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)
        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()
        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0
        # Put the model into the training mode
        model.train()
        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            # Zero out any previously calculated gradients
            model.zero_grad()
            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)
            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()
            # Perform a backward pass to calculate gradients
            loss.backward()
            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()
            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
            # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch
            # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")
            # Reset batch tracking variables
        batch_loss, batch_counts = 0, 0
        t0_batch = time.time()
        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)
        print("-"*70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
        # After the completion of each training epoch, measure the model's performance
        # on our validation set.
            val_loss, val_accuracy = evaluate_bert(device, model, val_dataloader)
        # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
    print("-"*70)
    print("\n")
    print("Training complete!")
def evaluate_bert(device, model, val_dataloader, weight=[1,1]):
    """After the completion of each training epoch, measure the model's performance
        on our validation set.
        """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()
    # Tracking variables
    val_accuracy = []
    val_loss = []
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weight, dtype=torch.float).to(device))
    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
            # Compute loss
            loss = loss_fn(logits, b_labels)
            val_loss.append(loss.item())
            # Get the predictions
            preds = torch.argmax(logits, dim=1).flatten()
            # Calculate the accuracy rate
            accuracy = (preds == b_labels).cpu().numpy().mean() * 100
            #accuracy=balanced_accuracy_score(b_labels.cpu().numpy(), preds.cpu().numpy())
            
            val_accuracy.append(accuracy)
    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)
    return val_loss, val_accuracy

def bert_predict(device, model, test_dataloader):
    """Perform a forward pass on the trained BERT model to predict probabilities
        on the test set.
        """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()
    all_logits = []
    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]
        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
            all_logits.append(logits)
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)
    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()
    return probs
def get_max_len_bert(tokenizer, train, val=None, include_val=False):
    # Concatenate train data and test data
    if include_val:
        all_sent = np.concatenate((np.array(train.Sentences.values), np.array(val.Sentences.values)))
    else:
        all_sent = np.array(train.Sentences.values)
    for i in range(len(all_sent)):
        all_sent[i]=str(all_sent[i])
    
    # Encode our concatenated data
    encoded_sentences = [tokenizer.encode(sent, add_special_tokens=True) for sent in all_sent]
    
    # Find the maximum length
    max_len = max([len(sent) for sent in encoded_sentences])
    return max_len
def get_train_x_bows(train):
    # Preprocess text
    X_train_preprocessed = [text_preprocessing(str(text)) for text in train.Sentences]
    #X_val_preprocessed = [text_preprocessing(str(text)) for text in val.Sentences]
    # Tokenize with binary encoding (not TF-IDF)
    vectorizer = TfidfVectorizer(ngram_range = (1,1), binary = True, use_idf = False, norm = None, tokenizer = my_tokenizer)
    
    #Create train and val inputs and outputs
    X_train = vectorizer.fit_transform(X_train_preprocessed)
    #X_val = vectorizer.transform(X_val_preprocessed)
    return vectorizer, X_train#, X_val
def get_val_x_bows(val, vectorizer):
    # Preprocess text
    X_val_preprocessed = [text_preprocessing(str(text)) for text in val.Sentences]
    #Create train and val inputs and outputs
    X_val = vectorizer.transform(X_val_preprocessed)
    return vectorizer, X_val


def get_train_x_bert(code, y_train, train, batch_size, tokenizer, max_len, balanced = False):
    weight=1/np.sum(y_train)
    #assign evenly split weights to all lines
    sample_weights = np.ones(len(y_train))/(len(y_train)-(np.sum(y_train)))
    
    #switch to assign weight to just instances of code
    sample_weights[y_train]=weight
    
    # Run function `preprocessing_for_bert` on the train set
    print('Tokenizing data...')
    train_inputs, train_masks = preprocessing_for_bert(train.Sentences, tokenizer, max_len)
    
    # Convert other data types to torch.Tensor
    train_labels = torch.tensor(y_train, dtype=torch.int64)

    # Create the DataLoader for our training set
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    if balanced:
        train_sampler=WeightedRandomSampler(sample_weights,len(y_train), replacement=True)
    else:
        train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return train_data, train_sampler, train_dataloader
def get_val_x_bert(code, y_val, val, batch_size, tokenizer, max_len):
        # Run function `preprocessing_for_bert` on the validation set
    #print('Tokenizing data...')
    val_inputs, val_masks = preprocessing_for_bert(val.explanation, tokenizer, max_len)
    
    # Convert other data types to torch.Tensor
    val_labels = torch.tensor(y_val, dtype=torch.int64)
    
    # Create the DataLoader for our validation set
    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)
    return val_data, val_sampler, val_dataloader



