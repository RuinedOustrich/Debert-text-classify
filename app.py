#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: gleb diakonov
"""

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import re

MAX_LEN = 256

target_idxs =  {0: 'Astrophysics', 1: 'Condensed Matter', 2: 'Computer Science',
                    3: 'Economics', 4: 'Electrical Engineering and Systems Science',
                    5: 'General Relativity and Quantum Cosmology', 6: 'High Energy Physics - Experiment', 
                    7: 'High Energy Physics - Lattice', 8: 'High Energy Physics - Phenomenology', 
                    9: 'High Energy Physics - Theory', 10: 'Mathematics', 11: 'Mathematical Physics', 
                    12: 'Nonlinear Sciences', 13: 'Nuclear Experiment', 14: 'Nuclear Theory',
                    15: 'Physics', 16: 'Quantitative Biology', 17: 'Quantitative Finance', 18: 'Quantum Physics', 19: 'Statistics'}

def is_ok(text):
    if not text:
        match = True
    else:
        match = re.match("[a-z]+", text)
                         
    return bool(match)

@st.cache_data
def define_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('./token')
    return tokenizer

def preprocess(text):
    
    tokenizer = define_tokenizer()
    
    encoded_text = tokenizer.encode_plus(
        text,
        max_length=MAX_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding='max_length',
        truncation = True,
        return_attention_mask=True,
        return_tensors='pt',
        )

    input_ids = encoded_text['input_ids']
    attention_mask = encoded_text['attention_mask']
    
    return input_ids, attention_mask

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.bert_model = AutoModel.from_pretrained("./bert", return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.batchnorm = nn.BatchNorm1d(768)
        self.pooler = MeanPooling()
        self.linear = torch.nn.Linear(768, 20)

    def forward(self, input_ids, attn_mask):
        output = self.bert_model(
            input_ids, 
            attention_mask=attn_mask,
        )
        output = self.pooler(output.last_hidden_state, attn_mask)
        output_dropout = self.dropout(output)
        output = self.linear(output_dropout)
        return output

@st.cache_data
def configure_model():
    model = BERTClass()
    model.load_state_dict(torch.load('ckpt_epoch8.pt', map_location = 'cpu'))
    return model

def predict(text):
    
    model = configure_model()
    
    model.eval()
    
    with torch.no_grad():
    
        input_ids, attention_mask = preprocess(text)
        preds = model(input_ids, attention_mask)
        output = F.softmax(preds, dim = 1).detach()
        output = output.flatten().numpy()
        sorted, indices = torch.sort(output)
        #output = {tag: round(float(prob)*100, 2) for tag, prob in zip(target_idxs, output)}
        #outputs = {k: v for k, v in sorted(output.items(), reverse = True, key = lambda x: x[1])}
        sorted = sorted * 100
    return sorted, indices


if __name__ == '__main__':
    
    st.markdown("<h1 style='text-align: center;'>Find out the topic of the article</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Please enter title or summary</h2>", unsafe_allow_html=True)
    form = st.form("my_form")
    title = form.text_input("TITLE")
    summary = form.text_area("SUMMARY")
    button = form.form_submit_button("Submit")
    summary = summary.lower()
    title = title.lower()
    if button:
        if not title and not summary:
            st.write("**PLEASE ENTER SOMETHING!**")
        else:
            text = title + ". " + summary
        
            if not is_ok(summary) and is_ok(title):
                st.write("**INCORRECT INPUT FORMAT: SUMMARY**")
            if is_ok(summary) and not is_ok(title):
                st.write("**INCORRECT INPUT FORMAT: TITLE**")
            elif not is_ok(title) and not is_ok(summary):
                st.write("**INCORRECT INPUT FORMAT: TITLE, SUMMARY**")
            elif len(summary.split()) in (1,2,3) and not title:
                st.write("**There are too few words in summary, result can be bad. Make shure you enter full text**")
            elif len(title.split()) in (1,2,3) and not summary:
                st.write("**There are too few words in title, result can be bad. Make shure you enter full text**")
            elif len(title.split()) in (1,2,3) and len(summary.split()) == 1:
                st.write("**There are too few words in title and summary, result can be bad. Make shure you enter full text**")
            else:
                sorted, indices = predict(text)
                sums_probs = []
                for k, v in zip(sorted, indices): 
                    st.write(f'Topic:  **:blue[{target_idxs[v]}]**, probability: **:green[{round(float(k), 2)}%]**')
                    sums_probs.append(round(float(k), 2))
                    if sum(sums_probs) >= 95:
                        break
                
                
