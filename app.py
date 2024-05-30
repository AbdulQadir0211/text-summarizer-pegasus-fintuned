import streamlit as st
import numpy as np
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from transformers import pipeline

import torch


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer = AutoTokenizer.from_pretrained("abdulqadir02/Pegasus-fine-tuned")
    model = AutoModelForSeq2SeqLM.from_pretrained("abdulqadir02/Pegasus-fine-tuned")
    return tokenizer, model


tokenizer,model = get_model()
pdf_file = st.file_uploader("Upload PDF file", type=('pdf'))


pdf_text = extract_text_from_pdf(pdf_file)

user_input = pdf_text
button = st.button("Summarize")



if user_input and button:
    pipeline_obj=pipeline("summarization",model=model,tokenizer=tokenizer)
    output = pipeline_obj(user_input)

    st.write("Summarized Text:",output[0]["summary_text"])
    
    