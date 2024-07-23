from __future__ import print_function
import os
import re
import PyPDF2
import fitz
import os.path
import io
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.http import MediaFileUpload
from docx import Document
import configparser
from GDriveOps.GDhandler import GoogleDriveHandler
import nltk
#from nltk.corpus import stopwords
#from nltk.tokenize import sent_tokenize, word_tokenize
#from nltk.stem import WordNetLemmatizer
import string
import openai
import streamlit as st
from langchain_openai import ChatOpenAI
import openai
from groq import Groq
from langchain.chains import LLMChain, RetrievalQA
#import time
#import re
import warnings
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import Runnable
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
import uuid
from datetime import datetime, timedelta
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import string





nltk.download('punkt')
#nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    sentences = sent_tokenize(text)
    punctuation = set(string.punctuation)
    
    processed_sentences = []
    for sent in sentences:
        words = word_tokenize(sent)
        filtered_words = [
            lemmatizer.lemmatize(word.lower()) 
            for word in words 
            if word.lower() not in punctuation and word.isalpha()
        ]
        processed_sentences.append(' '.join(filtered_words))
    
    # Join sentences back into a single string and remove numbers
    processed_text = ' '.join(processed_sentences)
    processed_text = re.sub(r'\d+', '', processed_text)
    
    return processed_text



def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_sections(text):
    sections = {
        "methodology": "",
        "methods": "",
        "results": "",
        "discussion": "",
        "conclusion": ""
    }
    
    current_section = None
    start_extracting = False
    for line in text.split('\n'):
        line_lower = line.lower()
        if "methodology" in line_lower or "methods" in line_lower:
            current_section = "methodology"
            start_extracting = True
        elif "results" in line_lower:
            current_section = "results"
        elif "discussion" in line_lower:
            current_section = "discussion"
        elif "conclusion" in line_lower:
            current_section = "conclusion"
        elif "references" in line_lower:
            start_extracting = False
        
        if start_extracting and current_section:
            sections[current_section] += line + "\n"
    
    combined_text = (sections["methodology"] + sections["results"] + 
                     sections["discussion"] + sections["conclusion"])
    
    return combined_text, sections



#ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=None, timeout=None, max_retries=2, api_key=OPENAI_API_KEY)


# Set OpenAI API key
OPENAI_API_KEY = os.getenv("My_OpenAI_API_key")

#chat = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=None, timeout=None, max_retries=2, api_key=OPENAI_API_KEY)


# Create a prompt template
system_prompt = "You are a helpful assistant. Use your own words to provide a high-level summary of the research articles starting from the methodology/methods section onwards and exclude the references section. Focus on the key findings and conclusions."
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=system_prompt),
    HumanMessagePromptTemplate.from_template("{text}")
])

# Initialize the LLMChain
llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=None, timeout=None, max_retries=2, api_key=OPENAI_API_KEY)
conversation = LLMChain(llm=llm, prompt=prompt)

def summarize_text(text):
    return conversation.run(text)

#set up streamlit
st.title("PDF Research Paper Summarizer")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    with open("uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Generating summary..."):
        text = extract_text_from_pdf("uploaded_file.pdf")
        combined_text, sections = extract_sections(text)
    
        #st.subheader("Combined Text for Summarization")
        #st.write(combined_text)
    
        summary = summarize_text(combined_text)
    
    st.subheader("Summary")
    st.write(summary)

    # st.subheader("Detailed Sections")
    # for section, content in sections.items():
    #     if section != "other":
    #         st.subheader(section.capitalize())
    #         st.write(content)
