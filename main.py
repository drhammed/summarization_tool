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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
import uuid
from datetime import datetime, timedelta
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from langchain.embeddings import HuggingFaceInstructEmbeddings
#from InstructorEmbedding import INSTRUCTOR
from sklearn.cluster import KMeans
import numpy as np
import voyageai
from langchain_voyageai import VoyageAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer


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

# Set API key
# OPENAI_API_KEY = os.getenv("My_OpenAI_API_key")
# GROQ_API_KEY = os.getenv("My_Groq_API_key")
# VOYAGEAI_API_key = os.getenv("My_voyageai_API_key")

OPENAI_API_KEY = st.secrets["api_keys"]["OPENAI_API_KEY"]
GROQ_API_KEY = st.secrets["api_keys"]["GROQ_API_KEY"]
VOYAGEAI_API_key = st.secrets["api_keys"]["VOYAGE_AI_API_KEY"]

model_options = ["llama3-8b-8192", "llama3-70b-8192", "gpt-4o-mini", "gpt-4o", "gpt-4"]
selected_model = st.sidebar.selectbox("Select a model", model_options)

# Initialize the language model
def get_model(selected_model):
    if selected_model == "llama3-8b-8192":
        return ChatGroq(groq_api_key=GROQ_API_KEY, model="llama3-8b-8192", temperature=0.02, max_tokens=None, timeout=None, max_retries=2)
    elif selected_model == "llama3-70b-8192":
        return ChatGroq(groq_api_key=GROQ_API_KEY, model="llama3-70b-8192", temperature=0.02, max_tokens=None, timeout=None, max_retries=2) 
    elif selected_model == "gpt-4o-mini":
        return ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=None, timeout=None, max_retries=2, api_key=OPENAI_API_KEY)
    elif selected_model == "gpt-4o":
        return ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=None, timeout=None, max_retries=2, api_key=OPENAI_API_KEY)
    elif selected_model == "gpt-4":
        return ChatOpenAI(model="gpt-4", temperature=0, max_tokens=None, timeout=None, max_retries=2, api_key=OPENAI_API_KEY)
    else:
        raise ValueError("Invalid model selected")

llm_mod = get_model(selected_model)

# Create a prompt template
system_prompt = "You are a helpful assistant. Use your own words to provide a high-level summary of the research articles starting from the methodology (materials and methods) section onwards and exclude the references section. Focus on the key findings, conservation (policy recommendations if any) and conclusions. Write it in paragraphs like document"
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=system_prompt),
    HumanMessagePromptTemplate.from_template("{text}")
])

# Initialize the LLMChain
conversation = LLMChain(llm=llm_mod, prompt=prompt)

def chunk_text_with_langchain(text, chunk_size=8000, chunk_overlap=500):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks

def embed_chunks(chunks):
    vo = voyageai.Client(api_key=VOYAGEAI_API_key)
    result = vo.embed(chunks, model="voyage-large-2-instruct", input_type="document")
    vectors = result.embeddings
    return np.array(vectors)

def clustering(vectors, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)
    labels = kmeans.labels_

    closest_indices = []
    for i in range(num_clusters):
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
        closest_index = np.argmin(distances)
        closest_indices.append(closest_index)

    selected_indices = sorted(closest_indices)
    return selected_indices

def filter_redundant_chunks(chunks, vectors, similarity_threshold=0.8):
    unique_chunks = []
    unique_vectors = []

    for i, vector in enumerate(vectors):
        if len(unique_vectors) == 0:
            unique_chunks.append(chunks[i])
            unique_vectors.append(vector)
        else:
            similarities = cosine_similarity([vector], unique_vectors)
            if max(similarities[0]) < similarity_threshold:
                unique_chunks.append(chunks[i])
                unique_vectors.append(vector)

    return unique_chunks, unique_vectors

def summarize_text(text):
    if selected_model in ["llama3-8b-8192", "llama3-70b-8192", "gpt-4"]:
        # Adjust chunk size to fit within the model's token limit
        chunk_size = 8000
        chunk_overlap = 500
        
        chunks = chunk_text_with_langchain(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        vectors = embed_chunks(chunks)
        
        # Filter out redundant chunks using cosine similarity
        unique_chunks, unique_vectors = filter_redundant_chunks(chunks, vectors, similarity_threshold=0.8)
        
        # Ensure num_clusters does not exceed number of unique chunks
        num_clusters = min(10, len(unique_chunks))
        
        selected_indices = clustering(unique_vectors, num_clusters)
        selected_chunks = [unique_chunks[i] for i in selected_indices]
        selected_text = ' '.join(selected_chunks)
        
        # Process the selected text in smaller chunks if needed
        if len(selected_text) > chunk_size:
            final_summary_chunks = []
            for i in range(0, len(selected_text), chunk_size):
                final_summary_chunks.append(conversation.run(selected_text[i:i + chunk_size]))
            summary = ' '.join(final_summary_chunks)
        else:
            summary = conversation.run(selected_text)
    else:
        summary = conversation.run(text)
    return summary

# Function to calculate ROUGE scores
def calculate_rouge(reference_summary, generated_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)
    return scores


# Streamlit setup
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
    # if section != "other":
    # st.subheader(section.capitalize())
    # st.write(content)

# Using reference summary to compare against
    reference_summary = st.text_area("Enter Reference Summary")
    
    if st.button("Evaluate Summary"):
        if reference_summary:
            rouge_scores = calculate_rouge(reference_summary, summary)
            st.subheader("ROUGE Scores")
            st.write(f"ROUGE-1: {rouge_scores['rouge1']}")
            st.write(f"ROUGE-2: {rouge_scores['rouge2']}")
            st.write(f"ROUGE-L: {rouge_scores['rougeL']}")
        else:
            st.warning("Please enter a reference summary for evaluation.")