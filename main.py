
from __future__ import print_function
import os
import re
import PyPDF2
import fitz
import os.path
import io
import json
import streamlit as st
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
import string
from groq import Groq
from langchain.chains import LLMChain, RetrievalQA
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
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import voyageai
from langchain_voyageai import VoyageAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from rouge_score import rouge_scorer
from io import BytesIO




# Initialize NLTK components
from nltk.stem import WordNetLemmatizer

nltk.download('punkt_tab')

#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')


# Set Streamlit page configuration
st.set_page_config(
    page_title="Ecological Research Synthesis",
    layout="wide",
    initial_sidebar_state="expanded",
)

class PDFSummarizer:
    def preprocess_text(self, text):
        lemmatizer = WordNetLemmatizer()
        sentences = nltk.sent_tokenize(text)
        punctuation = set(string.punctuation)

        processed_sentences = []
        for sent in sentences:
            words = nltk.word_tokenize(sent)
            filtered_words = [
                lemmatizer.lemmatize(word.lower()) 
                for word in words 
                if word.lower() not in punctuation and word.isalpha()
            ]
            processed_sentences.append(' '.join(filtered_words))

        processed_text = ' '.join(processed_sentences)
        processed_text = re.sub(r'\d+', '', processed_text)

        return processed_text

    def extract_text_from_pdf(self, pdf_file):
        try:
            pdf_file.seek(0)  # Ensure the file pointer is at the start
            doc = fitz.open(stream=pdf_file, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            st.error(f"Error reading PDF file: {e}")
            return ""

    def extract_sections(self, text, start_section="methodology"):
        sections = {
            "introduction": "",
            "methodology": "",
            "methods": "",
            "results": "",
            "discussion": "",
            "conclusion": ""
        }

        current_section = None
        start_extracting = False
        is_discussion = False

        for line in text.split('\n'):
            line_lower = line.strip().lower()
        
            # Stop extracting when "references" section is encountered
            if line_lower.startswith("references"):
                start_extracting = False

            # Start extracting from the "introduction" section
            elif start_section == "introduction" and ("introduction" in line_lower):
                current_section = "introduction"
                start_extracting = True

            # Start extracting from the "methodology" section, ensure not to start if "discussion" section already started
            elif start_section == "methodology" and ("methodology" in line_lower or
                                                 "methods" in line_lower or 
                                                 "materials and methods" in line_lower or 
                                                 "materials & methods" in line_lower) and not is_discussion:
                current_section = "methodology"
                start_extracting = True

            elif "results" in line_lower and not is_discussion:
                current_section = "results"
                start_extracting = True

            elif "discussion" in line_lower:
                current_section = "discussion"
                is_discussion = True  # To show that discussion section has started
                start_extracting = True

            elif "conclusion" in line_lower:
                current_section = "conclusion"
                start_extracting = True

            # Stop extracting when "acknowledgements" section is encountered
            elif "acknowledgements" in line_lower:
                start_extracting = False

            # Add lines to the current section if extracting is active
            if start_extracting and current_section:
                sections[current_section] += line + "\n"

        # Combine the extracted sections based on the start_section
        if start_section == "introduction":
            combined_text = (sections["introduction"] + sections["methodology"] + sections["results"] +
                             sections["discussion"] + sections["conclusion"])
        else:
            combined_text = (sections["methodology"] + sections["results"] + 
                             sections["discussion"] + sections["conclusion"])

        return combined_text, sections

    def get_model(self, selected_model, GROQ_API_KEY):
        model_mapping = {
            "llama3-8b-8192": "llama3-8b-8192",
            "llama3-70b-8192": "llama3-70b-8192",
            "llama-3.2-1b-preview": "llama-3.2-1b-preview",
            "llama-3.2-3b-preview": "llama-3.2-3b-preview",
            "llama-3.2-11b-text-preview": "llama-3.2-11b-text-preview",
            "llama-3.2-90b-text-preview": "llama-3.2-90b-text-preview"
        }
        if selected_model in model_mapping:
            return ChatGroq(
                groq_api_key=GROQ_API_KEY, 
                model=model_mapping[selected_model], 
                temperature=0.02, 
                max_tokens=None, 
                timeout=None, 
                max_retries=2
            )
        else:
            raise ValueError("Invalid model selected")

    def chunk_text_with_langchain(self, text, chunk_size=8000, chunk_overlap=500):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_text(text)
        return chunks

    def embed_chunks(self, chunks, VOYAGEAI_API_key):
        vo = voyageai.Client(api_key=VOYAGEAI_API_key)
        result = vo.embed(chunks, model="voyage-large-2-instruct", input_type="document")
        vectors = result.embeddings
        return np.array(vectors)

    def clustering(self, vectors, num_clusters):
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)
        labels = kmeans.labels_

        closest_indices = []
        for i in range(num_clusters):
            distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
            closest_index = np.argmin(distances)
            closest_indices.append(closest_index)

        selected_indices = sorted(closest_indices)
        return selected_indices

    def filter_redundant_chunks(self, chunks, vectors, similarity_threshold=0.8):
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

    def optimal_clusters_sil(self, vectors, max_k=50):
        best_k = 2
        best_score = -1

        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(vectors)
            score = silhouette_score(vectors, labels)
        
            if score > best_score:
                best_score = score
                best_k = k

        return best_k

    def summarize_text(self, text, selected_model, prompt, GROQ_API_KEY, VOYAGEAI_API_key, chunk_size=8000, chunk_overlap=500, similarity_threshold=0.8, num_clusters=10):
        llm_mod = self.get_model(selected_model, GROQ_API_KEY)
        system_prompt = prompt
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessagePromptTemplate.from_template("{text}")
        ])
        
        conversation = LLMChain(llm=llm_mod, prompt=prompt_template)
        
        chunks = self.chunk_text_with_langchain(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        vectors = self.embed_chunks(chunks, VOYAGEAI_API_key)
        
        # Filter redundant chunks
        unique_chunks, unique_vectors = self.filter_redundant_chunks(chunks, vectors, similarity_threshold=similarity_threshold)
         
        # Check if unique_chunks is not empty
        if not unique_chunks:
            st.warning(f"No unique chunks found for summarization. Skipping summarization.")
            return ""  # Return an empty summary since no unique chunks are available

        # If the num_clusters is too low compared to the requested num_clusters
        if len(unique_chunks) < num_clusters:
            st.warning(f"Requested {num_clusters} clusters, but only {len(unique_chunks)} unique chunks are available. Adjusting num_clusters to {len(unique_chunks)} clusters.")
            num_clusters = len(unique_chunks)
        
        # Check whether the specified num_clusters is appropriate for the data structure
        elif num_clusters > len(unique_chunks) / 2:
            st.warning(f"Requested {num_clusters} clusters might not capture the data's structure optimally. Considering an optimal cluster analysis using Silhouette Analysis.")
            num_clusters = self.optimal_clusters_sil(unique_vectors, max_k=min(len(unique_chunks), 50))
            st.info(f"Using {num_clusters} clusters after optimization.")

        selected_indices = self.clustering(unique_vectors, num_clusters)
        selected_chunks = [unique_chunks[i] for i in selected_indices]
        selected_text = ' '.join(selected_chunks)
        
        if len(selected_text) > chunk_size:
            summary_parts = []
            for i in range(0, len(selected_text), chunk_size):
                chunk = selected_text[i:i + chunk_size]
                part = conversation.run(chunk)
                summary_parts.append(part)
            summary = ' '.join(summary_parts)
        else:
            summary = conversation.run(selected_text)

        return summary

    def save_summary_as_docx(self, summary, pdf_filename):
        try:
            doc = Document()
            title = f'Summary - {os.path.splitext(pdf_filename)[0]}'
            doc.add_heading(title, 0)
            doc.add_paragraph(summary)
            byte_io = BytesIO()
            doc.save(byte_io)
            byte_io.seek(0)
            return byte_io
        except Exception as e:
            st.error(f"Error saving summary for {pdf_filename}: {e}")
            return None

    def save_summary_as_json(self, summary, pdf_filename):
        try:
            summary_data = {"title": pdf_filename, "summary": summary}
            byte_io = BytesIO()
            byte_io.write(json.dumps(summary_data, indent=4).encode())
            byte_io.seek(0)
            return byte_io
        except Exception as e:
            st.error(f"Error saving summary as JSON for {pdf_filename}: {e}")
            return None

    def save_summary_as_csv(self, summary, pdf_filename):
        try:
            df = pd.DataFrame([{'title': pdf_filename, 'summary': summary}])
            byte_io = BytesIO()
            df.to_csv(byte_io, index=False)
            byte_io.seek(0)
            return byte_io
        except Exception as e:
            st.error(f"Error saving summary as CSV for {pdf_filename}: {e}")
            return None

    def summarize_pdfs(self, uploaded_files, prompt, GROQ_API_KEY, VOYAGEAI_API_key, 
                      chunk_size=8000, chunk_overlap=500, similarity_threshold=0.8, 
                      num_clusters=10, start_section="methodology", output_format="docx"):
        summaries = {}
        total_files = len(uploaded_files)
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, uploaded_file in enumerate(uploaded_files):
            pdf_filename = uploaded_file.name
            status_text.text(f"Processing {pdf_filename} ({idx + 1}/{total_files})...")
            
            try:
                # Read PDF content
                pdf_bytes = uploaded_file.read()
                pdf_stream = BytesIO(pdf_bytes)

                text = self.extract_text_from_pdf(pdf_stream)

                # Skip processing if extracted text is empty
                if not text.strip():
                    st.warning(f"No text found in {pdf_filename}. Skipping...")
                    continue

                combined_text, _ = self.extract_sections(text, start_section=start_section)
                preprocessed_text = self.preprocess_text(combined_text)

                # Skip processing if preprocessed text is empty
                if not preprocessed_text.strip():
                    st.warning(f"No meaningful text after preprocessing for {pdf_filename}. Skipping...")
                    continue

                # Summarize the text
                summary = self.summarize_text(
                    preprocessed_text, 
                    selected_model=selected_model,  # Ensure 'selected_model' is defined in the scope
                    prompt=prompt, 
                    GROQ_API_KEY=GROQ_API_KEY, 
                    VOYAGEAI_API_key=VOYAGEAI_API_key, 
                    chunk_size=chunk_size, 
                    chunk_overlap=chunk_overlap, 
                    similarity_threshold=similarity_threshold, 
                    num_clusters=num_clusters
                )

                if not summary:
                    st.warning(f"Summary is empty for {pdf_filename}. Skipping...")
                    continue

                # Save the summary based on the chosen format
                if output_format == 'json':
                    summary_file = self.save_summary_as_json(summary, pdf_filename)
                    if summary_file:
                        summaries[pdf_filename] = summary_file
                elif output_format == 'csv':
                    summary_file = self.save_summary_as_csv(summary, pdf_filename)
                    if summary_file:
                        summaries[pdf_filename] = summary_file
                else:  # Default to docx
                    summary_file = self.save_summary_as_docx(summary, pdf_filename)
                    if summary_file:
                        summaries[pdf_filename] = summary_file

                st.success(f"Summary saved for {pdf_filename}")

            except Exception as e:
                st.error(f"An error occurred while processing {pdf_filename}: {e}")
                continue

            # Update progress bar
            progress = (idx + 1) / total_files
            progress_bar.progress(progress)

        status_text.text("Processing complete. Summaries generated.")
        progress_bar.empty()

        return summaries

# Initialize the summarizer
summarizer = PDFSummarizer()

# Initialize session state for summaries
if 'summaries' not in st.session_state:
    st.session_state.summaries = {}

# Streamlit App Layout
st.title("📄 Ecological Research Synthesis")
st.write("Upload your PDF files, select the desired options, and generate summaries.")

# Sidebar for Configuration
st.sidebar.header("Configuration")

# File Uploader
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    st.sidebar.success(f"{len(uploaded_files)} file(s) uploaded.")

# Model Selection
model_options = [
    "llama3-8b-8192", 
    "llama3-70b-8192", 
    "llama-3.2-1b-preview",
    "llama-3.2-3b-preview",
    "llama-3.2-11b-text-preview",
    "llama-3.2-90b-text-preview"
]
selected_model = st.sidebar.selectbox(
    "Select Model",
    options=model_options,
    index=0
)

# Start Section Selection
section_options = ["introduction", "methodology"]
start_section = st.sidebar.selectbox(
    "Start Summary From",
    options=section_options,
    index=1  # Default to 'methodology'
)

# Parameter Inputs
st.sidebar.subheader("Parameters")
similarity_threshold = st.sidebar.slider(
    "Similarity Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.8,
    step=0.05
)
chunk_size = st.sidebar.number_input(
    "Chunk Size",
    min_value=1000,
    max_value=10000,
    value=8000,
    step=500
)
chunk_overlap = st.sidebar.number_input(
    "Chunk Overlap",
    min_value=100,
    max_value=2000,
    value=500,
    step=100
)
num_clusters = st.sidebar.number_input(
    "Number of Clusters",
    min_value=2,
    max_value=100,
    value=10,
    step=1
)

# Output Format Selection
output_format = st.sidebar.selectbox(
    "Output Format",
    options=["docx", "json", "csv"],
    index=0
)

# API Keys Inputs
st.sidebar.subheader("API Keys")
GROQ_API_KEY = st.sidebar.text_input(
    "GROQ API Key",
    type="password"
)
VOYAGEAI_API_key = st.sidebar.text_input(
    "VOYAGEAI API Key",
    type="password"
)

# Prompt Input
st.sidebar.subheader("Prompt")
prompt = st.sidebar.text_area(
    "Enter the prompt for summarization:",
    value="Please provide your prompt."
)

# Start Processing Button
if st.sidebar.button("Start Processing"):
    if not uploaded_files:
        st.error("Please upload at least one PDF file.")
    elif not GROQ_API_KEY or not VOYAGEAI_API_key:
        st.error("Please provide both GROQ and VOYAGEAI API keys.")
    elif not prompt.strip():
        st.error("Please enter a prompt for summarization.")
    else:
        with st.spinner("Processing PDFs..."):
            summaries = summarizer.summarize_pdfs(
                uploaded_files=uploaded_files,
                prompt=prompt,
                GROQ_API_KEY=GROQ_API_KEY,
                VOYAGEAI_API_key=VOYAGEAI_API_key,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                similarity_threshold=similarity_threshold,
                num_clusters=num_clusters,
                start_section=start_section,
                output_format=output_format
            )
        
        if summaries:
            # Store summaries in session_state
            st.session_state.summaries.update(summaries)
            st.success("Summaries generated successfully!")
        else:
            st.warning("No summaries were generated.")

# Display Download Buttons from session_state
if st.session_state.summaries:
    st.header("Download Summaries")
    for pdf_filename, summary_file in st.session_state.summaries.items():
        if summary_file:
            if output_format == 'docx':
                st.download_button(
                    label=f"Download {pdf_filename} Summary (DOCX)",
                    data=summary_file,
                    file_name=f"Summary-{os.path.splitext(pdf_filename)[0]}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key=f"download_docx_{pdf_filename}"
                )
            elif output_format == 'json':
                st.download_button(
                    label=f"Download {pdf_filename} Summary (JSON)",
                    data=summary_file,
                    file_name=f"Summary-{os.path.splitext(pdf_filename)[0]}.json",
                    mime="application/json",
                    key=f"download_json_{pdf_filename}"
                )
            elif output_format == 'csv':
                st.download_button(
                    label=f"Download {pdf_filename} Summary (CSV)",
                    data=summary_file,
                    file_name=f"Summary-{os.path.splitext(pdf_filename)[0]}.csv",
                    mime="text/csv",
                    key=f"download_csv_{pdf_filename}"
                )