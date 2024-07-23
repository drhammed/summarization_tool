# PDF Research Paper Summarizer

This project is a PDF Research Paper Summarizer that uses Natural Language Processing (NLP) techniques to extract and summarize key sections from research papers. The summarizer focuses on the methodology, results, discussion, and conclusion sections, providing a high-level summary of the key findings and conclusions (although you could extend to cover introduction or other parts of the paper).

## Features

- **PDF Extraction:** Extract text content from PDF files.
- **Text Preprocessing:** Clean and preprocess the extracted text for better summarization.
- **Section Extraction:** Identify and extract specific sections from the research paper.
- **Text Summarization:** Generate high-level summaries of the extracted sections using OpenAI's GPT-4 model.
- **Streamlit Interface:** A user-friendly web interface for uploading PDF files and displaying summaries.

## Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/yourusername/pdf-research-paper-summarizer.git
   cd pdf-research-paper-summarizer



## Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

## Install the required packages:
`pip install -r requirements.txt`

## Download NLTK data:
`python -m nltk.downloader punkt wordnet`

## Configuration

1. Google Drive API Credentials:

- Create a project on the (Google Cloud Console).

- Enable the Google Drive API.

- Create credentials (OAuth 2.0 Client IDs) and download the credentials.json file.

- Place the credentials.json file in the project directory. For a full instruction on this, see my (GDriveOps python package0[https://pypi.org/project/GDriveOps/]


2. OpenAI API Key:

Obtain an API key from (OpenAI)[https://platform.openai.com/apps].

Set the environment variable My_OpenAI_API_key with your API key.

`export My_OpenAI_API_key='your_openai_api_key'`


## Usage

1. Run the Streamlit app:
   
   `streamlit run app.py`

2. Upload a PDF File:

- Open the web interface.
- Upload a PDF file containing the research paper.


3. Generate Summary:

- Wait for the text extraction and summarization to complete.
- View the generated summary on the web interface.


## Acknowledgments

- This project uses the OpenAI GPT-4 model for text summarization.
- The project is built using the Streamlit framework for the web interface.
- Thanks to the Google Drive API for providing the tools to interact with Google Drive.

