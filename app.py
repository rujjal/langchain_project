import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import  PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

import warnings
from urllib3.exceptions import InsecureRequestWarning

warnings.filterwarnings("ignore", category=InsecureRequestWarning)



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator= "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name = "bert-base-uncased")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def main():
    load_dotenv()
    st.set_page_config("Chatbot for married people", page_icon=":books:")

    st.header("Chatbot for married people :books:")
    st.text_input("Ask a question ")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload ur pdf file", accept_multiple_files=True)
        if st.button("Analyze"):
            with st.spinner("processing"):
            
                #get pdf text
                raw_text = get_pdf_text(pdf_docs)
                #st.write(raw_text)
                #get the text chuncks
                text_chunks = get_text_chunks(raw_text)
                #st.write(text_chunks)
                #create vector store
                vectorstore = get_vectorstore(text_chunks)
                #vectorstore.save("vectorstore.faiss")

if __name__ == '__main__':
    main()