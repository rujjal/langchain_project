import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import  PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

# import os

# os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
    embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conservation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key = "Chat-history", return_messages= True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory= memory
    )
    return conversation_chain


def main():
    load_dotenv()
    st.set_page_config("Chatbot for married people", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

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
                #create conservation chain
                st.session_state.conversation = get_conservation_chain(vectorstore)


if __name__ == '__main__':
    main()