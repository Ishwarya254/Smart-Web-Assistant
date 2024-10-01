import os
import streamlit as st
import pickle
import time
import langchain
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()


st.title("Smart Web Assistant! - Be Smarter")
st.sidebar.title("Web Links")

urls = []

for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

proceed_url_clicked = st.sidebar.button("Proceed")
file_path = "vector_index.pkl"

main_placefolder = st.empty()
llm = ChatGroq(model_name = 'llama-3.1-70b-versatile')

if proceed_url_clicked:
    #load data
    loader = WebBaseLoader(urls)
    main_placefolder.text("Processing...")
    data = loader.load()
    
    #split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators= ['\n\n', '\n', '.', ','],
        chunk_size = 1000
    )

    main_placefolder.text("Processing...")
    docs = text_splitter.split_documents(data)

    #create embeddings and save it to FAISS index

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorindex = FAISS.from_documents(docs, embeddings)

    main_placefolder.text("Processing...")
    time.sleep(2)

    # Storing vector index create in local
    with open(file_path, "wb") as f:
        pickle.dump(vectorindex, f)


query = main_placefolder.text_input("Questions please!!!")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorindex = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever = vectorindex.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)

            # {"answer": "", "sources": []}
            st.header("Answer")
            st.write(result["answer"])

            #dsiplay sources, if availabel
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n") #split the sources by newline
                for source in sources_list:
                    st.write(source)


