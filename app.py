import os
import streamlit as st
import pickle
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env

st.title("LifeBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
index_file_path = "faiss_index.pkl"
docstore_file_path = "docstore.pkl"
model_file_path = "model.pkl"

main_placeholder = st.empty()

if process_url_clicked and urls:
    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Create sentence embeddings using SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [doc.page_content for doc in docs]
    embeddings = model.encode(texts)

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Create document store
    docstore = {i: Document(page_content=texts[i]) for i in range(len(texts))}

    # Save the FAISS index, document store, and model separately to pickle files
    with open(index_file_path, "wb") as f:
        pickle.dump(index, f)
    with open(docstore_file_path, "wb") as f:
        pickle.dump(docstore, f)
    with open(model_file_path, "wb") as f:
        pickle.dump(model, f)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(index_file_path) and os.path.exists(docstore_file_path) and os.path.exists(model_file_path):
        with open(index_file_path, "rb") as f:
            index = pickle.load(f)
        with open(docstore_file_path, "rb") as f:
            docstore = pickle.load(f)
        with open(model_file_path, "rb") as f:
            model = pickle.load(f)

        # Load a QA model from Hugging Face
        qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")


        # Function to retrieve the most relevant context
        def retrieve_context(query, k=5):
            D, I = index.search(model.encode([query]), k)
            return " ".join([docstore[i].page_content for i in I[0]])


        # Function to answer questions
        def answer_question(query):
            context = retrieve_context(query)
            result = qa_model(question=query, context=context)
            return result['answer']


        # Get the answer to the query
        answer = answer_question(query)
        st.header("Answer")
        st.write(answer)

