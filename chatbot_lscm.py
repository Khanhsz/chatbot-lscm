
import streamlit as st
import pytesseract
from PIL import Image
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="LSCM Chatbot", layout="wide")
st.title("📦 Chatbot Giải Bài Tập LSCM – Sách + Đề + Hình Ảnh")

openai_key = os.getenv("OPENAI_API_KEY")
VECTORSTORE_PATH = "lscm_vectorstore"

@st.cache_resource
def load_vectorstore():
    return FAISS.load_local(VECTORSTORE_PATH, OpenAIEmbeddings())

def extract_text_from_image(uploaded_img):
    image = Image.open(uploaded_img)
    return pytesseract.image_to_string(image)

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(temperature=0.2)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

tab1, tab2 = st.tabs(["📄 Nhập câu hỏi", "🖼️ Upload ảnh câu hỏi"])
with tab1:
    question = st.text_input("Nhập câu hỏi:")
    if question:
        st.write("📘 Trả lời:")
        st.write(qa_chain.run(question))
with tab2:
    uploaded_img = st.file_uploader("Tải ảnh câu hỏi", type=["jpg", "png"])
    if uploaded_img:
        text = extract_text_from_image(uploaded_img)
        st.code(text)
        if st.button("Trả lời từ ảnh"):
            st.write(qa_chain.run(text))
