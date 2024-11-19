import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from transformers import pipeline

GROQ_API_KEY = "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" 
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model="llama3-8b-8192")


st.title("EcoLens - Your Personal Carbon Footprint Assistant")
st.write("Enter your daily activities or consumption patterns, along with any questions about reducing your CO2 emissions.")

user_input = st.text_area("Enter your daily activities or consumption patterns:")
question = st.text_input("Enter your question about CO2 emissions reduction:")


def answer_query(question, context):
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=2000, chunk_overlap=200)
    documents = text_splitter.create_documents([context])

    embeddings = HuggingFaceEmbeddings()

    faiss_index = FAISS.from_documents(documents, embeddings)

    prompt_template = """
    You are an AI assistant designed to help individuals understand and reduce their personal CO2 emissions. The user will provide details about their daily activities or consumption patterns, and you should calculate the CO2 emissions based on this information. Additionally, provide personalized suggestions for reducing their carbon footprint.
    
    The model must ONLY reply based on the information given by the user. Do not generate anything else on your own. The answer must be highly relevant, actionable, and tailored to the user's context.

    User input: {context}
    Question: {question}
    Emission calculation and personalized reduction strategies:
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=faiss_index.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT}
    )

    response = qa_chain({"context": context, "question": question, "query": question})
    return response["result"]

if st.button("Get Answer"):
    if user_input and question:
        response = answer_query(question, user_input)
        st.write("**Response:**", response)
    else:
        st.write("Please provide both your activities/consumption patterns and a question.")
