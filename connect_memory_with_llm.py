import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Step 1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN = os.getenv("HF_TOKEN")  # Ensure HF_TOKEN is set in your environment
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )
    return llm

# Step 2: Custom Prompt Template
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Step 3: Load Precomputed FAISS Index
DB_FAISS_PATH = "vectorstore/db_faiss"  # Ensure this is the correct path
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

try:
    # Enable dangerous deserialization explicitly
    db = FAISS.load_local(DB_FAISS_PATH, embeddings=embeddings, allow_dangerous_deserialization=True)
    print("FAISS database loaded successfully!")
except Exception as e:
    print(f"Error loading FAISS database: {e}")
    exit()

# Step 4: Create RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Step 5: Handle User Query
user_query = input("Write Query Here: ")
try:
    response = qa_chain.invoke({'query': user_query})
    print("RESULT: ", response["result"])
    print("SOURCE DOCUMENTS: ", response["source_documents"])
except Exception as e:
    print(f"Error during QA invocation: {e}")
