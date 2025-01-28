MediBot 
MediBot is a semantic search-based chatbot designed to retrieve and summarize information from medical documents. It leverages FAISS for efficient similarity search, Hugging Face models for embedding and text generation, and Streamlit for an intuitive conversational interface. MediBot provides users with accurate and context-aware answers to their queries based on pre-indexed medical documents.

Features
Semantic Search: Retrieve documents based on meaning, not just keywords.
Efficient Retrieval: Uses FAISS for high-performance similarity searches.
Hugging Face Models: Generates embeddings and responses using cutting-edge NLP models.
Streamlit Chatbot: User-friendly web-based chatbot interface.
PDF Document Processing: Preprocesses and indexes PDF files for fast querying.
Custom Prompts: Customizes LLM behavior for domain-specific accuracy.

Project Workflow
Document Preparation:
PDFs are loaded from the data/ folder and split into smaller chunks for better indexing.
Embedding Creation:
Text chunks are converted into dense vector embeddings using sentence-transformers.
Indexing with FAISS:
Embeddings are stored in a FAISS index for fast similarity-based retrieval.
Query Handling:
User queries are embedded and matched against the indexed documents.
Response Generation:
Relevant document chunks are provided as context to a Hugging Face language model, which generates accurate answers.
Streamlit Chat Interface:
Users interact with the system through a simple, real-time chatbot interface.

├── data/
│   └── [Your PDF files]
├── vectorstore/
│   └── db_faiss        # FAISS index storage
├── create_memory_for_llm.py
├── connect_memory_with_llm.py
├── medibot.py
├── requirements.txt
├── README.md
└── .env
This project requires an .env file to securely store sensitive credentials like the Hugging Face API token. The .env file is automatically loaded using the python-dotenv package.

Steps to Configure the .env File
Create the .env File:

In the project root directory, create a file named .env.
Add the Hugging Face API Token:

Obtain your Hugging Face API token from your Hugging Face account:
Log in to Hugging Face.
Navigate to your account settings → Access Tokens.
Generate a new token (select the read scope).
Add the token to the .env file as follows:
makefile

HF_TOKEN=your_huggingface_api_token
Ensure the .env File Is Secure:
.




