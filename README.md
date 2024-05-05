# Document Assistant by Ash
This Python application creates a simple document assistant using Streamlit, pinecone (vector store) and a language model (openai) for generating responses to user queries.

## Command to download docs in html format
```wget --mirror --convert-links --adjust-extension --page-requisites --no-parent -P langchain-docs --execute robots=off --user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36" https://api.python.langchain.com/en/latest/```

### 1. main.py
This script creates a web application using Streamlit. It provides a user interface for document assistance, allowing users to submit questions and receive answers that include sources.

Key Features
- Streamlit Interface: Uses Streamlit for the UI, including input fields and session management.
- Source URL Sorting: Includes a function create_sources_string that takes a set of URLs, sorts them, and formats them as a numbered list.

### 2. ingestion.py
This script is responsible for ingesting and processing documents, possibly for a machine learning or search application. It uses environment variables and interfaces with a service called Pinecone.

Key Features
- Document Loading and Splitting: Loads documents from a specified source and splits them into chunks.
- Pinecone Integration: Inserts document vectors into a Pinecone vector store.

### 3. core.py
This file defines core functionalities for running language model queries using a retrieval system. It seems to be tightly integrated with Pinecone for vector storage and retrieval.

Key Features
- Environment Configuration: Loads environment variables.
- Query Processing: Provides a function run_llm that processes a query using a conversational chain with a language model and a document retriever.