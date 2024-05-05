import os
from dotenv import load_dotenv
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# Using SDK
from langchain_community.vectorstores import Pinecone as PineconeLangChain

# Extra functionality
from pinecone import Pinecone

load_dotenv()

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))


def ingest_docs() -> None:
    loader = ReadTheDocsLoader(path="langchain-docs/api.python.langchain.com/en/latest")
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} documents.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Split into {len(documents)} Chunks.")

    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to insert {len(documents)} vectors to Pinecone.")
    embeddings = OpenAIEmbeddings()
    PineconeLangChain.from_documents(
        documents=documents, embedding=embeddings, index_name="langchain-doc-index"
    )
    print("******** Added to Pinecone Vector Store *********")


if __name__ == "__main__":
    print("Ingesting...")
    ingest_docs()
