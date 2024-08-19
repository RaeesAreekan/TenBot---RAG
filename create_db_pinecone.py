from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
import openai
from dotenv import load_dotenv
import os
import pandas as pd
import time

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

load_dotenv()

DATA_PATH = 'Data/'

openai.api_key = os.environ['OPENAI_API_KEY']

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents(batch=1)
    chunks = split_text(documents)
    save_to_pinecone(chunks)


def load_documents(batch:int , total_batches:int=2):
    filenames = [f for f in os.listdir(DATA_PATH) if f.endswith(".txt")]

    filenames.sort()

    batch_size = len(filenames) // total_batches
    remainder = len(filenames) % total_batches

    start_index = batch * batch_size + min(batch, remainder)
    end_index = start_index + batch_size + (1 if batch < remainder else 0)

    batch_filenames = filenames[start_index:end_index]




    documents = []
    for filename in batch_filenames:
        if filename.endswith(".txt"):
            file_path = os.path.join(DATA_PATH, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                documents.append(Document(page_content=content, metadata={"source": filename}))
    print(f"Loaded {len(documents)} documents.")
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_pinecone(chunks: list[Document]):
    pinecone = Pinecone(api_key=PINECONE_API_KEY)

    index_name = 'tensors-chat'

    existing_indexes = [
        index_info["name"] for index_info in pinecone.list_indexes()
    ]

    if index_name not in existing_indexes:
        pinecone.create_index(
            index_name,
            dimension=1536,
            metric='dotproduct',
            spec=ServerlessSpec(

                cloud="aws", 
                region="us-east-1"
        ) 
        )
        # Wait for index to be initialized
        while not pinecone.describe_index(index_name).status['ready']:
            time.sleep(1)

    index = pinecone.Index(index_name)
    time.sleep(1)
    index.describe_index_stats()

    embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")

    ids = [f"{chunk.metadata['source']}-{i}" for i, chunk in enumerate(chunks)]
    texts = [chunk.page_content for chunk in chunks]
    embeds = embed_model.embed_documents(texts)

    metadata = [{'text': chunk.page_content, 'source': chunk.metadata['source']} for chunk in chunks]
    
    vectors = list(zip(ids, embeds, metadata))

    index.upsert(vectors=vectors)
    print(f"Saved {len(chunks)} chunks to Pinecone index.")

if __name__ == "__main__":
    main()



