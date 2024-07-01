import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams

# Đặt API key của OpenAI
os.environ["OPENAI_API_KEY"] = ''
embeddings = OpenAIEmbeddings()

client = QdrantClient(url='http://localhost:6333')

# Kiểm tra và tạo collection nếu chưa tồn tại
def create_collection(client, collection_name):
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance='Cosine')  # Ensure the size matches the output dimension of your embeddings
        )
    else:
        print(f"Collection {collection_name} already exists.")

def text_load(files):
    documents = []
    for file in files:
        loader = TextLoader(file, encoding='utf-8')
        docs = loader.load()
        documents.extend(docs)
    return documents

def get_chunk(documents):
    text_split = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=48)
    split_txt = text_split.split_documents(documents)
    return split_txt

def vector_data(text_chunks, collection_name):
    vectors = embeddings.embed_documents([chunk.page_content for chunk in text_chunks])
    vector_store = client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=idx,
                vector=vectors[idx],
                payload={"text": text_chunks[idx].page_content}
            )
            for idx in range(len(text_chunks))
        ]
    )
    return vector_store

def retrieve_knowledge(query, collection_name):
    # Generate embeddings for the query
    query_vector = embeddings.embed_query(query)

    # Connect to Qdrant and search for the top 3 closest vectors
    file_txt = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=3
    )
    return file_txt

# Example usage
# Retrieve knowledge
# results = retrieve_knowledge("Xác định 5W - 1H", collection_name)
# print(results)
