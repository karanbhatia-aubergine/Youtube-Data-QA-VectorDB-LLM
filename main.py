# import psycopg2
# import chromadb
# from gensim.models import Word2Vec
# from decouple import config
# import numpy as np
# from chromadb.utils import embedding_functions
#
# # Function to chunk text into smaller pieces
# def chunk_text(text, chunk_size=500):
#     chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
#     return chunks
#
# try:
#     conn = psycopg2.connect(database=config("DATABASE_NAME"), user=config("DATABASE_USERNAME"), password=config("DATABASE_PASSWORD"), host=config("DATABASE_HOST"), port=config("DATABASE_PORT"))
#     cursor = conn.cursor()
#     cursor.execute("""
#         SELECT video_id, transcript, summary
#         FROM "summarizeVideo"
#     """)
#     data = cursor.fetchall()
#     conn.close()
# except psycopg2.DatabaseError as database_error:
#     print("Issue in database:", database_error)
#
# all_data = []
# for entry in data:
#     video_id, transcript, summary = entry
#
#     # Clean and format transcript and summary
#     transcript_chunks = [chunk_text(chunk) for chunk in transcript.split('\n')]
#     transcript = " ".join([" ".join(chunk) for chunk in transcript_chunks])
#     summary_chunk = " ".join(summary.replace(" ", "").split('\n'))
#
#     all_data.append({
#         "video": video_id,
#         "transcript": transcript,
#         "summary": summary_chunk
#     })
#
# collection_name = config('COLLECTION_NAME')
# chroma_client_host = config('CHROMA_CLIENT_HOST')
# chroma_client = chromadb.HttpClient(host=chroma_client_host, port=8000)
# try:
#     chroma_client.delete_collection(name=collection_name)
# except:
#     pass
# distance_functions = ["l2","ip","cosine"]
# sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="multi-qa-mpnet-base-dot-v1")
# collection = chroma_client.create_collection(name=collection_name,embedding_function=sentence_transformer_ef, metadata={"hnsw:space": distance_functions[0]})
# documents = []
# metadata = []
# ids = []
#
#
# # Prepare data for ChromaDB collection
# documents = [data['summary'] for data in all_data]
# metadata = [{"transcript": data['transcript']} for data in all_data]
# ids = [data['video'] for data in all_data]
#
# collection.add(
#     documents=documents,
#     metadatas=metadata,
#     ids=ids
# )
#
# result = collection.query(
#     query_texts=["hello","welcome"],
#     n_results=1,
#     include=['documents', 'uris', 'metadatas', 'distances',]
#
# )
# print(result)
#
# print("Done")


import psycopg2
import chromadb
from decouple import config
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# Function to chunk text into smaller pieces
def chunk_text(text, chunk_size=500):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

try:
    conn = psycopg2.connect(database=config("DATABASE_NAME"), user=config("DATABASE_USERNAME"),
                            password=config("DATABASE_PASSWORD"), host=config("DATABASE_HOST"),
                            port=config("DATABASE_PORT"))
    cursor = conn.cursor()
    cursor.execute("""
        SELECT video_id, transcript, summary
        FROM "summarizeVideo"
    """)
    data = cursor.fetchall()
    conn.close()
except psycopg2.DatabaseError as database_error:
    print("Issue in database:", database_error)

all_data = []
for entry in data:
    video_id, transcript, summary = entry

    # Clean and format transcript and summary
    transcript_chunks = [chunk_text(chunk) for chunk in transcript.split('\n')]
    transcript = " ".join([" ".join(chunk) for chunk in transcript_chunks])
    summary_chunk = " ".join(summary.replace(" ", "").split('\n'))

    all_data.append({
        "video": video_id,
        "transcript": transcript,
        "summary": summary_chunk
    })

collection_name = config('COLLECTION_NAME')
chroma_client_host = config('CHROMA_CLIENT_HOST')
chroma_client = chromadb.HttpClient(host=chroma_client_host, port=8000)

try:
    chroma_client.delete_collection(name=collection_name)
except:
    pass

distance_functions = ["l2", "ip", "cosine"]
sentence_transformer_model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="multi-qa-mpnet-base-dot-v1")
documents = []
metadata = []
ids = []
embeddings = []

# Prepare data for ChromaDB collection
distance_functions = ["l2", "ip", "cosine"]
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="multi-qa-mpnet-base-dot-v1")
collection = chroma_client.create_collection(name=collection_name, embedding_function=sentence_transformer_ef, metadata={"hnsw:space": distance_functions[0]})
documents = []
metadata = []
ids = []

for data in all_data:
    documents.append(data['transcript'])
    metadata.append({"video_id": data['video']})
    ids.append(data['video'])

collection.add(
    documents=documents,
    metadatas=metadata,
    ids=ids
)


# Query ChromaDB collection
result = collection.query(
    query_texts=["any programming language you'll"],
    n_results=1,
    include=['documents', 'uris', 'metadatas', 'distances']
)
print(result)

print("Done")
