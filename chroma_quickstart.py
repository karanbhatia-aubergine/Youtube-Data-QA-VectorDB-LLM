import csv

import chromadb
from chromadb.utils import embedding_functions
from decouple import config

with open('menu_items.csv') as file:
    lines = csv.reader(file)

    documents = []

    metadatas = []

    ids = []
    id = 1

    for i, line in enumerate(lines):
        if i==0:
            continue

        documents.append(line[1])
        metadatas.append({"item_id": line[0]})
        ids.append(str(id))
        id+=1


collection_name = config('COLLECTION_NAME')
chroma_client_host = config('CHROMA_CLIENT_HOST')
chroma_client = chromadb.HttpClient(host=chroma_client_host, port=8000)

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")

collection = chroma_client.get_or_create_collection(name="my_collection", embedding_function=sentence_transformer_ef)

collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

results = collection.query(
    query_texts=["vermiceli"],
    n_results=5,
    include=['documents', 'distances', 'metadatas']
)
print(results['documents'])
print(results['distances'])

results = collection.query(
    query_texts=["donut"],
    n_results=5,
    include=['documents', 'distances', 'metadatas']
)
print(results['documents'])
print(results['distances'])

results = collection.query(
    query_texts=["shrimp"],
    n_results=5,
    include=['documents', 'distances', 'metadatas']
)
print(results['documents'])
print(results['distances'])