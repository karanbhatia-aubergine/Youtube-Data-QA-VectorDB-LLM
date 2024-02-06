from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from decouple import config

directory = 'test-documents'

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

documents = load_docs(directory)
len(documents)


def split_docs(documents,chunk_size=1000,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
print(len(docs))

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
collection_name = config('COLLECTION_NAME')
chroma_client_host = config('CHROMA_CLIENT_HOST')
# chroma_client = chromadb.HttpClient(host=chroma_client_host, port=8000)

db = Chroma.from_documents(docs, embeddings)

query = "What are the different kinds of test-documents people commonly own?"
matching_docs = db.similarity_search(query)

matching_docs[0]


persist_directory = "chroma_db"

vectordb = Chroma.from_documents(
    documents=docs, embedding=embeddings, persist_directory=persist_directory
)

vectordb.persist()
