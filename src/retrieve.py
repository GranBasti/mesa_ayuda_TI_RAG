import os
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "mesa_ayuda_ti")
TOP_K = int(os.getenv("TOP_K", "4"))

embedder = SentenceTransformer(EMBED_MODEL)
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)


def retrieve_context(query: str, top_k: int = TOP_K):
    query_embedding = embedder.encode(query).tolist()

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    docs = result["documents"][0]
    metas = result["metadatas"][0]
    distances = result["distances"][0]

    return list(zip(docs, metas, distances))


if __name__ == "__main__":
    query = "No puedo entrar a Moodle con mis credenciales institucionales"
    contexts = retrieve_context(query)

    for doc, meta, dist in contexts:
        print("-" * 60)
        print(meta)
        print(dist)
        print(doc)