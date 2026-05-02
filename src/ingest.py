import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "tickets_mesa_ayuda.csv")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "mesa_ayuda_ti")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def build_collection(reset=True):
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except:
            pass

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

    possible_id_cols = ["ticket_id", "ticketid", "id"]
    id_col = next((c for c in possible_id_cols if c in df.columns), None)

    ids = []
    docs = []
    metas = []
    seen = set()

    for i, row in df.iterrows():
        if id_col:
            raw_id = str(row[id_col]).strip()
        else:
            raw_id = f"TCK-{i+1:03d}"

        if not raw_id or raw_id.lower() == "nan":
            raw_id = f"TCK-{i+1:03d}"

        if raw_id in seen:
            raw_id = f"{raw_id}-{i+1}"

        seen.add(raw_id)

        mensaje = str(row.get("mensaje_usuario", row.get("mensajeusuario", ""))).strip()
        categoria = str(row.get("categoria", "")).strip()
        subcategoria = str(row.get("subcategoria", "")).strip()
        prioridad = str(row.get("prioridad", "")).strip()
        sistema = str(row.get("sistema_afectado", row.get("sistemaafectado", ""))).strip()
        escalamiento = str(row.get("requiere_escalamiento", row.get("requiereescalamiento", ""))).strip()

        documento = f"""
Ticket: {mensaje}
Categoría: {categoria}
Subcategoría: {subcategoria}
Prioridad: {prioridad}
Sistema afectado: {sistema}
Requiere escalamiento: {escalamiento}
""".strip()

        metadata = {
            "ticket_id": raw_id,
            "categoria": categoria,
            "subcategoria": subcategoria,
            "prioridad": prioridad,
            "sistema": sistema,
            "escalamiento": escalamiento
        }

        ids.append(raw_id)
        docs.append(documento)
        metas.append(metadata)

    print("Columnas detectadas:", df.columns.tolist())
    print("Primeros IDs:", ids[:5])

    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metas
    )

    print(f"Colección creada con {len(ids)} documentos.")
    return collection, df


if __name__ == "__main__":
    build_collection(reset=True)