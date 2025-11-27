from chromadb import PersistentClient
from app.settings import settings

print("DB PATH:", settings.CHROMA_DB_PATH)

client = PersistentClient(path=settings.CHROMA_DB_PATH)

collections = client.list_collections()
print("Collections:", [c.name for c in collections])

if not collections:
    print("No collections found.")
    exit()

col = client.get_collection("multi_agent_rag_collection")

result = col.get(
    include=["documents", "embeddings", "metadatas"]
)

print("IDs:", len(result["ids"]))
print("Docs:", len(result["documents"]))
print("Metadatas:", len(result["metadatas"]))
print("Embeddings:", len(result["embeddings"]))
