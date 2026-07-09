pip install mcp mcp pymongo python-dotenv

from os import getenv
from dotenv import load_dotenv
from pymongo import MongoClient
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("mongo-mcp")

client = MongoClient(getenv("MONGO_URI"))

@mcp.tool()
def list_databases() -> list[str]:
    """Lista os bancos disponiveis no MongoDB."""
    return client.list_database_names()


@mcp.tool()
def list_collections(database: str) -> list[str]:
    """Lista as collections de um database."""
    db = client[database]
    return db.list_collection_names()


@mcp.tool()
def count_documents_mongo(database: str, collection: str, filtro: dict | None = None) -> int:
    col = client[database][collection]
    return col.count_documents(filtro or {})


@mcp.tool()
def find_documents_mongo(database: str, collection: str, filtro: dict, limite: int = 5) -> list[dict]:
    """Busca documentos em uma collection usando um filtro MongoDB."""
    limite = min(limite, 20)
    db = client[database]
    col = db[collection]

    docs = list(col.find(filtro).limit(limite))

    for doc in docs:
        doc["_id"] = str(doc["_id"])

    return docs


if __name__ == "__main__":
    mcp.run()