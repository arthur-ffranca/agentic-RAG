from pathlib import Path
import json
import requests
import os
from datetime import datetime, timezone
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


knowledge_path = r'C:\Users\otavi\OneDrive\Documentos\RAG + CAG + MCP\knowledge.txt'
cache_path = r'C:\Users\otavi\OneDrive\Documentos\RAG + CAG + MCP\cache.json'

MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")

client = OpenAI(
    base_url="http://localhost:11434",
    api_key="admin",
)

def read_knowledge() -> str:
    if not knowledge_path.exists():
        return FileNotFoundError(f"Arquivo não encontrado em: {knowledge_path}")

    text = knowledge_path.read_text(encoding='utf-8').strip()

    if not text:
        raise ValueError("knowledge.txt vazio")

    return text


def build_cache(knowledge: str) -> dict:
    prompt = f"""
Voce vai transformar o texto abaixo em um cache de conhecimento para um sistema CAG.

Gere APENAS JSON valido, sem markdown.

Formato esperado:
{{
  "summary": "resumo geral do conhecimento",
  "topics": ["tema 1", "tema 2"],
  "facts": ["fato importante 1", "fato importante 2"],
  "qa_pairs": [
    {{
      "question": "pergunta possivel",
      "answer": "resposta baseada no conhecimento"
    }}
  ]
}}

Texto base:
{knowledge}
"""

    response = client.responses.create(
        model=MODEL,
        input=prompt,
    )

    
    raw_text = response.output_text
    return json.loads(raw_text)

def save_cache(cache: dict) -> None:
    cache["_metadata"] = {
        "source": str(knowledge_path),
        "model": MODEL,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    cache_path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

def main():
    knowledge = read_knowledge()
    cache = build_cache(knowledge)
    save_cache(cache)

    print(f"Cache gerado com sucesso.")

if __name__ == "__main__":
    main()