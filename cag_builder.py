from pathlib import Path
import json
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=Path(".env"))

knowledge_path = Path("knowledge.txt")
cache_path = Path("cache.json")

MODEL = "sabiazinho-4"

client = OpenAI(
    base_url="https://chat.maritaca.ai/api",
    api_key=os.getenv("MARITACA_API_KEY"),
)


def read_knowledge() -> str:
    if not knowledge_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado em: {knowledge_path}")

    text = knowledge_path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError("knowledge.txt vazio")

    return text


def build_cache(knowledge: str) -> dict:
    prompt = f"""
Voce vai transformar o texto abaixo em um cache de conhecimento para um sistema CAG.

REGRAS OBRIGATORIAS:
- Responda sempre em portugues do Brasil.
- Gere APENAS JSON valido.
- Nao use markdown.
- Nao use texto antes ou depois do JSON.
- Todas as chaves e valores textuais devem estar em portugues do Brasil.

Formato esperado:
{{
  "resumo": "resumo geral do conhecimento",
  "temas": ["tema 1", "tema 2"],
  "fatos": ["fato importante 1", "fato importante 2"],
  "perguntas_respostas": [
    {{
      "pergunta": "pergunta possivel",
      "resposta": "resposta baseada no conhecimento"
    }}
  ]
}}

Texto base:
{knowledge}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    raw_text = response.choices[0].message.content

    if not raw_text:
        raise ValueError("O modelo retornou uma resposta vazia.")

    json_start = raw_text.find("{")
    json_end = raw_text.rfind("}")

    if json_start == -1 or json_end == -1:
        raise ValueError(f"Resposta sem JSON valido: {raw_text}")

    return json.loads(raw_text[json_start:json_end + 1])


def save_cache(cache: dict) -> None:
    cache["_metadata"] = {
        "source": str(knowledge_path),
        "model": MODEL,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    cache_path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    knowledge = read_knowledge()
    cache = build_cache(knowledge)
    save_cache(cache)

    print("Cache gerado com sucesso.")


if __name__ == "__main__":
    main()
