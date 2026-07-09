from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from pydantic import BaseModel

import json
import os
import time


load_dotenv(dotenv_path=Path(".env"))

MARITACA_BASE_URL = "https://chat.maritaca.ai/api"
DEFAULT_MODEL = "sabiazinho-4"
DEFAULT_MAX_RETRIES = 3

MODEL_PRICES_BRL_PER_1M = {
    "sabiazinho-4": {"input": 1.00, "output": 4.00},
    "sabia-4": {"input": 5.00, "output": 20.00},
    "sabia-4-thinking": {"input": 5.00, "output": 40.00},
}


class ToolScore(BaseModel):
    tool: Literal["cag", "rag"]
    expected_value: float
    estimated_cost: float
    reason: str


class GraphPlan(BaseModel):
    selected_tools: list[Literal["cag", "rag"]]
    scores: list[ToolScore]
    reason: str


class JudgeDecision(BaseModel):
    is_enough: bool
    reason: str
    missing_info: str | None = None


@dataclass
class ModelCall:
    text: str
    usage: dict[str, int]
    elapsed: float
    estimated_cost_brl: float


def _make_client() -> OpenAI:
    api_key = os.getenv("MARITACA_API_KEY")
    if not api_key:
        raise RuntimeError("MARITACA_API_KEY nao encontrada no .env")

    return OpenAI(base_url=MARITACA_BASE_URL, api_key=api_key)


client = _make_client()

_db: Chroma | None = None


def get_vector_db() -> Chroma:
    global _db
    if _db is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        _db = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings,
        )
    return _db


def extract_json(raw_text: str) -> dict[str, Any]:
    json_start = raw_text.find("{")
    json_end = raw_text.rfind("}")

    if json_start == -1 or json_end == -1:
        raise ValueError(f"Resposta sem JSON valido: {raw_text}")

    return json.loads(raw_text[json_start:json_end + 1])


def estimate_cost_brl(model: str, usage: dict[str, int]) -> float:
    prices = MODEL_PRICES_BRL_PER_1M.get(model)
    if not prices:
        return 0.0

    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)

    return (
        input_tokens * prices["input"] / 1_000_000
        + output_tokens * prices["output"] / 1_000_000
    )


def call_model(model: str, prompt: str) -> ModelCall:
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = round(time.time() - start, 2)
    text = response.choices[0].message.content or ""

    usage_obj = getattr(response, "usage", None)
    usage = {
        "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
    }

    return ModelCall(
        text=text,
        usage=usage,
        elapsed=elapsed,
        estimated_cost_brl=estimate_cost_brl(model, usage),
    )


def empty_graph_state(question: str, model: str) -> dict[str, Any]:
    return {
        "question": question,
        "model": model,
        "plan": None,
        "tool_results": [],
        "aggregated_context": None,
        "judge": None,
        "final_answer": None,
        "logs": [],
        "attempts": [],
        "total_elapsed": 0.0,
        "total_estimated_cost_brl": 0.0,
        "total_tokens": 0,
    }


def add_log(
    state: dict[str, Any],
    node: str,
    elapsed: float,
    output: Any,
    input_data: Any | None = None,
    usage: dict[str, int] | None = None,
    estimated_cost_brl: float = 0.0,
) -> None:
    usage = usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    state["logs"].append(
        {
            "node": node,
            "elapsed": elapsed,
            "input": input_data,
            "output": output,
            "usage": usage,
            "estimated_cost_brl": round(estimated_cost_brl, 8),
        }
    )
    state["total_estimated_cost_brl"] += estimated_cost_brl
    state["total_tokens"] += usage.get("total_tokens", 0)


def plan_node(question: str, state: dict[str, Any], model: str) -> GraphPlan:
    prompt = f"""
PAPEL:
Voce e o PLANNER de um sistema chamado Maritaca Hybrid Graph Agentic RAG.

OBJETIVO:
Escolher quais ferramentas devem ser usadas para responder a pergunta do usuario.

CONTEXTO:
O sistema possui duas memorias/ferramentas:
- CAG para conhecimento cacheado.
- RAG para documentos e evidencias textuais.

OPCOES DISPONIVEIS:
- cag
- rag

REGRAS:
- Se a pergunta for simples e ligada a FAQ, regras, horarios, planos ou precos, escolha ["cag"].
- Se a pergunta pedir documentos, contratos, PDFs, manuais ou evidencias textuais, escolha ["rag"].
- Se a pergunta precisar comparar conhecimento cacheado com documentos, escolha ["cag", "rag"].
- Nao escolha todas as ferramentas sem necessidade.
- Sempre gere score para cag e rag.
- expected_value deve ir de 0.0 a 1.0.
- estimated_cost deve ir de 0.0 a 1.0.

ENTRADA:
Pergunta do usuario:
{question}

FORMATO DE SAIDA:
Responda APENAS JSON valido.
Nao use markdown.
Nao escreva texto antes ou depois do JSON.
O JSON deve ter exatamente este formato:
{{
  "selected_tools": ["cag"],
  "scores": [
    {{"tool": "cag", "expected_value": 0.9, "estimated_cost": 0.1, "reason": "motivo curto"}},
    {{"tool": "rag", "expected_value": 0.2, "estimated_cost": 0.4, "reason": "motivo curto"}}
  ],
  "reason": "motivo geral da selecao"
}}
"""
    call = call_model(model, prompt)
    plan = GraphPlan(**extract_json(call.text))

    add_log(
        state,
        "plan_node",
        call.elapsed,
        plan.model_dump(),
        {"question": question},
        call.usage,
        call.estimated_cost_brl,
    )
    return plan


def cag_node(question: str, state: dict[str, Any], model: str) -> dict[str, Any]:
    start = time.time()
    cache = json.loads(Path("cache.json").read_text(encoding="utf-8"))

    prompt = f"""
PAPEL:
Voce e a ferramenta CAG.

OBJETIVO:
Buscar no cache de conhecimento informacoes relevantes para responder a pergunta.

CONTEXTO:
{json.dumps(cache, ensure_ascii=False, indent=2)}

REGRAS:
- Use apenas o cache.
- Se nao encontrar informacao relevante, diga isso claramente.
- Responda em portugues do Brasil.
- Seja objetivo.

ENTRADA:
Pergunta do usuario:
{question}

FORMATO DE SAIDA:
Texto normal com as informacoes relevantes encontradas no cache.
"""
    call = call_model(model, prompt)
    elapsed = round(time.time() - start, 2)
    output = {
        "tool": "cag",
        "query": question,
        "result": call.text,
        "elapsed": elapsed,
    }
    add_log(
        state,
        "cag_node",
        elapsed,
        output,
        {"question": question},
        call.usage,
        call.estimated_cost_brl,
    )
    return output


def rag_node(question: str, state: dict[str, Any], model: str) -> dict[str, Any]:
    start = time.time()
    db = get_vector_db()
    docs = db.similarity_search(question, k=3)
    rag_context = "\n\n".join([doc.page_content for doc in docs])
    elapsed = round(time.time() - start, 2)
    output = {
        "tool": "rag",
        "query": question,
        "result": rag_context,
        "elapsed": elapsed,
    }
    add_log(state, "rag_node", elapsed, output, {"question": question})
    return output


TOOL_NODES = {
    "cag": cag_node,
    "rag": rag_node,
}


def run_parallel_tools(
    question: str,
    selected_tools: list[str],
    state: dict[str, Any],
    model: str,
) -> list[dict[str, Any]]:
    start = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=max(1, len(selected_tools))) as executor:
        futures = {
            executor.submit(TOOL_NODES[tool], question, state, model): tool
            for tool in selected_tools
        }
        for future in as_completed(futures):
            tool = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = {
                    "tool": tool,
                    "query": question,
                    "result": f"Erro ao executar {tool}: {error}",
                    "elapsed": None,
                    "error": True,
                }
            results.append(result)

    elapsed = round(time.time() - start, 2)
    add_log(
        state,
        "parallel_tools",
        elapsed,
        results,
        {"selected_tools": selected_tools, "question": question},
    )
    return results


def aggregator_node(tool_results: list[dict[str, Any]], state: dict[str, Any]) -> str:
    start = time.time()
    parts = []
    for item in tool_results:
        parts.append(
            f"""
FONTE: {item["tool"]}
QUERY: {item["query"]}
RESULTADO:
{item["result"]}
"""
        )
    aggregated_context = "\n---\n".join(parts)
    elapsed = round(time.time() - start, 2)
    add_log(state, "aggregator_node", elapsed, aggregated_context, tool_results)
    return aggregated_context


def judge_node(
    question: str,
    aggregated_context: str,
    state: dict[str, Any],
    model: str,
) -> JudgeDecision:
    prompt = f"""
PAPEL:
Voce e um juiz de suficiencia.

TAREFA:
Verificar se o contexto abaixo e suficiente para responder a pergunta do usuario.

IMPORTANTE:
Voce NAO deve responder a pergunta.
Voce deve apenas avaliar se da para responder com o contexto disponivel.

PERGUNTA:
{question}

CONTEXTO:
{aggregated_context}

REGRAS:
- Se o contexto contem a resposta, use is_enough true.
- Se falta informacao importante, use is_enough false.
- Se a pergunta pede comparacao com documentos e a fonte RAG nao traz evidencia relevante, use is_enough false.
- Se uma fonte trouxer textos irrelevantes ao tema da pergunta, nao trate isso como evidencia suficiente.
- Nao invente informacoes.
- Se is_enough for true, missing_info deve ser null.
- Se is_enough for false, missing_info deve dizer o que falta.

RESPONDA APENAS JSON VALIDO NESTE FORMATO:
{{
  "is_enough": true,
  "reason": "motivo curto",
  "missing_info": null
}}
"""
    call = call_model(model, prompt)
    judge = JudgeDecision(**extract_json(call.text))
    add_log(
        state,
        "judge_node",
        call.elapsed,
        judge.model_dump(),
        {"question": question, "aggregated_context": aggregated_context},
        call.usage,
        call.estimated_cost_brl,
    )
    return judge


def final_node(
    question: str,
    aggregated_context: str,
    state: dict[str, Any],
    model: str,
) -> str:
    prompt = f"""
PAPEL:
Voce e o Decisor Final.

OBJETIVO:
Responder a pergunta do usuario usando apenas o contexto fornecido.

CONTEXTO:
{aggregated_context}

REGRAS:
- Responda em portugues do Brasil.
- Use apenas informacoes presentes no contexto.
- Nao invente informacoes.
- Se o contexto nao tiver informacao suficiente, diga isso claramente.
- Seja claro e direto.

ENTRADA:
Pergunta do Usuario:
{question}

FORMATO DE SAIDA:
Responda em texto normal, sem JSON ou markdown.
"""
    call = call_model(model, prompt)
    add_log(
        state,
        "final_node",
        call.elapsed,
        call.text,
        {"question": question, "aggregated_context": aggregated_context},
        call.usage,
        call.estimated_cost_brl,
    )
    return call.text


def retry_question(original_question: str, missing_info: str | None) -> str:
    if not missing_info:
        return original_question
    return (
        f"{original_question}\n\n"
        f"Informacao faltante identificada pelo judge: {missing_info}\n"
        "Tente selecionar a memoria mais adequada para recuperar essa informacao."
    )


def run_graph(
    question: str,
    model: str = DEFAULT_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict[str, Any]:
    start = time.time()
    state = empty_graph_state(question, model)
    current_question = question

    for attempt_index in range(max_retries + 1):
        plan = plan_node(current_question, state, model)
        state["plan"] = plan.model_dump()

        tool_results = run_parallel_tools(current_question, plan.selected_tools, state, model)
        state["tool_results"] = tool_results

        aggregated_context = aggregator_node(tool_results, state)
        state["aggregated_context"] = aggregated_context

        judge = judge_node(current_question, aggregated_context, state, model)
        state["judge"] = judge.model_dump()

        state["attempts"].append(
            {
                "attempt": attempt_index + 1,
                "question": current_question,
                "plan": plan.model_dump(),
                "judge": judge.model_dump(),
            }
        )

        if judge.is_enough:
            answer = final_node(question, aggregated_context, state, model)
            state["final_answer"] = answer
            break

        if attempt_index >= max_retries:
            answer = f"Contexto insuficiente: {judge.missing_info}"
            state["final_answer"] = answer
            break

        current_question = retry_question(question, judge.missing_info)

    state["total_elapsed"] = round(time.time() - start, 2)
    state["total_estimated_cost_brl"] = round(state["total_estimated_cost_brl"], 8)
    return state
