import os
import json
import google.generativeai as genai
from pydantic import BaseModel, Field
from typing import Dict, Any
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIR = "route"

genai.configure(api_key="private")


def init_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)


vdb = init_vector_db()


class ScikitQueryIntelligence(BaseModel):
    rewritten_query: str = Field(
        description="Cleaned, technical, optimized version of the user query "
                    "in English. Translate to English if original is Arabic. "
                    "Use precise Scikit-Learn terminology."
    )
    query_category: str = Field(
        description="Classify into exactly one: 'Factual/Syntax Lookup', "
                    "'Conceptual Explanation', 'Model Comparison', "
                    "'Model Recommendation', or 'Out of Scope'"
    )
    extracted_filters: Dict[str, Any] = Field(
        description="Strict metadata filters. Keys limited to: 'module' "
                    "(e.g. 'linear_model', 'svm', 'tree', 'ensemble') or "
                    "'task' ('classification', 'regression', 'clustering'). "
                    "Empty dict if none apply."
    )


def run_query_intelligence(user_query: str) -> ScikitQueryIntelligence:
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""
    You are an expert AI system analyzing developer queries for a Scikit-Learn
    documentation RAG system. Analyze the user query and output a structured
    JSON response matching the required schema strictly.

    User Query: "{user_query}"
    """
    response = model.generate_content(
        prompt,
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": ScikitQueryIntelligence,
        },
    )
    return ScikitQueryIntelligence(**json.loads(response.text))


def run_rag_pipeline_level2(original_query: str):
    print("\n" + "=" * 50)
    print(f"📥 [Original User Query]: {original_query}")
    print("=" * 50)

    intelligence = run_query_intelligence(original_query)

    print("\n✍️ [2.1 — Query Rewriting Output]:")
    print(f"    -> {intelligence.rewritten_query}")

    print("\n🏷️ [2.2 — Query Classification Output]:")
    print(f"    -> Category: {intelligence.query_category}")

    print("\n⚙️ [2.3 — Structured Parameter Extraction Output]:")
    print(f"    -> Extracted Filters: {intelligence.extracted_filters}")

    print("\n🔍 [4 — Filtered Retrieval & Scores Output]:")
    search_kwargs = {"k": 3}
    if intelligence.extracted_filters:
        search_kwargs["filter"] = intelligence.extracted_filters
        print(f"    [System Info] Narrowing search using filter: "
              f"{intelligence.extracted_filters}")

    results = vdb.similarity_search_with_score(
        intelligence.rewritten_query, **search_kwargs
    )

    if not results:
        print("    ⚠️ No chunks retrieved. Check PERSIST_DIR path or filters.")
        context_text = ""
    else:
        context_chunks = []
        for idx, (doc, score) in enumerate(results):
            print(f"    -> Chunk #{idx + 1} (Score/Distance: {score:.4f})")
            print(f"       Metadata: {doc.metadata}")
            print(f"       Sample: {doc.page_content[:150]}...\n")
            context_chunks.append(doc.page_content)
        context_text = "\n\n".join(context_chunks)

    print("🎯 [5 — Gemini-Based Answer Generation]:")
    generation_prompt = f"""
    You are an expert Retrieval-Augmented Generation (RAG) assistant
    specializing in Scikit-Learn. Answer the user question based ONLY on
    the provided context below. If the answer cannot be found or deduced
    from the context, reply exactly with:
    "I cannot find the answer in the provided context."

    Context:
    {context_text}

    Question:
    {intelligence.rewritten_query}

    Answer:
    """
    gen_model = genai.GenerativeModel("gemini-2.5-flash")
    response = gen_model.generate_content(generation_prompt)
    answer = response.text
    print(answer)
    print("=" * 50 + "\n")
    return answer


if __name__ == "__main__":
    run_rag_pipeline_level2("ازاي اغير الـ learning rate في الـ SGDRegressor؟")
    run_rag_pipeline_level2("What is the difference between Ridge and Lasso?")
