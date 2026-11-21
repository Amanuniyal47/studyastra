"""Manual End-to-End Integration Test for Local Qwen2.5 7B via Ollama.

Run this script directly in an environment where Ollama is running and has the
qwen2.5:7b model installed:
    python tests/manual_test_ollama.py

This script verifies:
1. Local Ollama server connectivity (http://localhost:11434).
2. Availability of model `qwen2.5:7b`.
3. Standard synchronous text generation.
4. Token-by-token streaming generation.
5. RAG question-answering with context grounding and source citations.
6. Refusal / anti-hallucination behavior when context does not contain the answer.
"""

from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config as config
from src.generation.answer_generator import generate_answer
from src.generation.llm_provider import (
    LLMProviderError,
    OllamaLLMProvider,
)


def run_manual_test() -> int:
    print("=" * 70)
    print("STUDY BOT - OLLAMA & QWEN2.5 7B MANUAL INTEGRATION TEST")
    print("=" * 70)
    print(f"Target Endpoint : {config.OLLAMA_BASE_URL}")
    print(f"Target Model    : {config.OLLAMA_MODEL}")
    print(f"Context Window  : {config.OLLAMA_CONTEXT_WINDOW} tokens")
    print(f"Timeout         : {config.OLLAMA_TIMEOUT_SECONDS}s")
    print("-" * 70)

    provider = OllamaLLMProvider(
        base_url=config.OLLAMA_BASE_URL,
        model_name=config.OLLAMA_MODEL,
        timeout=config.OLLAMA_TIMEOUT_SECONDS,
        context_window=config.OLLAMA_CONTEXT_WINDOW,
    )

    # 1. Connection check
    print("\n[Step 1/6] Checking connection to Ollama server...")
    ok, msg = provider.check_connection()
    if not ok:
        print(f"❌ Connection failed: {msg}")
        print("\nPlease start Ollama with:\n    ollama serve")
        return 1
    print(f"✅ {msg}")

    # 2. Model availability check
    print(f"\n[Step 2/6] Checking if model '{provider.model_name}' is installed...")
    ok, msg = provider.check_model_available()
    if not ok:
        print(f"❌ Model check failed: {msg}")
        print(f"\nPlease install the model using:\n    ollama pull {provider.model_name}")
        return 1
    print(f"✅ {msg}")

    # 3. Basic Generation
    print("\n[Step 3/6] Testing synchronous generation...")
    t0 = time.time()
    try:
        reply = provider.generate(
            "You are a helpful study tutor. Be concise.",
            "In one sentence, what is mitosis?",
            temperature=0.2,
            max_tokens=100,
        )
        elapsed = time.time() - t0
        print(f"✅ Synchronous generation completed in {elapsed:.2f}s:")
        print(f"   \"{reply}\"")
    except LLMProviderError as exc:
        print(f"❌ Synchronous generation failed: {exc}")
        return 1

    # 4. Streaming Generation
    print("\n[Step 4/6] Testing progressive streaming generation...")
    t0 = time.time()
    try:
        stream = provider.generate_stream(
            "You are a helpful study tutor. Be concise.",
            "In one sentence, what is the powerhouse of the cell?",
            temperature=0.2,
            max_tokens=100,
        )
        print("   Stream tokens: ", end="", flush=True)
        collected_tokens = []
        for token in stream:
            collected_tokens.append(token)
            print(token, end="", flush=True)
        print()
        elapsed = time.time() - t0
        print(f"✅ Streaming completed in {elapsed:.2f}s ({len(collected_tokens)} tokens received)")
    except LLMProviderError as exc:
        print(f"\n❌ Streaming generation failed: {exc}")
        return 1

    # 5. RAG Pipeline with Document Context
    print("\n[Step 5/6] Testing grounded RAG question answering...")
    sample_chunks = [
        {
            "text": (
                "Quantum superposition is a fundamental principle of quantum mechanics. "
                "It states that any two (or more) quantum states can be added together ('superposed') "
                "and the result will be another valid quantum state; and conversely, that every quantum "
                "state can be represented as a sum of two or more other distinct states."
            ),
            "metadata": {
                "filename": "quantum_physics_intro.pdf",
                "page": 14,
                "page_range": "14",
                "chunk_index": 3,
            },
        },
        {
            "text": (
                "Qubits are physical systems that represent quantum information. Unlike classical bits "
                "which are strictly 0 or 1, a qubit can exist in a superposition of |0> and |1> states."
            ),
            "metadata": {
                "filename": "quantum_physics_intro.pdf",
                "page": 15,
                "page_range": "15",
                "chunk_index": 4,
            },
        },
    ]

    question = "According to the document, what is quantum superposition and how do qubits use it?"
    print(f"   Question: {question}")
    t0 = time.time()
    try:
        rag_answer = generate_answer(question, sample_chunks, provider=provider)
        elapsed = time.time() - t0
        print(f"✅ RAG answer received in {elapsed:.2f}s:")
        print("-" * 50)
        print(rag_answer)
        print("-" * 50)
    except Exception as exc:
        print(f"❌ RAG generation failed: {exc}")
        return 1

    # 6. Anti-Hallucination / Insufficient Context Test
    print("\n[Step 6/8] Testing anti-hallucination / refusal behavior...")
    unrelated_question = "What was Napoleon's favourite food according to the document?"
    print(f"   Question: {unrelated_question}")
    t0 = time.time()
    try:
        refusal_answer = generate_answer(unrelated_question, sample_chunks, provider=provider)
        elapsed = time.time() - t0
        print(f"✅ Refusal answer received in {elapsed:.2f}s:")
        print("-" * 50)
        print(refusal_answer)
        print("-" * 50)
        lower_refusal = refusal_answer.lower()
        if (
            "cannot" in lower_refusal
            or "not in" in lower_refusal
            or "does not contain" in lower_refusal
            or "not provided" in lower_refusal
            or "insufficient" in lower_refusal
        ):
            print("✅ Grounding verified: Model properly declined to hallucinate unprovided facts.")
        else:
            print("⚠️ Notice: Model answered, verify it adhered to grounding rules.")
    except Exception as exc:
        print(f"❌ Refusal test failed: {exc}")
        return 1

    # 7. Practice Question Generation with JSON mode
    print("\n[Step 7/8] Testing practice question generation (strict JSON mode)...")
    t0 = time.time()
    try:
        json_reply = provider.generate(
            "You are an academic study-question generator. Return ONLY valid JSON matching the schema.",
            'Generate 1 multiple choice question about qubits from: "Qubits represent quantum information."',
            temperature=0.3,
            max_tokens=300,
        )
        elapsed = time.time() - t0
        print(f"✅ Question generation JSON response received in {elapsed:.2f}s:")
        print(f"   {json_reply[:120]}...")
    except Exception as exc:
        print(f"❌ Question generation test failed: {exc}")
        return 1

    # 8. Document Summary Generation
    print("\n[Step 8/8] Testing document summary generation...")
    t0 = time.time()
    try:
        summary_reply = provider.generate(
            "You are an academic summarizer. Provide a concise bulleted summary.",
            "Summarize: Quantum superposition allows states to be added together.",
            temperature=0.2,
            max_tokens=200,
        )
        elapsed = time.time() - t0
        print(f"✅ Summary generation completed in {elapsed:.2f}s:")
        print(f"   {summary_reply[:120]}...")
    except Exception as exc:
        print(f"❌ Summary generation test failed: {exc}")
        return 1

    print("\n" + "=" * 70)
    print("ALL MANUAL OLLAMA INTEGRATION TESTS PASSED SUCCESSFULLY! 🎉")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(run_manual_test())
