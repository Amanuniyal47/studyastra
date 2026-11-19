SYSTEM_PROMPT = (
    "You answer questions using only the provided PDF context. "
    "If the answer is not in the context, say you do not know. "
    "Cite sources using the source numbers provided."
)


def build_context(chunks):
    return "\n\n".join(
        f"Source {index} (page {chunk['metadata']['page']}):\n{chunk['text']}"
        for index, chunk in enumerate(chunks, start=1)
    )


def build_user_prompt(question, chunks):
    context = build_context(chunks)
    return f"Context:\n{context}\n\nQuestion: {question}"
