from langchain_core.prompts import ChatPromptTemplate


def get_rag_prompt():
    """
    Create and return the RAG prompt template.

    Returns:
        ChatPromptTemplate
    """

    prompt = ChatPromptTemplate.from_template(
        """
You are an AI assistant that answers questions only using the provided context.

Instructions:
- Use only the information present in the context.
- Do not make up information.
- If the answer is not available in the context, reply:
  "I couldn't find the answer in the provided documents."
- Keep the answer clear and concise.

Context:
{context}

Question:
{input}

Answer:
"""
    )

    return prompt