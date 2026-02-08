"""
Chat Engine - Conversational Q&A with memory over the indexed codebase.
Supports multi-turn conversations where follow-up questions are
understood in the context of previous exchanges.
"""

from typing import List, Optional

from llama_index.core import Settings, VectorStoreIndex


CONDENSE_PROMPT = """\
Given the following conversation history and a new follow-up question, \
rephrase the follow-up question into a standalone question that captures \
all relevant context from the conversation.

If it's already a standalone question (first question or unrelated to prior ones), \
return it as-is.

Conversation history:
{chat_history}

Follow-up question: {question}

Standalone question:"""

ANSWER_PROMPT = """\
You are a knowledgeable assistant answering questions about a software project's \
codebase. You have access to the project's source code and generated documentation \
via search.

Answer the user's question based on the actual code found in the project. \
Be accurate, concise, and reference specific files and modules when relevant. \
If you don't know or can't find the answer in the code, say so honestly.

Question: {question}
"""


class ChatEngine:
    """
    Conversational Q&A engine with memory over the indexed codebase.

    Maintains conversation history so follow-up questions like
    "what about its configuration?" are resolved correctly using
    the context of previous exchanges.
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        documentation_markdown: Optional[str] = None,
    ):
        self.index = index
        self.documentation_markdown = documentation_markdown
        self.history: List[dict] = []  # [{"role": "user"|"assistant", "content": str}]
        self.query_engine = index.as_query_engine(
            response_mode="tree_summarize",
            similarity_top_k=10,
        )

    def ask(self, question: str) -> str:
        """
        Ask a question with full conversation context.

        If there is prior history, the question is first condensed into
        a standalone query so the retrieval step can find relevant code.
        """
        # If there is conversation history, condense the follow-up
        if self.history:
            standalone = self._condense_question(question)
        else:
            standalone = question

        # Query the index
        prompt = ANSWER_PROMPT.format(question=standalone)
        response = self.query_engine.query(prompt)
        answer = str(response).strip()

        # Store in history
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        return answer

    def _condense_question(self, question: str) -> str:
        """Use the LLM to rewrite a follow-up question as a standalone one."""
        history_text = self._format_history()
        prompt = CONDENSE_PROMPT.format(
            chat_history=history_text,
            question=question,
        )
        response = Settings.llm.complete(prompt)
        condensed = str(response).strip()
        return condensed if condensed else question

    def _format_history(self, max_turns: int = 5) -> str:
        """Format the most recent conversation turns for the condense prompt."""
        recent = self.history[-(max_turns * 2) :]
        lines = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"]
            # Truncate long answers to keep prompt manageable
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"{role}: {content}")
        return "\n".join(lines) if lines else "(no history)"

    def reset(self):
        """Clear all conversation history."""
        self.history.clear()

    def get_history(self) -> List[dict]:
        """Return a copy of the conversation history."""
        return list(self.history)

