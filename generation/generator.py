from __future__ import annotations

import httpx

from config import SETTINGS


SYSTEM = """You are the answer-generation layer of a Neural Memory LLM.

The memory system has already performed:
1. Query analysis
2. Associative memory retrieval
3. Spreading activation
4. Source-memory selection

Your job is to turn the retrieved evidence into a natural,
accurate answer for the user.

Rules:
- Answer using ONLY the supplied activated memories and source passages.
- Do not invent facts.
- Do not rely on outside knowledge.
- Do not mention graph paths, activation scores, chunks,
  retrieval algorithms, prompts, or internal memory machinery
  unless the user explicitly asks about the system itself.
- Write like a normal helpful LLM.
- For multi-step questions, explain the connection clearly.
- When several documents support an answer, synthesize them.
- When the evidence is insufficient, clearly say that the
  available documents do not contain enough information.
- Prefer concise answers unless more explanation is necessary.
"""


class AnswerGenerator:
    """
    Generates final natural-language answers from recalled
    Neural Memory evidence.

    Supported providers:
        - mock
        - ollama
        - openai
        - openrouter
    """

    def __init__(self) -> None:
        self.provider = SETTINGS.llm_provider

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(
        self,
        query: str,
        retrieval: dict,
    ) -> str:

        if self.provider == "mock":
            return self._mock_answer(
                query,
                retrieval,
            )

        if self.provider == "ollama":
            return self._answer_ollama(
                query,
                retrieval,
            )

        if self.provider == "openai":
            return self._answer_openai(
                query,
                retrieval,
            )

        if self.provider == "openrouter":
            return self._answer_openrouter(
                query,
                retrieval,
            )

        raise ValueError(
            "Unsupported LLM_PROVIDER: "
            f"{self.provider}. "
            "Use mock, ollama, openai, or openrouter."
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        query: str,
        retrieval: dict,
    ) -> str:

        memories = []

        for node in retrieval.get(
            "activations",
            [],
        )[:12]:

            name = node.get(
                "name",
                "Unknown",
            )

            node_type = node.get(
                "type",
                "Concept",
            )

            activation = node.get(
                "activation",
                0,
            )

            memories.append(
                f"- {name} "
                f"({node_type}) "
                f"| activation={activation:.4f}"
            )

        memory_text = (
            "\n".join(memories)
            if memories
            else "- No activated memories"
        )

        chunks = []

        for chunk in retrieval.get(
            "chunks",
            [],
        )[:8]:

            source = chunk.get(
                "source",
                "unknown-source",
            )

            chunk_id = chunk.get(
                "chunk_id",
                "unknown-chunk",
            )

            text = chunk.get(
                "text",
                "",
            )

            chunks.append(
                f"[{source} | {chunk_id}]\n{text}"
            )

        source_text = (
            "\n\n".join(chunks)
            if chunks
            else "No supporting source passages were retrieved."
        )

        return (
            "User question:\n"
            f"{query}\n\n"
            "Activated memories:\n"
            f"{memory_text}\n\n"
            "Supporting source passages:\n"
            f"{source_text}\n\n"
            "Compose the final answer for the user."
        )

    # ------------------------------------------------------------------
    # OpenRouter
    # ------------------------------------------------------------------

    def _answer_openrouter(
        self,
        query: str,
        retrieval: dict,
    ) -> str:

        if not SETTINGS.openrouter_api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is required when "
                "LLM_PROVIDER=openrouter."
            )

        base_url = (
            SETTINGS.openrouter_base_url
            .rstrip("/")
        )

        url = (
            f"{base_url}/chat/completions"
        )

        payload = {
            "model": SETTINGS.openrouter_model,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM,
                },
                {
                    "role": "user",
                    "content": self._build_prompt(
                        query,
                        retrieval,
                    ),
                },
            ],
        }

        headers = {
            "Authorization": (
                f"Bearer "
                f"{SETTINGS.openrouter_api_key}"
            ),
            "Content-Type": "application/json",
        }

        if SETTINGS.openrouter_http_referer:
            headers["HTTP-Referer"] = (
                SETTINGS.openrouter_http_referer
            )

        if SETTINGS.openrouter_app_name:
            headers["X-Title"] = (
                SETTINGS.openrouter_app_name
            )

        try:
            with httpx.Client(
                timeout=180,
            ) as client:

                response = client.post(
                    url,
                    json=payload,
                    headers=headers,
                )

                response.raise_for_status()

        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                "OpenRouter answer request failed "
                f"with HTTP "
                f"{exc.response.status_code}.\n"
                f"{exc.response.text[:3000]}"
            ) from exc

        except httpx.RequestError as exc:
            raise RuntimeError(
                "Could not connect to OpenRouter "
                f"while generating the answer: {exc}"
            ) from exc

        try:
            body = response.json()
        except ValueError as exc:
            raise RuntimeError(
                "OpenRouter returned a non-JSON response "
                "while generating the answer."
            ) from exc

        answer = (
            body.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        if not answer:
            raise RuntimeError(
                "OpenRouter returned an empty answer."
            )

        return answer

    # ------------------------------------------------------------------
    # Ollama
    # ------------------------------------------------------------------

    def _answer_ollama(
        self,
        query: str,
        retrieval: dict,
    ) -> str:

        self._check_ollama()

        url = (
            SETTINGS.ollama_base_url.rstrip("/")
            + "/api/chat"
        )

        payload = {
            "model": SETTINGS.ollama_model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM,
                },
                {
                    "role": "user",
                    "content": self._build_prompt(
                        query,
                        retrieval,
                    ),
                },
            ],
            "options": {
                "temperature": 0.1,
            },
        }

        try:
            with httpx.Client(
                timeout=180,
            ) as client:

                response = client.post(
                    url,
                    json=payload,
                )

                response.raise_for_status()

        except httpx.ConnectError as exc:
            raise RuntimeError(
                "Ollama is not running. Start Ollama, "
                "pull the configured model, then retry. "
                f"Expected endpoint: "
                f"{SETTINGS.ollama_base_url}"
            ) from exc

        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                "Ollama answer request failed "
                f"with HTTP "
                f"{exc.response.status_code}: "
                f"{exc.response.text[:2000]}"
            ) from exc

        except httpx.RequestError as exc:
            raise RuntimeError(
                f"Could not connect to Ollama: {exc}"
            ) from exc

        body = response.json()

        answer = (
            body.get("message", {})
            .get("content", "")
            .strip()
        )

        if not answer:
            raise RuntimeError(
                "Ollama returned an empty answer."
            )

        return answer

    def _check_ollama(self) -> None:
        try:
            with httpx.Client(
                timeout=5,
            ) as client:

                response = client.get(
                    SETTINGS.ollama_base_url.rstrip("/")
                    + "/api/tags"
                )

                response.raise_for_status()

        except httpx.ConnectError as exc:
            raise RuntimeError(
                "Ollama is not running. "
                "Start Ollama, then retry."
            ) from exc

        except httpx.RequestError as exc:
            raise RuntimeError(
                f"Could not connect to Ollama: {exc}"
            ) from exc

        models = response.json().get(
            "models",
            [],
        )

        model_names = {
            model.get("name")
            for model in models
            if model.get("name")
        }

        if SETTINGS.ollama_model not in model_names:
            raise RuntimeError(
                "Ollama is running, but model "
                f"'{SETTINGS.ollama_model}' is not installed. "
                f"Run: ollama pull "
                f"{SETTINGS.ollama_model}"
            )

    # ------------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------------

    def _answer_openai(
        self,
        query: str,
        retrieval: dict,
    ) -> str:

        if not SETTINGS.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required when "
                "LLM_PROVIDER=openai."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install the 'openai' package to use "
                "LLM_PROVIDER=openai."
            ) from exc

        client = OpenAI(
            api_key=SETTINGS.openai_api_key,
            base_url=(
                SETTINGS.openai_base_url
                or None
            ),
        )

        response = (
            client.chat.completions.create(
                model=SETTINGS.openai_model,
                temperature=0.1,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM,
                    },
                    {
                        "role": "user",
                        "content": self._build_prompt(
                            query,
                            retrieval,
                        ),
                    },
                ],
            )
        )

        answer = (
            response.choices[0]
            .message
            .content
            or ""
        ).strip()

        return (
            answer
            or "No answer was generated."
        )

    # ------------------------------------------------------------------
    # Offline mock mode
    # ------------------------------------------------------------------

    def _mock_answer(
        self,
        query: str,
        retrieval: dict,
    ) -> str:
        """
        Deterministic test-only answerer.

        Production usage should use OpenRouter, Ollama,
        or another actual LLM provider.
        """

        chunks = retrieval.get(
            "chunks",
            [],
        )

        if not chunks:
            return (
                "I could not find enough information "
                "in the ingested documents to answer that."
            )

        text = " ".join(
            c.get("text", "")
            for c in chunks[:3]
        ).strip()

        question = query.casefold()

        import re

        patterns = [
            (
                r"(.+?)\s+founded\s+(.+?)(?:\.|$)",
                lambda m: (
                    f"{m.group(1).strip()} "
                    f"founded "
                    f"{m.group(2).strip()}."
                ),
            ),
            (
                r"(.+?)\s+acquired\s+(.+?)(?:\.|$)",
                lambda m: (
                    f"{m.group(1).strip()} "
                    f"acquired "
                    f"{m.group(2).strip()}."
                ),
            ),
            (
                r"(.+?)\s+developed\s+(.+?)(?:\.|$)",
                lambda m: (
                    f"{m.group(1).strip()} "
                    f"developed "
                    f"{m.group(2).strip()}."
                ),
            ),
            (
                r"(.+?)\s+joined\s+(.+?)(?:\.|$)",
                lambda m: (
                    f"{m.group(1).strip()} "
                    f"joined "
                    f"{m.group(2).strip()}."
                ),
            ),
        ]

        for pattern, formatter in patterns:
            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match and any(
                keyword in question
                for keyword in (
                    "who",
                    "what",
                    "which",
                    "how",
                )
            ):
                return formatter(match)

        first = chunks[0].get(
            "text",
            "",
        ).strip()

        first_sentence = re.split(
            r"(?<=[.!?])\s+",
            first,
        )[0]

        return (
            first_sentence
            or "No answer could be generated."
        )