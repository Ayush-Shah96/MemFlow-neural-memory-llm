from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone

import httpx

from config import SETTINGS
from .mock import extract_heuristically
from .prompt import EXTRACTION_SYSTEM
from .schema import normalize_extraction


logger = logging.getLogger("neural-memory.extraction")


class ExtractionService:
    def __init__(self):
        self.provider = SETTINGS.llm_provider
        logger.info(
            "Extraction service initialized | provider=%s | model=%s",
            self.provider,
            self._model_name(),
        )

    def _model_name(self) -> str:
        if self.provider == "openrouter":
            return SETTINGS.openrouter_model
        if self.provider == "ollama":
            return SETTINGS.ollama_model
        if self.provider == "openai":
            return SETTINGS.openai_model
        return "mock"

    def extract(self, text: str) -> dict:
        started = time.perf_counter()
        logger.info(
            "Extraction started | provider=%s | chars=%d",
            self.provider,
            len(text or ""),
        )

        try:
            if self.provider == "mock":
                data = extract_heuristically(text)
            elif self.provider == "openrouter":
                data = self._extract_openrouter(text)
            elif self.provider == "openai":
                data = self._extract_openai(text)
            elif self.provider == "ollama":
                data = self._extract_ollama(text)
            else:
                raise ValueError(
                    f"Unsupported LLM_PROVIDER: {self.provider}"
                )

            data = normalize_extraction(data)
            data["timestamp"] = datetime.now(timezone.utc).isoformat()
            data["confidence"] = self._confidence(data)

            logger.info(
                "Extraction finished | provider=%s | entities=%d | relationships=%d | %.2fs",
                self.provider,
                len(data.get("entities", [])),
                len(data.get("relationships", [])),
                time.perf_counter() - started,
            )
            return data

        except Exception:
            logger.exception(
                "Extraction failed | provider=%s | elapsed=%.2fs",
                self.provider,
                time.perf_counter() - started,
            )
            raise

    def _confidence(self, data: dict) -> float:
        rels = data.get("relationships", [])
        ents = data.get("entities", [])
        if not ents:
            return 0.1
        return min(
            0.99,
            0.55
            + 0.08 * len(rels)
            + 0.02 * min(len(ents), 10),
        )

    def _parse_json(self, raw: str) -> dict:
        raw = raw.strip()
        raw = re.sub(
            r"^```(?:json)?\s*",
            "",
            raw,
            flags=re.I,
        )
        raw = re.sub(
            r"\s*```$",
            "",
            raw,
        )
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("LLM returned no JSON object")
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid LLM JSON: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError(
                "LLM extraction response must be a JSON object"
            )
        return parsed

    def _openrouter_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": (
                f"Bearer {SETTINGS.openrouter_api_key}"
            ),
            "Content-Type": "application/json",
        }
        if SETTINGS.openrouter_http_referer:
            headers["HTTP-Referer"] = SETTINGS.openrouter_http_referer
        if SETTINGS.openrouter_app_name:
            headers["X-Title"] = SETTINGS.openrouter_app_name
        return headers

    def _extract_openrouter(self, text: str) -> dict:
        if not SETTINGS.openrouter_api_key:
            logger.error("OpenRouter API key is missing")
            raise RuntimeError(
                "OPENROUTER_API_KEY is required when "
                "LLM_PROVIDER=openrouter"
            )

        url = (
            SETTINGS.openrouter_base_url.rstrip("/")
            + "/chat/completions"
        )
        payload = {
            "model": SETTINGS.openrouter_model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": EXTRACTION_SYSTEM,
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
        }

        logger.info(
            "OpenRouter extraction request START | model=%s | chars=%d | timeout=120s",
            SETTINGS.openrouter_model,
            len(text),
        )
        request_started = time.perf_counter()

        try:
            with httpx.Client(timeout=120) as client:
                response = client.post(
                    url,
                    json=payload,
                    headers=self._openrouter_headers(),
                )
        except httpx.TimeoutException as exc:
            logger.error(
                "OpenRouter extraction TIMEOUT after %.2fs",
                time.perf_counter() - request_started,
            )
            raise RuntimeError(
                "OpenRouter extraction request timed out after 120 seconds."
            ) from exc
        except httpx.RequestError as exc:
            logger.error(
                "OpenRouter extraction CONNECTION ERROR after %.2fs | %s",
                time.perf_counter() - request_started,
                exc,
            )
            raise RuntimeError(
                f"Could not connect to OpenRouter: {exc}"
            ) from exc

        elapsed = time.perf_counter() - request_started
        logger.info(
            "OpenRouter extraction HTTP response | status=%d | %.2fs",
            response.status_code,
            elapsed,
        )

        if response.status_code != 200:
            logger.error(
                "OpenRouter extraction FAILED | status=%d | body=%s",
                response.status_code,
                response.text[:2000],
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(
                    "OpenRouter extraction request failed with "
                    f"HTTP {response.status_code}: "
                    f"{response.text[:2000]}"
                ) from exc

        try:
            body = response.json()
        except ValueError as exc:
            logger.error(
                "OpenRouter returned invalid JSON | body=%s",
                response.text[:2000],
            )
            raise RuntimeError(
                "OpenRouter returned a non-JSON response."
            ) from exc

        content = (
            body.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        if not content:
            logger.error(
                "OpenRouter returned an empty extraction content | body=%s",
                body,
            )
            raise RuntimeError(
                "OpenRouter returned an empty extraction response."
            )

        logger.info(
            "OpenRouter extraction COMPLETE | response_chars=%d | %.2fs",
            len(content),
            elapsed,
        )
        return self._parse_json(content)

    def _extract_openai(self, text: str) -> dict:
        if not SETTINGS.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required when LLM_PROVIDER=openai"
            )
        base = (
            SETTINGS.openai_base_url
            or "https://api.openai.com/v1"
        ).rstrip("/")
        url = (
            base
            if base.endswith("/chat/completions")
            else base + "/chat/completions"
        )
        payload = {
            "model": SETTINGS.openai_model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": EXTRACTION_SYSTEM,
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
        }
        headers = {
            "Authorization": f"Bearer {SETTINGS.openai_api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=120) as client:
            response = client.post(
                url,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            body = response.json()
        return self._parse_json(
            body.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

    def _extract_ollama(self, text: str) -> dict:
        url = (
            SETTINGS.ollama_base_url.rstrip("/")
            + "/api/chat"
        )
        payload = {
            "model": SETTINGS.ollama_model,
            "stream": False,
            "format": "json",
            "messages": [
                {
                    "role": "system",
                    "content": EXTRACTION_SYSTEM,
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
        }
        with httpx.Client(timeout=120) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            body = response.json()
        return self._parse_json(
            body.get("message", {}).get("content", "")
        )