"""Thin OpenAI-compatible chat client with an on-disk cache and thread-pool batching.

Backend is configured by env vars, so the same code talks to a local vLLM server or an API:
    STREAMDUO_LLM_BASE_URL   default http://localhost:8000/v1
    STREAMDUO_LLM_MODEL      default Qwen/Qwen3-30B-A3B-Instruct-2507
    STREAMDUO_LLM_API_KEY    default "EMPTY"
"""
import concurrent.futures as cf
import hashlib
import json
import os
import sqlite3
import threading

from openai import OpenAI
from tqdm import tqdm

from streamduo.paths import CACHE_ROOT


class LLM:
    def __init__(self, model=None, base_url=None, api_key=None, cache_name="llm_cache.sqlite"):
        self.model = model or os.environ.get("STREAMDUO_LLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
        self.client = OpenAI(
            base_url=base_url or os.environ.get("STREAMDUO_LLM_BASE_URL", "http://localhost:8000/v1"),
            api_key=api_key or os.environ.get("STREAMDUO_LLM_API_KEY", "EMPTY"),
            timeout=600,
            max_retries=5,
        )
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(CACHE_ROOT / cache_name, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        self._lock = threading.Lock()

    def _key(self, messages, kw):
        blob = json.dumps([self.model, messages, kw], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def chat(self, messages, temperature=0.0, max_tokens=1024, json_mode=False):
        kw = {"temperature": temperature, "max_tokens": max_tokens, "json": json_mode}
        key = self._key(messages, kw)
        with self._lock:
            row = self._db.execute("SELECT v FROM c WHERE k=?", (key,)).fetchone()
        if row is not None:
            return row[0]
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature,
            max_tokens=max_tokens, **extra,
        )
        out = resp.choices[0].message.content
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO c VALUES (?, ?)", (key, out))
            self._db.commit()
        return out

    def chat_json(self, messages, **kw):
        text = self.chat(messages, json_mode=True, **kw)
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        return json.loads(text)

    def map(self, fn, items, workers=32, desc=None):
        """Apply fn(self, item) concurrently, preserving order."""
        with cf.ThreadPoolExecutor(workers) as ex:
            return list(tqdm(ex.map(lambda it: fn(self, it), items), total=len(items), desc=desc))
