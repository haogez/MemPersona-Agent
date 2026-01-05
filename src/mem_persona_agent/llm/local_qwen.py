from __future__ import annotations

import threading
from itertools import cycle
import os
import re
import logging
from typing import Dict, Generator, Iterable, List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from mem_persona_agent.config import settings


DEFAULT_MODEL_PATH = "/models/Qwen3-14B"
logger = logging.getLogger(__name__)
_visible_devices_applied = False


def _apply_visible_devices() -> None:
    global _visible_devices_applied
    if _visible_devices_applied:
        return
    devices = (settings.qwen_visible_devices or "").strip()
    if devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = devices
        logger.info("Qwen set CUDA_VISIBLE_DEVICES=%s", devices)
    _visible_devices_applied = True


class _QwenInstance:
    """Single in-process Qwen3 model instance guarded by a lock."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        _apply_visible_devices()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True,
        )
        if torch.cuda.is_available():
            gpu_info = [
                f"{idx}:{torch.cuda.get_device_name(idx)}"
                for idx in range(torch.cuda.device_count())
            ]
            logger.info("Loading Qwen instance on GPUs: %s", "; ".join(gpu_info))
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        try:
            device_map = getattr(self.model, "hf_device_map", None)
            if device_map:
                logger.info("Qwen instance device_map: %s", device_map)
        except Exception:
            logger.debug("Failed to log device_map", exc_info=True)
        self.device = getattr(self.model, "device", None) or next(self.model.parameters()).device
        self._lock = threading.Lock()

    def _build_inputs(self, messages: Sequence[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_lines = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages]
            prompt_text = "\n".join(prompt_lines) + "\nassistant:"
        encoded = self.tokenizer(prompt_text, return_tensors="pt")
        return {k: v.to(self.device) for k, v in encoded.items()}

    def _build_generate_kwargs(
        self,
        inputs: Dict[str, torch.Tensor],
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> Dict[str, torch.Tensor | float | int | bool]:
        do_sample = temperature > 1e-8
        safe_temperature = max(float(temperature), 1e-5)
        return {
            **inputs,
            "do_sample": do_sample,
            "temperature": safe_temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

    def generate(
        self,
        messages: Sequence[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 1024,
    ) -> str:
        with self._lock:
            inputs = self._build_inputs(messages)
            prompt_len = inputs["input_ids"].shape[-1]
            gen_kwargs = self._build_generate_kwargs(inputs, temperature, top_p, max_new_tokens)
            with torch.no_grad():
                output_ids = self.model.generate(**gen_kwargs)
            new_tokens = output_ids[0][prompt_len:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    def stream_generate(
        self,
        messages: Sequence[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 1024,
    ) -> Generator[str, None, None]:
        def _iterator() -> Generator[str, None, None]:
            self._lock.acquire()
            try:
                inputs = self._build_inputs(messages)
                streamer = TextIteratorStreamer(
                    self.tokenizer,
                    skip_special_tokens=True,
                    decode_kwargs={"skip_special_tokens": True},
                )
                gen_kwargs = self._build_generate_kwargs(inputs, temperature, top_p, max_new_tokens)
                gen_kwargs["streamer"] = streamer

                def _strip_think_stream() -> Generator[str, None, None]:
                    in_think = False
                    buffer = ""
                    pending_emit = ""
                    for token in streamer:
                        buffer += token
                        while buffer:
                            if in_think:
                                end = buffer.find("</think>")
                                if end == -1:
                                    buffer = buffer[-8:]  # keep small tail to detect closing
                                    break
                                buffer = buffer[end + len("</think>") :]
                                in_think = False
                                continue
                            start = buffer.find("<think>")
                            if start == -1:
                                pending_emit += buffer
                                buffer = ""
                                break
                            if start > 0:
                                pending_emit += buffer[:start]
                            buffer = buffer[start + len("<think>") :]
                            in_think = True
                        if pending_emit:
                            yield_text, pending_emit = pending_emit, ""
                            yield yield_text
                    if not in_think and pending_emit:
                        yield pending_emit

                def _run() -> None:
                    with torch.no_grad():
                        self.model.generate(**gen_kwargs)

                thread = threading.Thread(target=_run, daemon=True)
                thread.start()

                try:
                    for text in _strip_think_stream():
                        yield text
                finally:
                    thread.join()
            finally:
                self._lock.release()

        return _iterator()


class LocalQwenEngine:
    """Maintains two in-process Qwen3 instances with round-robin routing."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.instances: List[_QwenInstance] = [
            _QwenInstance(model_path=model_path),
            _QwenInstance(model_path=model_path),
        ]
        self._rr = cycle(self.instances)
        self._rr_lock = threading.Lock()

    def _pick_instance(self) -> _QwenInstance:
        with self._rr_lock:
            return next(self._rr)

    def generate(
        self,
        messages: Sequence[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 1024,
    ) -> str:
        instance = self._pick_instance()
        return instance.generate(messages, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens)

    def stream_generate(
        self,
        messages: Sequence[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 1024,
    ) -> Generator[str, None, None]:
        instance = self._pick_instance()
        return instance.stream_generate(
            messages,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )

    def get_instances(self) -> Iterable[_QwenInstance]:
        return tuple(self.instances)


_singleton_lock = threading.Lock()
_singleton_engine: LocalQwenEngine | None = None


def get_local_qwen_engine(model_path: str = DEFAULT_MODEL_PATH) -> LocalQwenEngine:
    global _singleton_engine
    if _singleton_engine is None:
        with _singleton_lock:
            if _singleton_engine is None:
                _singleton_engine = LocalQwenEngine(model_path=model_path)
    return _singleton_engine
