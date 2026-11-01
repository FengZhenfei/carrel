"""Synthetic request contracts for free-form model registration and inference."""
from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from kb_pipeline import llm_compat as compat
from kb_pipeline.graph.llm import ChatClient, LLMCache, LLMSpec, chat_once
from kb_pipeline.vision import vlm
from kb_server import service


def response(status=200, content="OK", protocol="openai", error=None, finish="stop"):
    payload = ({"content": [{"type": "text", "text": content}], "stop_reason": "end_turn"}
               if protocol == "anthropic" else
               {"choices": [{"message": {"content": content}, "finish_reason": finish}]})
    if error is not None:
        payload = {"error": error}
    return SimpleNamespace(status_code=status, text=json.dumps(payload), json=lambda: payload, headers={})


class CompatibilityTests(unittest.TestCase):
    def setUp(self):
        compat._ACCEPTED.clear()
        self.addCleanup(compat._ACCEPTED.clear)
        env = patch.dict(os.environ, {"KB_LLM_COMPAT_OVERRIDES": ""})
        env.start()
        self.addCleanup(env.stop)

    def params(self, model, **kwargs):
        return compat.generation_parameters(model, max_tokens=1024, temperature=0, top_p=0.8, **kwargs)

    def test_free_form_models_preserve_explicit_zero_and_budget(self):
        for model in ("custom-deployment", "gpt-4o", "deepseek-chat", "qwen-vl"):
            with self.subTest(model=model):
                self.assertEqual(self.params(model), {"max_tokens": 1024, "temperature": 0.0, "top_p": 0.8})

    def test_reasoning_models_use_completion_budget_without_sampling(self):
        for model in ("gpt-5", "gpt-5-mini", "gpt-5.4", "openai/gpt-5-nano", "o1", "o3-mini", "o4-mini"):
            with self.subTest(model=model):
                self.assertEqual(self.params(model), {"max_completion_tokens": 1024})
        params = self.params("gpt-5.4", disable_thinking=True)
        self.assertEqual(params["reasoning_effort"], "none")
        self.assertEqual(params["temperature"], 0)
        self.assertEqual(self.params("gpt-5", disable_thinking=True)["reasoning_effort"], "minimal")

    def test_claude_restrictions_apply_to_native_and_proxy_protocols(self):
        for protocol in ("openai", "anthropic"):
            for model in ("claude-opus-4-7", "claude-sonnet-5-5", "anthropic/claude-opus-4-7-20261001"):
                with self.subTest(model=model, protocol=protocol):
                    self.assertEqual(self.params(model, protocol=protocol), {"max_tokens": 1024})
        self.assertEqual(self.params("claude-sonnet-4-6", protocol="anthropic"),
                         {"max_tokens": 1024, "temperature": 0.0})

    def test_fixed_sampling_kimi_and_provider_thinking_controls(self):
        self.assertEqual(self.params("kimi-k2.6"), {"max_tokens": 1024})
        self.assertEqual(self.params("deepseek-chat", disable_thinking=True)["thinking"], {"type": "disabled"})
        self.assertNotIn("enable_thinking", self.params("deepseek-chat", disable_thinking=True))
        self.assertEqual(self.params("Qwen3.8-VL", disable_thinking=True)["enable_thinking"], False)

    def test_endpoint_scoped_override_supports_custom_deployment_ids(self):
        override = [{"model_id": "custom-deployment", "base_url": "https://models.example/v1",
                     "max_tokens_field": "max_completion_tokens", "omit": ["temperature", "top_p"],
                     "set": {"reasoning_effort": "low"}}]
        with patch.dict(os.environ, {"KB_LLM_COMPAT_OVERRIDES": json.dumps(override)}):
            actual = self.params("custom-deployment", base_url="https://models.example/v1/")
            self.assertEqual(actual, {"max_completion_tokens": 1024, "reasoning_effort": "low"})
            self.assertIn("max_tokens", self.params("custom-deployment", base_url="https://other.example/v1"))
            self.assertIn("temperature", self.params("different-model", base_url="https://models.example/v1"))
        with patch.dict(os.environ, {"KB_LLM_COMPAT_OVERRIDES": '[{"model_id":"x","set":{"messages":[]}}]'}):
            with self.assertRaises(ValueError):
                self.params("x")

    def test_only_named_optional_parameters_are_removed(self):
        request = compat.CompatibleRequest("https://a.example/v1", "alias", "openai",
                                           {"max_tokens": 1024, "temperature": 0, "enable_thinking": False,
                                            "thinking": {"type": "disabled"}})
        self.assertTrue(request.adapt(400, "Unrecognized request argument supplied: enable_thinking"))
        self.assertNotIn("enable_thinking", request.body)
        self.assertIn("thinking", request.body)
        self.assertEqual(request.body["max_tokens"], 1024)
        self.assertEqual(compat._ACCEPTED, {})

    def test_thinking_override_only_applies_when_the_caller_requests_it(self):
        override = [{"model_id": "qwen-custom", "base_url": "https://models.example/v1",
                     "thinking_parameters": {"chat_template_kwargs": {"enable_thinking": False}}}]
        with patch.dict(os.environ, {"KB_LLM_COMPAT_OVERRIDES": json.dumps(override)}):
            graph_params = self.params("qwen-custom", base_url="https://models.example/v1", disable_thinking=True)
            self.assertEqual(graph_params["chat_template_kwargs"], {"enable_thinking": False})
            self.assertNotIn("enable_thinking", graph_params)
            image_params = self.params("qwen-custom", base_url="https://models.example/v1", repetition_penalty=1.05)
            self.assertEqual(image_params, {"max_tokens": 1024, "temperature": 0.0, "top_p": 0.8, "repetition_penalty": 1.05})
            self.assertNotIn("chat_template_kwargs", compat.sdk_arguments(image_params).get("extra_body", {}))
            self.assertEqual(self.params("qwen-custom", base_url="https://other.example/v1", disable_thinking=True)["enable_thinking"], False)

    def test_invalid_thinking_override_cannot_change_message_or_token_budget(self):
        for key in ("messages", "max_tokens", "temperature"):
            with self.subTest(key=key), patch.dict(os.environ, {"KB_LLM_COMPAT_OVERRIDES": json.dumps([
                    {"model_id": "custom", "thinking_parameters": {key: 1}}])}):
                with self.assertRaises(ValueError):
                    self.params("custom")

    def test_token_field_fallback_keeps_the_budget(self):
        request = compat.CompatibleRequest("https://a.example/v1", "alias", "openai", {"max_tokens": 2048})
        error = json.dumps({"error": {"param": "max_tokens", "code": "unsupported_parameter",
                           "message": "max_tokens is not supported. Use max_completion_tokens instead."}})
        self.assertTrue(request.adapt(400, error))
        self.assertEqual(request.body, {"max_completion_tokens": 2048})
        self.assertFalse(request.adapt(400, "max_completion_tokens must be at most 1024"))
        self.assertEqual(request.body, {"max_completion_tokens": 2048})

    def test_unknown_auth_input_and_service_errors_never_adapt(self):
        for status, error in ((400, "invalid request"), (401, "Unknown parameter: temperature"),
                              (403, "temperature unsupported"), (429, "temperature unsupported"),
                              (500, "temperature unsupported"), (400, "context_length_exceeded temperature unsupported"),
                              (400, "invalid api_key; temperature unsupported"),
                              (400, "data_inspection_failed: unsupported content"),
                              (400, "temperature must be between 0 and 2")):
            request = compat.CompatibleRequest("https://a.example/v1", "alias", "openai", {"temperature": 0})
            with self.subTest(status=status, error=error):
                self.assertFalse(request.adapt(status, error))
                self.assertEqual(request.body, {"temperature": 0})
        self.assertEqual(compat._ACCEPTED, {})

    def test_successful_adaptations_are_scoped_and_bounded(self):
        body = {"temperature": 0, "max_tokens": 1024}
        def make(base="https://a.example/v1", model="alias", protocol="openai", params=None):
            return compat.CompatibleRequest(base, model, protocol, body if params is None else params)
        request = make()
        self.assertTrue(request.adapt(400, "temperature is not supported"))
        self.assertIn("temperature", make().body)  # no success yet
        request.accept()
        self.assertNotIn("temperature", make().body)
        self.assertIn("temperature", make(model="another").body)
        self.assertIn("temperature", make(base="https://b.example/v1").body)
        self.assertIn("temperature", make(protocol="anthropic").body)
        self.assertIn("temperature", make(params={"temperature": 0.5, "max_tokens": 1024}).body)
        request = make(params={"max_tokens": 1024})
        for _ in range(compat.MAX_COMPAT_RETRIES):
            current = next(iter(request.body))
            replacement = "max_completion_tokens" if current == "max_tokens" else "max_tokens"
            self.assertTrue(request.adapt(400, f"Unsupported parameter: {current}. Use {replacement} instead."))
        self.assertFalse(request.adapt(400, "Unsupported max_tokens. Use max_completion_tokens instead."))

    def test_probe_graph_and_image_share_known_model_rules(self):
        model = "gpt-5-nano"
        with patch.object(service.requests, "post", return_value=response()) as post:
            service._probe_llm("https://models.example/v1", "", model)
            probe = post.call_args.kwargs["json"]
        session = SimpleNamespace(post=Mock(return_value=response()))
        chat_once(LLMSpec("demo", "https://models.example/v1", "", model),
                  [{"role": "user", "content": "q"}], session=session)
        graph = session.post.call_args.kwargs["json"]
        completion = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps({"summary": "A sample figure", "facts": []})),
            finish_reason="stop")]))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
        with patch.object(vlm, "_client_for", return_value=client), patch.object(vlm, "image_data_url", return_value="data:,"):
            vlm.caption_image(image_path=Path("unused.png"), base_url="https://models.example/v1", api_key="", model_id=model)
        image = completion.call_args.kwargs
        for body in (probe, graph, image):
            self.assertIn("max_completion_tokens", body)
            self.assertNotIn("max_tokens", body)
            self.assertNotIn("temperature", body)
            self.assertNotIn("top_p", body)
            self.assertNotIn("enable_thinking", body)
        self.assertNotIn("extra_body", image)
        self.assertIn("response_format", image)

    def test_probe_requires_a_completed_real_response(self):
        for reply in (response(400, error={"message": "invalid request"}), response(content=""),
                      response(content="OK", finish="length")):
            with patch.object(service.requests, "post", return_value=reply) as post:
                with self.assertRaises(ValueError):
                    service._probe_llm("https://models.example/v1", "", "custom")
                self.assertEqual(post.call_count, 1)

    def test_native_anthropic_probe_and_graph_share_parameters_and_url(self):
        model = "claude-opus-4-7"
        with patch.object(service.requests, "post", return_value=response(protocol="anthropic")) as post:
            service._probe_llm("https://models.example/v1", "", model, "anthropic")
            self.assertEqual(post.call_args.args[0], "https://models.example/v1/messages")
            self.assertNotIn("temperature", post.call_args.kwargs["json"])
        session = SimpleNamespace(post=Mock(return_value=response(protocol="anthropic")))
        chat_once(LLMSpec("demo", "https://models.example/v1", "", model, "anthropic"),
                  [{"role": "system", "content": "system"}, {"role": "user", "content": "q"}], session=session)
        body = session.post.call_args.kwargs["json"]
        self.assertEqual(body["system"], "system")
        self.assertEqual(len(body["messages"]), 1)
        self.assertNotIn("temperature", body)
        self.assertIn("max_tokens", body)

    def test_graph_caches_the_effective_request_after_adaptation(self):
        spec = LLMSpec("demo", "https://models.example/v1", "", "alias")
        with tempfile.TemporaryDirectory() as tmp:
            cache = LLMCache(Path(tmp) / "responses.db")
            client = ChatClient(spec, cache=cache, attempts=1)
            post = Mock(side_effect=[response(400, error={"param": "temperature", "message": "Unsupported temperature"}),
                                    response(content="result")])
            try:
                with patch.object(client._session, "post", post):
                    self.assertEqual(client.chat("q"), "result")
                    self.assertEqual(client.chat("q"), "result")
                self.assertEqual(post.call_count, 2)
                self.assertEqual(cache.get(client.key_for("q")), "result")
            finally:
                cache.close()
                client._session.close()

    def test_caption_cache_tracks_accepted_parameters(self):
        calls = []
        class Rejected(Exception):
            status_code = 400
        def create(**kwargs):
            calls.append(copy.deepcopy(kwargs))
            if "temperature" in kwargs:
                raise Rejected("Unsupported parameter: temperature")
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content=json.dumps({"summary": "Sample figure"})), finish_reason="stop")])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with tempfile.TemporaryDirectory() as tmp, patch.object(vlm, "_client_for", return_value=client), \
                patch.object(vlm, "image_data_url", return_value="data:,"):
            args = dict(image_path=Path("unused.png"), base_url="https://models.example/v1", api_key="", model_id="alias",
                        cache_json=Path(tmp) / "caption.json")
            vlm.caption_image(**args)
            second = vlm.caption_image(**args)
            self.assertTrue(second["vlm_cache_hit"])
            self.assertEqual(len(calls), 2)
            saved = json.loads(args["cache_json"].read_text())
            self.assertNotIn("temperature", saved["_cache"]["request_parameters"])
