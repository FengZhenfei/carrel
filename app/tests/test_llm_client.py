"""Model client: retries, rejection detection, response metadata, cache accounting."""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from kb_pipeline.graph.llm import ChatClient, HTTPStatusError, LLMCache, LLMCallError, LLMCircuitOpen, LLMSpec, cache_count, cache_key

from _support import SPEC, _client


class ChatClientTests(unittest.TestCase):
    def test_cache_key_includes_the_model_and_sampling_params(self) -> None:
        msgs = [{"role": "user", "content": "hi"}]
        k1 = cache_key(SPEC, msgs, {"max_tokens": 10, "temperature": 0.0})
        self.assertNotEqual(k1, cache_key(LLMSpec("m", "http://x/v1", "", "model-b"), msgs, {"max_tokens": 10, "temperature": 0.0}))
        self.assertNotEqual(k1, cache_key(SPEC, msgs, {"max_tokens": 11, "temperature": 0.0}))
        # base_url / api_key are not part of the key
        self.assertEqual(k1, cache_key(LLMSpec("n", "http://y", "k", "model-a"), msgs, {"max_tokens": 10, "temperature": 0.0}))

    def test_cache_hits_skip_the_call_and_persist_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.sqlite"
            calls = {"n": 0}

            def chat(messages):
                calls["n"] += 1
                return "OUT"
            client = _client(chat, cache=LLMCache(path))
            self.assertEqual(client.chat("q"), "OUT")
            self.assertEqual(client.chat("q"), "OUT")
            self.assertEqual((calls["n"], client.stats["cache_hits"]), (1, 1))
            client.cache.close()
            self.assertEqual(cache_count(path), 1)
            again = _client(chat, cache=LLMCache(path))
            self.assertEqual(again.chat("q"), "OUT")
            self.assertEqual(calls["n"], 1)
            again.cache.close()
        self.assertEqual(cache_count(Path(tmp) / "missing.sqlite"), 0)

    def test_retries_transient_errors_but_not_deterministic_ones(self) -> None:
        client = _client([HTTPStatusError(503, "down"), RuntimeError("empty response"), "OK"], attempts=3)
        self.assertEqual(client.chat("q"), "OK")
        self.assertEqual(client.stats["retries"], 2)
        client = _client([HTTPStatusError(400, "bad request"), "never"], attempts=3)
        with self.assertRaises(LLMCallError):
            client.chat("q")
        self.assertEqual(client.stats["failures"], 1)

    def test_circuit_opens_after_consecutive_failures_and_stops_the_pool(self) -> None:
        client = _client([HTTPStatusError(500, "x")] * 10, attempts=1, circuit_fails=2, workers=1)
        with self.assertRaises(LLMCallError):
            client.chat("a")
        with self.assertRaises(LLMCircuitOpen):
            client.chat("b")
        self.assertTrue(client.stop.is_set())
        with self.assertRaises(LLMCircuitOpen):
            client.chat("c")

    def test_run_parallel_keeps_order_and_isolates_failures(self) -> None:
        def chat(messages):
            text = messages[-1]["content"]
            if text == "bad":
                raise HTTPStatusError(400, "no")
            return text.upper()
        client = _client(chat, workers=3, attempts=1)
        seen: list[tuple[int, int]] = []
        out = client.run_parallel(["a", "bad", "c"], lambda item: client.chat(item), progress=lambda d, t: seen.append((d, t)))
        self.assertEqual([(item, res) for item, res, _ in out], [("a", "A"), ("bad", None), ("c", "C")])
        self.assertIsInstance(out[1][2], LLMCallError)
        self.assertEqual(seen[-1], (3, 3))

    def test_stop_event_interrupts_waiting_workers(self) -> None:
        from kb_pipeline.graph.llm import LLMInterrupted

        stop = threading.Event()
        stop.set()
        client = _client(["x"], stop=stop)
        with self.assertRaises(LLMInterrupted):
            client.chat("q")

    def test_run_parallel_reports_the_circuit_even_when_an_interrupted_worker_finishes_first(self) -> None:
        """A circuit break halts the other threads by setting stop, and they raise LLMInterrupted; which result the
        main thread receives first is not fixed. When the halted thread's arrives first, what is reported is still
        the circuit break with its reason, not "stop requested"."""
        import time

        from kb_pipeline.graph.llm import LLMInterrupted

        def chat(messages):
            raise HTTPStatusError(503, "provider down")

        client = _client(chat, attempts=1, circuit_fails=1, workers=2)

        def work(item):
            if item == "trips":
                try:
                    return client.chat("q")
                except LLMCircuitOpen:
                    time.sleep(0.3)             # the thread that tripped the circuit hands its result back a bit later
                    raise
            client.stop.wait(5)                 # the other thread waits in its backoff and is woken by the circuit break
            raise LLMInterrupted("stop requested")

        with self.assertRaises(LLMCircuitOpen) as ctx:
            client.run_parallel(["trips", "waits"], work)
        self.assertIn("provider down", str(ctx.exception))
        # a stop without a circuit break (signal, pause) is still reported as an interruption
        plain = _client(["x"], workers=2)

        def stopped(item):
            plain.stop.set()
            raise LLMInterrupted("stop requested")

        with self.assertRaises(LLMInterrupted):
            plain.run_parallel(["a"], stopped)


class EmptyResponseRetryTests(unittest.TestCase):
    def test_empty_responses_are_retried_once_only(self) -> None:
        """Health check R7: an empty response is most likely deterministic, so it is retried only once instead
        of going through the full five-attempt backoff."""
        from kb_pipeline.graph.llm import LLMEmptyResponse

        client = _client([LLMEmptyResponse("empty response")] * 5, attempts=5)
        with self.assertRaises(LLMCallError):
            client.chat("q")
        self.assertEqual(client.stats["retries"], 1)
        client = _client([LLMEmptyResponse("empty response"), "OK"], attempts=5)
        self.assertEqual(client.chat("q"), "OK")


class ProviderRejectionTests(unittest.TestCase):
    """2026-09-11 on the real box: Alibaba Cloud rejected one unit with a 400 from content moderation, chat_once
    mistook it for "enable_thinking rejected", and from then on the whole process stopped disabling thinking;
    every later unit had its output budget eaten by reasoning and came back empty, 20 consecutive empty
    responses tripped the circuit breaker, and the graph build failed three times at the same spot."""

    def setUp(self) -> None:
        from kb_pipeline.graph import llm as llm_mod
        self.llm = llm_mod
        self.base = f"http://reject-{id(self)}/v1"
        self.spec = LLMSpec(name="m", base_url=self.base, api_key="", model_id="x", protocol="openai")
        self.addCleanup(llm_mod._THINKING_PARAMS_REJECTED.pop, self.base, None)

    @staticmethod
    def _resp(status: int, payload: dict):
        return SimpleNamespace(status_code=status, text=json.dumps(payload, ensure_ascii=False), json=lambda: payload)

    @staticmethod
    def _session(responses: list):
        queue = list(responses)
        calls: list[dict] = []

        def post(url, headers=None, json=None, timeout=None, allow_redirects=True):
            calls.append(dict(json))
            return queue.pop(0)
        return SimpleNamespace(post=post), calls

    def _ask(self, session, **kw) -> str:
        return self.llm.chat_once(self.spec, [{"role": "user", "content": "q"}], session=session, **kw)

    def test_content_inspection_400_is_a_unit_failure_and_keeps_the_thinking_flag(self) -> None:
        body = {"error": {"code": "data_inspection_failed", "type": "data_inspection_failed",
                          "message": "Input text data may contain inappropriate content."}}
        session, calls = self._session([self._resp(400, body)])
        with self.assertRaises(self.llm.LLMInputRejected) as ctx:
            self._ask(session)
        self.assertEqual(len(calls), 1)                                   # no second request with the parameter removed
        self.assertIn("data_inspection_failed", str(ctx.exception))
        self.assertNotIn(self.base, self.llm._THINKING_PARAMS_REJECTED)
        self.assertFalse(self.llm.is_retryable(ctx.exception))
        ok = {"choices": [{"message": {"content": "fine"}, "finish_reason": "stop"}]}
        session, calls = self._session([self._resp(200, ok)])
        self.assertEqual(self._ask(session), "fine")
        # later calls still disable thinking, and carry both vendors' spellings: Qwen / DashScope's enable_thinking
        # and DeepSeek's official thinking
        self.assertEqual(calls[0]["enable_thinking"], False)
        self.assertEqual(calls[0]["thinking"], {"type": "disabled"})

    def test_unknown_parameter_400_drops_the_flag_only_when_the_retry_succeeds(self) -> None:
        generic = {"error": {"message": "invalid request", "type": "invalid_request_error"}}
        ok = {"choices": [{"message": {"content": "fine"}, "finish_reason": "stop"}]}
        session, calls = self._session([self._resp(400, generic), self._resp(200, ok)])
        self.assertEqual(self._ask(session), "fine")
        self.assertEqual(len(calls), 2)
        self.assertNotIn("enable_thinking", calls[1])
        self.assertNotIn("thinking", calls[1])
        self.assertEqual(self.llm._THINKING_PARAMS_REJECTED[self.base], {"enable_thinking", "thinking"})   # remembered only because the retry without it succeeded
        self.llm._THINKING_PARAMS_REJECTED.pop(self.base)
        session, calls = self._session([self._resp(400, generic), self._resp(400, generic)])
        with self.assertRaises(self.llm.HTTPStatusError):                  # still 400 without it: the original error is raised, nothing remembered
            self._ask(session)
        self.assertEqual(len(calls), 2)
        self.assertNotIn(self.base, self.llm._THINKING_PARAMS_REJECTED)

    def test_400_naming_one_parameter_drops_only_that_one_and_remembers_it(self) -> None:
        named = {"error": {"message": "Unrecognized request argument supplied: thinking"}}
        session, calls = self._session([self._resp(400, named), self._resp(429, {"error": {"message": "rate"}})])
        with self.assertRaises(self.llm.HTTPStatusError) as ctx:
            self._ask(session)
        self.assertEqual(ctx.exception.status, 429)                       # the result of the resent request is raised as is (429 is retryable)
        self.assertNotIn("thinking", calls[1])
        self.assertEqual(calls[1]["enable_thinking"], False)              # the one not named is still sent
        self.assertEqual(self.llm._THINKING_PARAMS_REJECTED[self.base], {"thinking"})
        ok = {"choices": [{"message": {"content": "fine"}, "finish_reason": "stop"}]}
        session, calls = self._session([self._resp(200, ok)])
        self.assertEqual(self._ask(session), "fine")
        self.assertEqual(set(calls[0]) & {"enable_thinking", "thinking"}, {"enable_thinking"})
        # the other way round, naming enable_thinking: "thinking" is a substring of it and must not be dropped too
        self.llm._THINKING_PARAMS_REJECTED.pop(self.base)
        named = {"error": {"message": "Unrecognized request argument supplied: enable_thinking"}}
        session, calls = self._session([self._resp(400, named), self._resp(200, ok)])
        self.assertEqual(self._ask(session), "fine")
        self.assertNotIn("enable_thinking", calls[1])
        self.assertEqual(calls[1]["thinking"], {"type": "disabled"})
        self.assertEqual(self.llm._THINKING_PARAMS_REJECTED[self.base], {"enable_thinking"})

    def test_deepseek_style_interrupted_generation_is_retried_with_backoff(self) -> None:
        """DeepSeek's official docs: finish_reason=insufficient_system_resource / aborted means generation was
        interrupted; the content may be empty or cut off, and either way it goes through the full backoff as a
        retryable error rather than the single retry of an empty response."""
        payload = {"choices": [{"message": {"content": "(\"entity\"<|>半截"}, "finish_reason": "insufficient_system_resource"}]}
        session, _ = self._session([self._resp(200, payload)])
        with self.assertRaises(self.llm.LLMTransientResponse) as ctx:
            self._ask(session)
        self.assertTrue(self.llm.is_retryable(ctx.exception))
        client = _client([self.llm.LLMTransientResponse("中断")] * 3 + ["OK"], attempts=5)
        self.assertEqual(client.chat("q"), "OK")
        self.assertEqual(client.stats["retries"], 3)

    def test_reasoning_exhausted_empty_content_is_named_and_not_retried(self) -> None:
        payload = {"choices": [{"message": {"content": "", "reasoning_content": "We need to..."}, "finish_reason": "length"}],
                   "usage": {"completion_tokens": 4096, "completion_tokens_details": {"reasoning_tokens": 4096}}}
        session, _ = self._session([self._resp(200, payload)])
        with self.assertRaises(self.llm.LLMReasoningExhausted) as ctx:
            self._ask(session, max_tokens=4096)
        self.assertIn("reasoning", str(ctx.exception))
        self.assertIn("4096", str(ctx.exception))
        self.assertFalse(self.llm.is_retryable(ctx.exception))
        self.assertIsInstance(ctx.exception, self.llm.LLMEmptyResponse)
        # through ChatClient: no retry, counted as a consecutive failure, and the circuit reason carries the message
        client = _client([self.llm.LLMReasoningExhausted("推理吃光")] * 3, attempts=5, circuit_fails=2)
        with self.assertRaises(LLMCallError):
            client.chat("q1")
        with self.assertRaises(self.llm.LLMCircuitOpen) as ctx2:
            client.chat("q2")
        self.assertEqual(client.stats["retries"], 0)
        self.assertIn("推理吃光", str(ctx2.exception))

    def test_plain_empty_content_still_retries_once(self) -> None:
        payload = {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}
        session, _ = self._session([self._resp(200, payload)])
        with self.assertRaises(self.llm.LLMEmptyResponse) as ctx:
            self._ask(session)
        self.assertNotIsInstance(ctx.exception, self.llm.LLMReasoningExhausted)
        self.assertTrue(self.llm.is_retryable(ctx.exception))

    def test_content_filter_finish_reason_is_an_input_rejection(self) -> None:
        payload = {"choices": [{"message": {"content": ""}, "finish_reason": "content_filter"}]}
        session, _ = self._session([self._resp(200, payload)])
        with self.assertRaises(self.llm.LLMInputRejected):
            self._ask(session)

    def test_input_rejections_do_not_count_toward_the_circuit(self) -> None:
        client = _client([self.llm.LLMInputRejected("审核")] * 5, attempts=5, circuit_fails=2)
        for _ in range(5):
            with self.assertRaises(self.llm.LLMInputRejected):
                client.chat("q")
        self.assertEqual(client.stats["rejected"], 5)
        self.assertEqual(client.stats["consecutive_failures"], 0)
        self.assertEqual(client.stats["retries"], 0)
        self.assertIsNone(client.circuit_reason)


class ResponseMetaTests(unittest.TestCase):
    """The meta of chat_once / ChatClient.chat: finish_reason and "truncated by max_tokens" must reach the caller,
    and truncated responses must not be cached (once cached, a resumed run would never see that signal)."""

    def test_chat_once_fills_meta(self) -> None:
        from kb_pipeline.graph import llm as llm_mod
        spec = LLMSpec(name="m", base_url=f"http://meta-{id(self)}/v1", api_key="", model_id="x", protocol="openai")
        self.addCleanup(llm_mod._THINKING_PARAMS_REJECTED.pop, spec.base_url, None)
        payload = {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}], "usage": {"completion_tokens": 4096}}
        session = SimpleNamespace(post=lambda *a, **k: SimpleNamespace(status_code=200, text=json.dumps(payload), json=lambda: payload))
        meta: dict = {}
        self.assertEqual(llm_mod.chat_once(spec, [{"role": "user", "content": "q"}], session=session, meta=meta), "partial")
        self.assertEqual((meta["finish_reason"], meta["truncated"], meta["completion_tokens"], meta["cached"]), ("length", True, 4096, False))
        payload["choices"][0]["finish_reason"] = "stop"
        meta = {}
        llm_mod.chat_once(spec, [{"role": "user", "content": "q"}], session=session, meta=meta)
        self.assertFalse(meta["truncated"])

    def test_truncated_responses_are_not_cached(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LLMCache(Path(tmp) / "c.sqlite")
            cut = iter([True, False])

            def chat(spec, messages, meta=None, **_):
                truncated = next(cut)
                if meta is not None:
                    meta.update(finish_reason="length" if truncated else "stop", truncated=truncated, cached=False)
                return "R"
            client = ChatClient(SPEC, chat=chat, cache=cache, backoff_base=0.0, backoff_max=0.0)
            meta: dict = {}
            self.assertEqual(client.chat("q", meta=meta), "R")
            self.assertTrue(meta["truncated"])
            self.assertEqual(cache.count(), 0)                    # truncated responses are not cached
            meta = {}
            client.chat("q", meta=meta)
            self.assertFalse(meta["truncated"])
            self.assertEqual(cache.count(), 1)
            meta = {}
            self.assertEqual(client.chat("q", meta=meta), "R")
            self.assertTrue(meta["cached"])
            cache.close()


class ConnectionPoolTests(unittest.TestCase):
    def test_session_pool_follows_the_worker_count(self) -> None:
        """2026-09-12 concurrency was clean all the way up to 256: requests keeps 10 persistent connections per
        host by default, so at higher concurrency the remaining connections are discarded after one use and
        re-handshake every time; the pool has to scale with the worker count."""
        client = ChatClient(SPEC, workers=64)
        self.assertEqual(client._session.get_adapter("https://api.deepseek.com/x")._pool_maxsize, 64)
        self.assertEqual(ChatClient(SPEC, workers=2)._session.get_adapter("http://x/")._pool_maxsize, 10)


class LlmCallAccountingTests(unittest.TestCase):
    """The graph-QA generation protocol was deleted together with answer.py on 2026-09-09 (answering happens on
    the hybrid-recall side); only the model client's accounting remains here."""


    def test_chat_client_counts_tokens_of_real_calls_only(self) -> None:
        from kb_pipeline.graph.llm import LLMCache

        with tempfile.TemporaryDirectory() as tmp:
            client = _client(["回答一", "回答二"], cache=LLMCache(Path(tmp) / "c.sqlite"))
            client.chat("问题一")
            client.chat("问题一")            # cache hit: no tokens counted
            self.assertEqual((client.stats["calls"], client.stats["cache_hits"]), (1, 1))
            self.assertGreater(client.stats["prompt_tokens"], 0)
            self.assertGreater(client.stats["completion_tokens"], 0)
            before = dict(client.stats)
            client.chat("问题二")
            self.assertEqual(client.stats["calls"], 2)
            self.assertGreater(client.stats["prompt_tokens"], before["prompt_tokens"])
            client.cache.close()

    def test_figure_units_are_body_not_conclusion_or_toc(self) -> None:
        from kb_pipeline.graph.units import classify_unit_text, unit_signals

        figure = ("CAPTION: 逻辑框图[1、2、3]\nVISUAL SUMMARY: 这是一个静态随机存取存储器(SRAM)阵列的内部结构框图,展示了行解码器、列 I/O、控制单元等组件及其连接关系。\n"
                  "FACTS: 静态 RAM 阵列规格为 2048 X 2048 X 2。;行解码器接收地址输入 A0 至 A19。\nENTITIES: STATIC RAM ARRAY, ROW DECODER\nKEYWORDS: SRAM\n"
                  "```mermaid\ngraph LR\n" + "\n".join(f'  N{i}["A{i}-A{i+1}"] --> N{i+1}["ROW {i}"]' for i in range(24)) + "\n```\n"
                  "FOOTNOTE: 勘误表:在器件中,AutoStore Disable 特性被禁用。")
        sig = unit_signals(figure, ["功能说明"])
        self.assertFalse(sig["conclusion"])
        self.assertLess(sig["toc"], 5)
        self.assertEqual(classify_unit_text(figure, ["功能说明"]), "body")
        self.assertEqual(classify_unit_text("CAPTION: 图 5. 读周期\nVISUAL SUMMARY: 时序图。", ["开关波形"]), "body")
        self.assertEqual(classify_unit_text("异常结果汇总\n1. 总胆固醇偏高,建议复查。", ["异常结果汇总"]), "conclusion")   # a real conclusion section is unaffected
        self.assertEqual(classify_unit_text("Summary\nThe device passed all tests.", ["Summary"]), "conclusion")


    def test_cli_no_longer_exposes_answer_and_eval(self) -> None:
        root = Path(__file__).resolve().parents[1] / "kb_pipeline"
        src = (root / "cli.py").read_text(encoding="utf-8")
        for needle in ('graph_sub.add_parser("answer"', 'graph_sub.add_parser("eval"', 'args.graph_command in ("answer", "eval")',
                       "kb_pipeline.graph.answer", "graph answer", "graph eval"):
            self.assertNotIn(needle, src, needle)
        self.assertIn('graph_sub.add_parser("factcheck"', src)
        self.assertFalse((root / "graph" / "answer.py").exists())
        self.assertNotIn("ANSWER_PROMPT", (root / "graph" / "prompts.py").read_text(encoding="utf-8"))
