"""Synthetic batching contracts: correspondence, cache reuse, and bounded work."""
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests
from PIL import Image

from kb_pipeline.embedding import client as text_client, visual


def text_embedder(monkeypatch, create, **kwargs):
    monkeypatch.setattr(text_client, "_client_for", lambda *a: SimpleNamespace(embeddings=SimpleNamespace(create=create)))
    return text_client.EmbeddingClient(base_url="http://synthetic.invalid/v1", api_key="", model_id="test",
                                       dim=2, batch_size=2, sleep_seconds=0, retry=1, **kwargs)


def test_text_concurrent_batches_preserve_order_progress_and_limit(monkeypatch):
    lock = threading.Lock()
    active = maximum = 0
    calls, finished = [], []
    gate = threading.Event()
    def create(**kw):
        nonlocal active, maximum
        batch = kw["input"]
        with lock:
            calls.append(batch)
            active += 1
            maximum = max(active, maximum)
        try:
            if batch[0] == "0":
                assert gate.wait(3)
            else:
                gate.set()
            time.sleep(0.01)
            finished.append(batch[0])
            return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[float(x), 1.0]) for i, x in enumerate(batch)])
        finally:
            with lock:
                active -= 1
    client = text_embedder(monkeypatch, create, concurrency=2)
    progress = []
    vectors = client.embed([str(i) for i in range(5)], progress_cb=lambda *v: progress.append((*v, threading.get_ident())))
    assert vectors == [[float(i), 1.0] for i in range(5)]
    assert sorted(calls) == [["0", "1"], ["2", "3"], ["4"]]
    assert maximum == 2
    assert progress == [(i, 3, threading.get_ident()) for i in range(4)]


def test_text_rejected_input_has_global_index_across_concurrent_batches(monkeypatch):
    class Rejected(Exception):
        status_code = 400
    bad = "X" * 80
    def create(**kw):
        if bad in kw["input"]:
            raise Rejected("input too long")
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[0.0, 1.0]) for i, _ in enumerate(kw["input"])])
    client = text_embedder(monkeypatch, create, concurrency=2)
    with pytest.raises(text_client.EmbeddingInputRejected) as caught:
        client.embed(["a", "b", "c", bad, "e"])
    assert caught.value.index == 3


@pytest.fixture
def pictures(tmp_path):
    jobs = []
    for i in range(5):
        path = tmp_path / f"image-{i}.png"
        Image.new("RGB", (16, 16), (i * 40, 80, 120)).save(path)
        jobs.append((path, tmp_path / f"cache-{i}.json"))
    return jobs


def visual_embedder(monkeypatch, handler, **kwargs):
    def post(url, *, json, **options):
        return handler(json)
    monkeypatch.setattr(visual, "_session_for", lambda *a: SimpleNamespace(post=post))
    client = visual.VisualEmbeddingClient(base_url="http://synthetic.invalid/v1", api_key="", model_id="test",
                                           dim=2, retry=1, **kwargs)
    # The wire structure remains real; names stand in for base64 bytes.
    monkeypatch.setattr(visual, "image_data_url", lambda path, pixels: path.stem)
    return client


def conversations(body):
    messages = body["messages"]
    return [messages] if isinstance(messages[0], dict) else messages


def image_ids(body):
    chats = conversations(body)
    assert all(len(chat) == 2 and len(chat[1]["content"]) == 1 for chat in chats)
    return [int(chat[1]["content"][0]["image_url"]["url"].split("-")[-1]) for chat in chats]


def response(body, status=200, text=""):
    return SimpleNamespace(status_code=status, text=text, json=lambda: body)


def valid_response(body):
    return response({"data": [{"index": i, "embedding": [float(n), 1.0]} for i, n in reversed(list(enumerate(image_ids(body))))]})


def test_visual_batch_cache_partial_hits_and_index_mapping(monkeypatch, pictures):
    calls = []
    def handle(body):
        calls.append(body)
        return valid_response(body)
    client = visual_embedder(monkeypatch, handle, batch_size=3, concurrency=1)
    client.embed_image(pictures[1][0], cache_json=pictures[1][1])
    progress = []
    vectors = client.embed_images(pictures, progress_cb=lambda *v: progress.append(v))
    assert vectors == [[float(i), 1.0] for i in range(5)]
    assert [image_ids(body) for body in calls] == [[1], [0, 2], [3, 4]]
    assert isinstance(calls[0]["messages"][0], dict)  # batch=1 keeps the original request
    assert progress == [(i, 5) for i in range(6)]
    # Throughput settings do not change vector space/cache identity.
    other = visual_embedder(monkeypatch, handle, batch_size=1, concurrency=3)
    assert other.embed_images(pictures) == vectors
    assert len(calls) == 3
    for i, (_, cache) in enumerate(pictures):
        assert json.loads(cache.read_text())["vector"] == [float(i), 1.0]


@pytest.mark.parametrize("data", [
    [{"index": 0, "embedding": [0., 1.]}],
    [{"index": 0, "embedding": [0., 1.]}] * 2,
    [{"embedding": [0., 1.]}] * 2,
    [{"index": 0, "embedding": [0., 1.]}, {"index": 1, "embedding": [1.]}],
    [{"index": 0, "embedding": [0., 1.]}, {"index": 1, "embedding": [float("nan"), 1.]}],
])
def test_invalid_visual_batch_never_writes_mismatched_cache(monkeypatch, pictures, data):
    client = visual_embedder(monkeypatch, lambda body: response({"data": data}), batch_size=2, concurrency=1)
    with pytest.raises(RuntimeError):
        client.embed_images(pictures[:2])
    assert not any(cache.exists() for _, cache in pictures)


@pytest.mark.parametrize("poison", [True, False])
def test_visual_rejected_batch_isolates_bad_image_or_recovers_payload_limit(monkeypatch, pictures, poison):
    def handle(body):
        ids = image_ids(body)
        if len(ids) > 1:
            return response({}, status=413, text="payload too large")
        if poison and ids == [1]:
            return response({}, status=400, text="invalid image")
        return valid_response(body)
    client = visual_embedder(monkeypatch, handle, batch_size=3, concurrency=1)
    result = client.embed_images(pictures[:3])
    assert result == [[0., 1.], None if poison else [1., 1.], [2., 1.]]
    assert pictures[1][1].exists() is not poison


def test_visual_transport_failure_raises_and_does_not_skip_batch(monkeypatch, pictures):
    client = visual_embedder(monkeypatch, Mock(side_effect=requests.ConnectionError("offline")), batch_size=2)
    with pytest.raises(requests.ConnectionError):
        client.embed_images(pictures)
    assert not any(cache.exists() for _, cache in pictures)


def test_visual_concurrency_is_requests_not_images_and_preserves_order(monkeypatch, pictures):
    lock, gate = threading.Lock(), threading.Event()
    active = maximum = 0
    def handle(body):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        try:
            if image_ids(body)[0] == 0:
                assert gate.wait(3)
            else:
                gate.set()
            time.sleep(0.01)
            return valid_response(body)
        finally:
            with lock:
                active -= 1
    client = visual_embedder(monkeypatch, handle, batch_size=2, concurrency=2)
    assert client.embed_images(pictures) == [[float(i), 1.] for i in range(5)]
    assert maximum == 2


def test_graph_pause_checks_between_bounded_concurrent_groups():
    from kb_pipeline.graph.build import _StoppableEmbedder
    calls = []
    client = SimpleNamespace(batch_size=2, concurrency=2, embed=lambda texts: calls.append(texts) or [[1.] for _ in texts])
    def check_stop():
        if calls:
            raise RuntimeError("paused")
    with pytest.raises(RuntimeError, match="paused"):
        _StoppableEmbedder(client, check_stop).embed([str(i) for i in range(10)])
    assert calls == [["0", "1", "2", "3"]]
