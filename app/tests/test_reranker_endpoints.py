"""Hosted endpoint configuration and local model-discovery compatibility."""
from unittest.mock import Mock

from kb_pipeline.config import load_settings
from kb_search import rerank as module


def response(body):
    return Mock(ok=True, json=Mock(return_value=body))


def test_explicit_model_sends_bearer_without_model_discovery(monkeypatch):
    get = Mock(side_effect=AssertionError("this provider has no /models endpoint"))
    post = Mock(return_value=response({"results": [{"index": 1, "relevance_score": 0.9},
                                                  {"index": 0, "relevance_score": 0.2}]}))
    monkeypatch.setattr(module._session, "get", get)
    monkeypatch.setattr(module._session, "post", post)
    client = module.Reranker("https://models.example/v2/rerank", model_id="chosen-model", api_key="test-key")
    assert client.score("question", ["first", "second"]) == [0.2, 0.9]
    get.assert_not_called()
    post.assert_called_once_with("https://models.example/v2/rerank",
        json={"model": "chosen-model", "query": "question", "documents": ["first", "second"]},
        timeout=20.0, headers={"Authorization": "Bearer test-key"})


def test_discovery_cache_is_scoped_to_credentials(monkeypatch):
    monkeypatch.setattr(module, "_model_ids", {})
    get = Mock(side_effect=[response({"data": [{"id": "account-a-model"}]}),
                            response({"data": [{"id": "account-b-model"}]})])
    monkeypatch.setattr(module._session, "get", get)
    a = module.Reranker("https://models.example/v1", api_key="account-a")
    b = module.Reranker("https://models.example/v1", api_key="account-b")
    assert a.resolve_model() == "account-a-model"
    assert b.resolve_model() == "account-b-model"
    assert a.resolve_model() == "account-a-model"
    assert get.call_count == 2
    assert get.call_args_list[0].kwargs["headers"] == {"Authorization": "Bearer account-a"}
    assert get.call_args_list[1].kwargs["headers"] == {"Authorization": "Bearer account-b"}


def test_anonymous_local_endpoint_keeps_model_discovery(monkeypatch):
    monkeypatch.setattr(module, "_model_ids", {})
    get = Mock(return_value=response({"data": [{"id": "local-model"}]}))
    post = Mock(return_value=response({"results": [{"index": 0, "relevance_score": 0.8}]}))
    monkeypatch.setattr(module._session, "get", get)
    monkeypatch.setattr(module._session, "post", post)
    assert module.Reranker("http://localhost:8102").score("q", ["d"]) == [0.8]
    get.assert_called_once_with("http://localhost:8102/v1/models", timeout=5.0)
    assert post.call_args.args == ("http://localhost:8102/v1/rerank",)
    assert "headers" not in post.call_args.kwargs


def test_settings_load_explicit_model_and_key(tmp_path, monkeypatch):
    monkeypatch.setenv("RERANKER_BASE_URL", "https://models.example/v1")
    monkeypatch.setenv("RERANKER_MODEL_ID", "chosen-model")
    monkeypatch.setenv("RERANKER_API_KEY", "test-key")
    monkeypatch.setenv("KB_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("KB_STATE_DB", str(tmp_path / "state.db"))
    cfg = load_settings(tmp_path / "empty.env")
    assert (cfg.reranker_model_id, cfg.reranker_api_key) == ("chosen-model", "test-key")


def test_search_text_branch_keeps_hosted_model_and_credentials(monkeypatch):
    import time
    from types import SimpleNamespace
    from kb_search import service
    from kb_search.config import load_search_settings

    factory = Mock(return_value=Mock())
    monkeypatch.setattr(service, "Reranker", factory)
    monkeypatch.setattr(service, "rerank", Mock(return_value=[0.8]))
    cfg = SimpleNamespace(reranker_base_url="https://models.example/v2/rerank",
                          reranker_model_id="chosen-model", reranker_api_key="test-key")
    result = service._rank_text(cfg, load_search_settings({}), "question",
                                [{"point_id": "example", "payload": {"text": "source"}}],
                                until=time.time() + 3)
    assert result["status"] == "ok"
    assert factory.call_args.args == (cfg.reranker_base_url,)
    assert factory.call_args.kwargs["model_id"] == "chosen-model"
    assert factory.call_args.kwargs["api_key"] == "test-key"
