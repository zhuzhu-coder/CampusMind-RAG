from fastapi.testclient import TestClient
from concurrent.futures import ThreadPoolExecutor
import time

from campus_rag.pipeline import RAGResponse, RAGTrace, RetrievedSource


class FakeDataModule:
    def get_statistics(self):
        return {
            "total_documents": 23,
            "total_chunks": 39,
            "categories": {"教务教学": 5},
            "departments": {"教务处": 3},
            "file_types": {"md": 8, "txt": 9, "pdf": 6},
        }


class FakeRAGSystem:
    def __init__(self):
        self.initialize_calls = 0
        self.build_calls = 0
        self.ask_calls = []
        self.data_module = FakeDataModule()

    def initialize_system(self):
        self.initialize_calls += 1

    def build_knowledge_base(self):
        self.build_calls += 1

    def ask_question(self, question, stream=False, return_sources=False, return_trace=False):
        self.ask_calls.append(
            {
                "question": question,
                "stream": stream,
                "return_sources": return_sources,
                "return_trace": return_trace,
            }
        )
        trace = None
        if return_trace:
            trace = RAGTrace(
                retrieval_strategy="hybrid",
                filters={},
                timings_ms={
                    "analysis": 1.0,
                    "retrieval": 3.0,
                    "context_build": 4.0,
                    "generation": 5.0,
                    "total": 13.0,
                },
                retrieval_params={
                    "top_k": 3,
                    "candidate_k": 10,
                    "rrf_k": 60,
                    "context_window_size": 1,
                },
                retrieved_chunks=[
                    {
                        "rank": 1,
                        "doc_title": "学生请假管理办法",
                        "section": "请假审批",
                        "chunk_index": 0,
                        "rrf_score": 0.03,
                    }
                ],
                context_documents=[
                    {
                        "source_id": 1,
                        "doc_title": "学生请假管理办法",
                        "context_window_size": 1,
                        "context_chunk_indices": [0, 1],
                    }
                ],
                source_count=1,
            )
        return RAGResponse(
            question=question,
            route_type="detail",
            rewritten_query=question,
            answer=f"answer:{question}",
            sources=[
                RetrievedSource(
                    source_id=1,
                    doc_title="学生请假管理办法",
                    doc_category="规章制度",
                    department="学生处",
                    file_type="md",
                    section="请假审批",
                    source="data/campus/regulations/student_affairs/学生请假管理办法.md",
                    page=None,
                    chunk_index=0,
                    rrf_score=0.03,
                    snippet="学生请假超过三天需由辅导员和学院审批。",
                )
            ],
            trace=trace,
        )


def build_client():
    from campus_rag.api import create_app

    created_systems = []

    def factory():
        system = FakeRAGSystem()
        created_systems.append(system)
        return system

    return TestClient(create_app(system_factory=factory)), created_systems


def build_client_with_factory(factory):
    from campus_rag.api import create_app

    return TestClient(create_app(system_factory=factory))


def test_health_does_not_initialize_rag_system():
    client, created_systems = build_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert created_systems == []


def test_root_serves_frontend_shell_without_initializing_rag_system():
    client, created_systems = build_client()

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert 'id="app"' in response.text
    assert "校园知识库 RAG" in response.text
    assert created_systems == []


def test_static_frontend_assets_are_served():
    client, created_systems = build_client()

    script_response = client.get("/static/app.js")
    style_response = client.get("/static/styles.css")

    assert script_response.status_code == 200
    assert "javascript" in script_response.headers["content-type"]
    assert "checkReady" in script_response.text
    assert style_response.status_code == 200
    assert "text/css" in style_response.headers["content-type"]
    assert ".app-shell" in style_response.text
    assert created_systems == []


def test_response_includes_request_id_and_process_time_headers():
    client, _ = build_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]
    assert float(response.headers["X-Process-Time-MS"]) >= 0


def test_response_preserves_incoming_request_id_header():
    client, _ = build_client()

    response = client.get("/health", headers={"X-Request-ID": "req-from-client"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req-from-client"


def test_ready_reports_not_ready_before_rag_initialization():
    client, created_systems = build_client()

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "ready": False,
        "status": "not_ready",
        "total_documents": 0,
        "total_chunks": 0,
        "last_error": None,
    }
    assert created_systems == []


def test_warmup_initializes_rag_system_and_returns_ready_status():
    client, created_systems = build_client()

    response = client.post("/warmup")

    assert response.status_code == 200
    assert response.json() == {
        "ready": True,
        "status": "ready",
        "total_documents": 23,
        "total_chunks": 39,
        "last_error": None,
    }
    assert len(created_systems) == 1
    assert created_systems[0].initialize_calls == 1
    assert created_systems[0].build_calls == 1


def test_ready_reports_initialized_rag_statistics_after_warmup():
    client, _ = build_client()

    client.post("/warmup")
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "ready": True,
        "status": "ready",
        "total_documents": 23,
        "total_chunks": 39,
        "last_error": None,
    }


def test_rag_service_initializes_only_once_for_concurrent_callers():
    from campus_rag.api import RAGService

    created_systems = []

    class SlowFakeRAGSystem(FakeRAGSystem):
        def initialize_system(self):
            time.sleep(0.02)
            super().initialize_system()

    def factory():
        system = SlowFakeRAGSystem()
        created_systems.append(system)
        return system

    service = RAGService(factory)

    with ThreadPoolExecutor(max_workers=8) as executor:
        systems = list(executor.map(lambda _: service.get_system(), range(8)))

    assert len(created_systems) == 1
    assert created_systems[0].initialize_calls == 1
    assert created_systems[0].build_calls == 1
    assert all(system is created_systems[0] for system in systems)


def test_warmup_returns_structured_error_when_initialization_fails():
    class FailingInitializeRAGSystem(FakeRAGSystem):
        def initialize_system(self):
            raise RuntimeError("missing api key")

    client = build_client_with_factory(FailingInitializeRAGSystem)

    response = client.post("/warmup", headers={"X-Request-ID": "req-init-fail"})

    assert response.status_code == 503
    assert response.headers["X-Request-ID"] == "req-init-fail"
    assert response.json() == {
        "error": {
            "code": "rag_initialization_failed",
            "message": "RAG service failed to initialize",
            "request_id": "req-init-fail",
        }
    }


def test_ask_returns_structured_error_when_rag_call_fails():
    class FailingAskRAGSystem(FakeRAGSystem):
        def ask_question(self, question, stream=False, return_sources=False, return_trace=False):
            raise RuntimeError("llm timeout")

    client = build_client_with_factory(FailingAskRAGSystem)

    response = client.post(
        "/ask",
        json={"question": "学生请假超过三天需要谁审批？"},
        headers={"X-Request-ID": "req-ask-fail"},
    )

    assert response.status_code == 500
    assert response.headers["X-Request-ID"] == "req-ask-fail"
    assert response.json() == {
        "error": {
            "code": "internal_error",
            "message": "服务内部错误",
            "request_id": "req-ask-fail",
        }
    }


def test_ask_returns_structured_rag_response_with_sources():
    client, created_systems = build_client()

    response = client.post(
        "/ask",
        json={"question": "学生请假超过三天需要谁审批？", "return_sources": True},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["question"] == "学生请假超过三天需要谁审批？"
    assert payload["route_type"] == "detail"
    assert payload["rewritten_query"] == "学生请假超过三天需要谁审批？"
    assert payload["answer"] == "answer:学生请假超过三天需要谁审批？"
    assert payload["sources"][0]["doc_title"] == "学生请假管理办法"
    assert created_systems[0].initialize_calls == 1
    assert created_systems[0].build_calls == 1
    assert created_systems[0].ask_calls == [
        {
            "question": "学生请假超过三天需要谁审批？",
            "stream": False,
            "return_sources": True,
            "return_trace": False,
        }
    ]
    assert "trace" not in payload


def test_ask_can_hide_sources_while_preserving_response_shape():
    client, _ = build_client()

    response = client.post(
        "/ask",
        json={"question": "校园卡丢了第一步应该做什么？", "return_sources": False},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["question"] == "校园卡丢了第一步应该做什么？"
    assert payload["answer"] == "answer:校园卡丢了第一步应该做什么？"
    assert payload["sources"] == []


def test_ask_returns_trace_when_requested():
    client, created_systems = build_client()

    response = client.post(
        "/ask",
        json={
            "question": "学生请假超过三天需要谁审批？",
            "return_sources": True,
            "return_trace": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["trace"]["retrieval_strategy"] == "hybrid"
    assert payload["trace"]["timings_ms"]["retrieval"] == 3.0
    assert payload["trace"]["retrieved_chunks"][0]["doc_title"] == "学生请假管理办法"
    assert created_systems[0].ask_calls == [
        {
            "question": "学生请假超过三天需要谁审批？",
            "stream": False,
            "return_sources": True,
            "return_trace": True,
        }
    ]


def test_ask_can_hide_sources_while_returning_trace():
    client, _ = build_client()

    response = client.post(
        "/ask",
        json={
            "question": "校园卡丢了第一步应该做什么？",
            "return_sources": False,
            "return_trace": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["sources"] == []
    assert payload["trace"]["source_count"] == 1


def test_stats_returns_knowledge_base_statistics():
    client, created_systems = build_client()

    response = client.get("/stats")

    assert response.status_code == 200
    assert response.json() == {
        "total_documents": 23,
        "total_chunks": 39,
        "categories": {"教务教学": 5},
        "departments": {"教务处": 3},
        "file_types": {"md": 8, "txt": 9, "pdf": 6},
    }
    assert created_systems[0].initialize_calls == 1
    assert created_systems[0].build_calls == 1


def test_empty_question_is_rejected_before_rag_call():
    client, created_systems = build_client()

    response = client.post("/ask", json={"question": "   ", "return_sources": True})

    assert response.status_code == 422
    assert created_systems == []

