from rag_modules import RAGResponse, RetrievedSource


def test_rag_response_serializes_sources_to_plain_dicts():
    source = RetrievedSource(
        source_id=1,
        doc_title="学生请假管理办法",
        doc_category="规章制度",
        department="学生处",
        file_type="md",
        section="请假审批",
        source="data/campus/regulations/学生请假管理办法.md",
        page=None,
        chunk_index=2,
        rrf_score=0.0325,
        snippet="学生请假超过三天需由辅导员和学院审批。",
    )
    response = RAGResponse(
        question="学生请假超过三天需要谁审批？",
        route_type="detail",
        rewritten_query="学生请假超过三天需要谁审批？",
        answer="需要辅导员和学院审批。[1]",
        sources=[source],
    )

    assert response.to_dict() == {
        "question": "学生请假超过三天需要谁审批？",
        "route_type": "detail",
        "rewritten_query": "学生请假超过三天需要谁审批？",
        "answer": "需要辅导员和学院审批。[1]",
        "sources": [
            {
                "source_id": 1,
                "doc_title": "学生请假管理办法",
                "doc_category": "规章制度",
                "department": "学生处",
                "file_type": "md",
                "section": "请假审批",
                "source": "data/campus/regulations/学生请假管理办法.md",
                "page": None,
                "chunk_index": 2,
                "rrf_score": 0.0325,
                "snippet": "学生请假超过三天需由辅导员和学院审批。",
            }
        ],
    }
