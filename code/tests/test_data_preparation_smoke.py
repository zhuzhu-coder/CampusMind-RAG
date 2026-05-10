from config import PROJECT_ROOT
from langchain_core.documents import Document
from rag_modules.data_preparation import DataPreparationModule


def test_campus_documents_can_be_loaded_and_chunked():
    module = DataPreparationModule(str(PROJECT_ROOT / "data" / "campus"))

    documents = module.load_documents()
    chunks = module.chunk_documents()
    stats = module.get_statistics()

    assert documents
    assert chunks
    assert stats["total_documents"] >= 1
    assert "规章制度" in stats["categories"]
    assert set(stats["file_types"]) >= {"md", "txt", "pdf"}
    assert all("doc_title" in doc.metadata for doc in documents)
    assert all("section" not in doc.metadata for doc in documents)
    assert all("dish_name" not in doc.metadata for doc in documents)
    assert all("difficulty" not in doc.metadata for doc in documents)
    assert all("chunk_id" in chunk.metadata for chunk in chunks)
    assert all("doc_title" in chunk.metadata for chunk in chunks)
    assert all("section" in chunk.metadata for chunk in chunks)
    assert all("dish_name" not in chunk.metadata for chunk in chunks)
    assert all("difficulty" not in chunk.metadata for chunk in chunks)
    assert "difficulties" not in stats


def test_normalize_parent_metadata_does_not_add_parent_section():
    module = DataPreparationModule(str(PROJECT_ROOT / "data" / "campus"))
    parent_doc = Document(
        page_content="学生请假管理办法",
        metadata={
            "source": "data/campus/regulations/student_affairs/leave.md",
            "relative_path": "regulations/student_affairs/leave.md",
            "doc_title": "学生请假管理办法",
            "doc_id": "stable-parent-id",
        },
    )

    module._normalize_parent_metadata(parent_doc)

    assert parent_doc.metadata["doc_id"] == "stable-parent-id"
    assert parent_doc.metadata["parent_id"] == "stable-parent-id"
    assert parent_doc.metadata["doc_type"] == "parent"
    assert "section" not in parent_doc.metadata


def test_chunk_documents_normalizes_raw_splitter_output_once(monkeypatch):
    module = DataPreparationModule(str(PROJECT_ROOT / "data" / "campus"))
    parent_doc = Document(
        page_content="第一行\r\n\r\n   第二行",
        metadata={
            "doc_title": "测试文档",
            "doc_id": "parent-1",
            "parent_id": "parent-1",
            "doc_type": "parent",
            "file_type": "txt",
        },
    )
    module.documents = [parent_doc]

    def fake_split_parent_document(_parent_doc):
        return [
            Document(page_content=" 第一行\r\n\r\n   第二行 ", metadata={}),
            Document(page_content="   \n\t ", metadata={}),
        ]

    monkeypatch.setattr(module, "_split_parent_document", fake_split_parent_document)

    chunks = module.chunk_documents()

    assert len(chunks) == 1
    assert chunks[0].page_content == "第一行\n\n第二行"
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["batch_index"] == 0
