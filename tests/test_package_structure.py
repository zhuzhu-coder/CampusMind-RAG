from pathlib import Path


def test_public_package_imports_are_available():
    from campus_rag.api import create_app
    from campus_rag.config import PROJECT_ROOT, RAGConfig
    from campus_rag.pipeline import RAGResponse
    from campus_rag.system import CampusRAGSystem

    assert create_app is not None
    assert CampusRAGSystem is not None
    assert RAGResponse is not None
    assert Path(RAGConfig().data_path) == PROJECT_ROOT / "data" / "campus"
    assert Path(RAGConfig().index_save_path) == PROJECT_ROOT / "vector_index"

