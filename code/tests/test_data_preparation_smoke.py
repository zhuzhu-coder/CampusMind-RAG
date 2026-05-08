from config import PROJECT_ROOT


def test_campus_sample_corpus_exists():
    root = PROJECT_ROOT / "data" / "campus"

    assert root.exists()
    assert list(root.rglob("*.md"))
    assert list(root.rglob("*.txt"))
    assert list(root.rglob("*.pdf"))
