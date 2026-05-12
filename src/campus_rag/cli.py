"""Command-line entrypoint for interactive campus RAG Q&A."""

import logging

from .system import CampusRAGSystem

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the interactive Q&A client."""
    try:
        rag_system = CampusRAGSystem()
        rag_system.run_interactive()
    except Exception as exc:
        logger.error("系统运行出错: %s", exc)
        print(f"系统错误: {exc}")


if __name__ == "__main__":
    main()
