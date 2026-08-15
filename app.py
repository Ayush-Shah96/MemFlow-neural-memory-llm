from __future__ import annotations

import logging
import sys

from ui.app_ui import build_app


# Configure application logging once, so ingestion and LLM calls are
# visible in the VS Code terminal as well as the Gradio UI.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

logger = logging.getLogger("neural-memory")


def main():
    logger.info("Starting Neural Memory LLM application")
    logger.info("Python logging level: INFO")
    app = build_app()
    app.launch()


if __name__ == "__main__":
    main()