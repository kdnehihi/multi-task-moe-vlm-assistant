"""Start the FastAPI routed VLM QA service."""

from __future__ import annotations

import uvicorn


def main() -> None:
    """Run the API server."""
    uvicorn.run(
        "src.serving.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
