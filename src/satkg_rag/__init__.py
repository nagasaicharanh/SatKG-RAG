"""SatKG-RAG package."""

__all__ = ["SatKGRAGPipeline"]


def __getattr__(name: str):
    if name == "SatKGRAGPipeline":
        from .pipeline import SatKGRAGPipeline

        return SatKGRAGPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
