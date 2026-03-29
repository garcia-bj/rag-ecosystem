# Core model — always safe to import
from ingestion.parsers.base import ParsedChunk, BaseParser, Modality

# Heavy parsers use lazy imports internally; import here only when needed.
# To use a parser:  from ingestion.parsers.document import DocumentParser

__all__ = [
    "ParsedChunk",
    "BaseParser",
    "Modality",
]
