"""Knowledge base system for learning and pattern storage."""

from .base import KnowledgeBase
from .storage import FileSystemStorage

__all__ = ["KnowledgeBase", "FileSystemStorage"]
