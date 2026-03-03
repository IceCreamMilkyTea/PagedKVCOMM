from KVCOMM.llm.llm_registry import LLMRegistry
from KVCOMM.llm.gpt_chat import GPTChat, LLMChat
from KVCOMM.llm.config import KVCommConfig
from KVCOMM.llm.visual_llm_registry import VisualLLMRegistry

# Conditional import: PagedLLMChat requires nano-vllm (nanovllm package).
# If nano-vllm is not available, paged attention backend is simply not registered.
try:
    from KVCOMM.llm.paged_llm_chat import PagedLLMChat
    _has_paged = True
except ImportError:
    PagedLLMChat = None  # type: ignore[assignment,misc]
    _has_paged = False

__all__ = ["LLMRegistry",
           "VisualLLMRegistry",
           "GPTChat",
           "LLMChat",
           "PagedLLMChat",
           "KVCommConfig"]
