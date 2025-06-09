import os

from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig

from .openai_gpt import CacheOpenAI
from .base import BaseLLM

logger = get_logger(__name__)


from .vllm_offline import VLLMOffline
from .openai_gpt import CacheOpenAI

def _get_llm_class(config: BaseConfig):
    if config.llm_base_url and "localhost" in config.llm_base_url:
        return VLLMOffline(global_config=config)
    return CacheOpenAI.from_experiment_config(config)