from .engine.llm_engine import llm_engine, llm_engine_self

class LLM(llm_engine):
    """
    将 llm_engine 重命名为 LLM，让外部使用时接口更简洁
    """
    pass

class LLM_self(llm_engine_self):
    """
    将 llm_engine_self 重命名为 LLM，让外部使用时接口更简洁
    """
    def __init__(self, tensor_parallel_size=1, enable_profile=False):
        super().__init__(
            tensor_parallel_size=tensor_parallel_size,
            enable_profile=enable_profile,
        )
