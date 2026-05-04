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
    pass