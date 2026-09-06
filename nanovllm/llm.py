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
    def __init__(self, model_path="/home/xhk/model/Qwen3-0.6B", tensor_parallel_size=1, enable_profile=False):
        # TP/大模型测试主线：把模型路径从外部入口透传到自研 runtime。
        # 这样测试脚本只需要设置 MODEL_PATH，就可以在 0.6B / 14B 等模型之间切换，
        # 避免在 engine / model_runner / model 内部到处手改硬编码路径。
        super().__init__(
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            enable_profile=enable_profile,
        )
