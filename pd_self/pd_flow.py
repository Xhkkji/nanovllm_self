from transformers import AutoTokenizer

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from nanovllm.config import Config
from .coordinator import PDCoordinator


def main():
    config = Config(
        model_path="/home/xhk/model/Qwen3-0.6B/",
        device="cuda:0",
        max_num_seqs=256,
        max_num_batched_tokens=16384,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        block_size=256,
        num_blocks=256,
    )

    pd = PDCoordinator(config)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    prompt = "What is a large language model?"
    output_token_ids = pd.generate(
        text=prompt,
        max_tokens=32,
        temperature=0.0,
        ignore_eos=True,
    )

    print(tokenizer.decode(output_token_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()