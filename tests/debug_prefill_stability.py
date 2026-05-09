import torch
from transformers import AutoConfig, AutoTokenizer
from nanovllm.engine.Sequence import Sequence
from nanovllm.engine.block_manager import block_manager as BlockManager
from nanovllm.models.qwen3 import Qwen3Model

MODEL_PATH = '/home/xhk/model/Qwen3-0.6B/'
DEVICE = 'cuda:0'
PROMPT = '请用自然、生动、带有画面感的语言，介绍中国的ACG文化，并简要对比日本ACG文化。'

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
config = AutoConfig.from_pretrained(MODEL_PATH)
model = Qwen3Model(config).to(DEVICE).eval()
input_ids = tok(PROMPT, return_tensors='pt')['input_ids'].to(DEVICE)[:, :15]

for run in range(3):
    bm = BlockManager(num_blocks=100, block_size=16, num_layers=model.num_layers, num_kv_heads=model.num_kv_heads, head_dim=model.head_dim)
    seq = Sequence(seq_idx=0, token_ids=input_ids[0].tolist())
    seq.block_size = 16
    seq.block_table = bm.allocate_with_prefill(seq)
    positions = torch.arange(0, len(seq.token_ids), device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        out = model(input_ids[0], positions=positions, block_manager=bm, seq=seq, is_prefill=True)
        logits = out[-1, :]
        vals, idx = torch.topk(logits, k=5, dim=-1)
    print('run', run, 'top5_ids', idx.tolist())
    print('run', run, 'top5_vals', [float(v) for v in vals])
    print('run', run, 'argmax', int(logits.argmax().item()))
