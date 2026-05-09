import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from nanovllm.engine.Sequence import Sequence
from nanovllm.engine.block_manager import block_manager as BlockManager
from nanovllm.models.qwen3 import Qwen3Model

MODEL_PATH = '/home/xhk/model/Qwen3-0.6B/'
DEVICE = 'cuda:0'
PROMPT = '请用自然、生动、带有画面感的语言，介绍中国的ACG文化，并简要对比日本ACG文化。'

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
config = AutoConfig.from_pretrained(MODEL_PATH)
hf = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE).eval()
self_model = Qwen3Model(config).to(DEVICE).eval()
input_ids = tok(PROMPT, return_tensors='pt')['input_ids'].to(DEVICE)[:, :25]

# HF
hf_tokens = input_ids.clone()
hf_past = None
hf_hist = []
with torch.no_grad():
    for step in range(8):
        current = hf_tokens if hf_past is None else hf_tokens[:, -1:]
        out = hf(current, past_key_values=hf_past, use_cache=True)
        hf_past = out.past_key_values
        logits = out.logits[:, -1, :]
        vals, idx = torch.topk(logits, k=5, dim=-1)
        next_token = logits.argmax(dim=-1, keepdim=True)
        hf_hist.append((int(next_token.item()), idx[0].tolist(), [float(v) for v in vals[0]]))
        hf_tokens = torch.cat([hf_tokens, next_token], dim=1)

# SELF
bm = BlockManager(num_blocks=100, block_size=16, num_layers=self_model.num_layers, num_kv_heads=self_model.num_kv_heads, head_dim=self_model.head_dim)
seq = Sequence(seq_idx=0, token_ids=input_ids[0].tolist())
seq.block_size = 16
seq.block_table = bm.allocate_with_prefill(seq)
self_tokens = input_ids.clone()
self_hist = []
is_prefill = True
with torch.no_grad():
    for step in range(8):
        if is_prefill:
            current = self_tokens[0]
            positions = torch.arange(0, len(seq.token_ids), device=DEVICE).unsqueeze(0)
            out = self_model(current, positions=positions, block_manager=bm, seq=seq, is_prefill=True)
            logits = out[-1, :].unsqueeze(0)
            is_prefill = False
        else:
            current = self_tokens[0, -1:]
            positions = torch.tensor([[len(seq.token_ids) - 1]], device=DEVICE)
            out = self_model(current, positions=positions, block_manager=bm, seq=seq, is_prefill=False)
            logits = out.unsqueeze(0)
        vals, idx = torch.topk(logits, k=5, dim=-1)
        next_token = logits.argmax(dim=-1, keepdim=True)
        token_id = int(next_token[0,0].item())
        self_hist.append((token_id, idx[0].tolist(), [float(v) for v in vals[0]]))
        self_tokens = torch.cat([self_tokens, next_token], dim=1)
        seq.append_token(token_id)
        if len(seq.token_ids) > len(seq.block_table) * bm.block_size:
            seq.block_table.append(bm.allocate_block(1)[0])

for step, (h, s) in enumerate(zip(hf_hist, self_hist)):
    print('step', step)
    print(' hf next', h[0], 'top5', h[1], 'vals', h[2])
    print('self next', s[0], 'top5', s[1], 'vals', s[2])
    print()
