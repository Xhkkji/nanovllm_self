import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from nanovllm.engine.Sequence import Sequence
from nanovllm.engine.block_manager import block_manager as BlockManager
from nanovllm.models.qwen3 import Qwen3Model

MODEL_PATH = '/home/xhk/model/Qwen3-0.6B/'
DEVICE = 'cuda:0'
CHINESE_PROMPT = '请用自然、生动、带有画面感的语言，介绍中国的ACG文化，并简要对比日本ACG文化。'

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
config = AutoConfig.from_pretrained(MODEL_PATH)
hf = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE).eval()
self_model = Qwen3Model(config).to(DEVICE).eval()
full_ids = tokenizer(CHINESE_PROMPT, return_tensors='pt')['input_ids'].to(DEVICE)


def hf_prefill_topk(input_ids, k=10):
    with torch.no_grad():
        out = hf(input_ids, use_cache=True)
        logits = out.logits[:, -1, :]
        vals, idx = torch.topk(logits, k=k, dim=-1)
    return idx[0].tolist(), vals[0].tolist()


def self_prefill_topk(input_ids, block_size=16, k=10):
    bm = BlockManager(num_blocks=100, block_size=block_size, num_layers=self_model.num_layers, num_kv_heads=self_model.num_kv_heads, head_dim=self_model.head_dim)
    seq = Sequence(seq_idx=0, token_ids=input_ids[0].tolist())
    seq.block_size = block_size
    seq.block_table = bm.allocate_with_prefill(seq)
    positions = torch.arange(0, len(seq.token_ids), device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        out = self_model(input_ids[0], positions=positions, block_manager=bm, seq=seq, is_prefill=True)
        logits = out[-1, :].unsqueeze(0)
        vals, idx = torch.topk(logits, k=k, dim=-1)
    return idx[0].tolist(), vals[0].tolist(), bm, seq


def hf_greedy(input_ids, steps):
    tokens = input_ids.clone()
    past = None
    generated = []
    with torch.no_grad():
        for step in range(steps):
            current = tokens if past is None else tokens[:, -1:]
            out = hf(current, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(int(next_token.item()))
            tokens = torch.cat([tokens, next_token], dim=1)
    return generated


def self_greedy(input_ids, steps, block_size=16):
    bm = BlockManager(num_blocks=100, block_size=block_size, num_layers=self_model.num_layers, num_kv_heads=self_model.num_kv_heads, head_dim=self_model.head_dim)
    seq = Sequence(seq_idx=0, token_ids=input_ids[0].tolist())
    seq.block_size = block_size
    seq.block_table = bm.allocate_with_prefill(seq)
    tokens = input_ids.clone()
    generated = []
    is_prefill = True
    with torch.no_grad():
        for step in range(steps):
            if is_prefill:
                current = tokens[0]
                positions = torch.arange(0, len(seq.token_ids), device=DEVICE).unsqueeze(0)
                out = self_model(current, positions=positions, block_manager=bm, seq=seq, is_prefill=True)
                logits = out[-1, :].unsqueeze(0)
                is_prefill = False
            else:
                current = tokens[0, -1:]
                positions = torch.tensor([[len(seq.token_ids) - 1]], device=DEVICE)
                out = self_model(current, positions=positions, block_manager=bm, seq=seq, is_prefill=False)
                logits = out.unsqueeze(0)
            next_token = logits.argmax(dim=-1, keepdim=True)
            token_id = int(next_token[0,0].item())
            generated.append(token_id)
            tokens = torch.cat([tokens, next_token], dim=1)
            seq.append_token(token_id)
            if len(seq.token_ids) > len(seq.block_table) * bm.block_size:
                seq.block_table.append(bm.allocate_block(1)[0])
    return generated

for prompt_len in [15, 16, 17, 25]:
    sliced = full_ids[:, :prompt_len]
    print(f'=== prompt_len={prompt_len} ===')
    hf_topk, hf_vals = hf_prefill_topk(sliced)
    self_topk, self_vals, _, _ = self_prefill_topk(sliced)
    print('hf_prefill_top1', hf_topk[0], 'self_prefill_top1', self_topk[0])
    print('hf_prefill_top10', hf_topk)
    print('self_prefill_top10', self_topk)
    hg = hf_greedy(sliced, 8)
    sg = self_greedy(sliced, 8)
    print('hf_greedy_8', hg)
    print('self_greedy_8', sg)
    for i, (a,b) in enumerate(zip(hg, sg)):
        if a != b:
            print('first_diverge', i, a, b)
            break
    else:
        print('match_all_8')
