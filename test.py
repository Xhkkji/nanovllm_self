import torch
from nanovllm.models.qwen3 import Qwen3Model
from nanovllm.engine.block_manager import block_manager as bm
from nanovllm.engine.Sequence import Sequence
from transformers import AutoConfig

def test():
    print("=" * 50)
    print("开始测试 Qwen3Model")
    print("=" * 50)

    config = AutoConfig.from_pretrained("/home/xhk/model/Qwen3-0.6B/")
    # 1. 配置参数（匹配 Qwen3-0.6B）
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = hidden_size // num_kv_heads
    vocab_size = config.vocab_size

    print(f"\n配置:")
    print(f"  层数: {num_layers}")
    print(f"  隐藏层大小: {hidden_size}")
    print(f"  注意力头数: {num_heads}")
    print(f"  头维度: {head_dim}")
    print(f"  词表大小: {vocab_size}")

    # 2. 创建模型
    print("\n创建模型...")
    model = Qwen3Model(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        vocab_size=vocab_size
    )
    model.cuda()
    model.eval()
    print("✅ 模型创建成功")

    # 3. 创建 BlockManager
    print("\n创建 BlockManager...")
    block_manager = bm(
        num_blocks=100,
        block_size=16,
        num_layers=num_layers,
        num_heads=num_kv_heads,
        head_dim=head_dim
    )
    print("✅ BlockManager 创建成功")

    # 4. 创建测试序列
    print("\n创建测试序列...")
    prompt_tokens = [1, 2, 3, 4, 5]
    seq = Sequence(seq_idx=0, token_ids=prompt_tokens)
    seq.block_size = 16  # 添加 block_size 属性

    # 分配块
    seq.block_table = block_manager.allocate_with_prefix(seq)
    print(f"  Prompt tokens: {prompt_tokens}")
    print(f"  Block table: {seq.block_table}")
    print(f"  使用块数: {len(seq.block_table)}")

    # 5. 测试前向传播
    print("\n测试前向传播...")
    token_tensor = torch.tensor([prompt_tokens[-1]], device='cuda')
    positions = torch.tensor([len(prompt_tokens) - 1], device='cuda')

    with torch.no_grad():
        logits = model.forward(token_tensor, positions, block_manager, seq)

    print(f"  Logits 形状: {logits.shape}")
    print(f"  Logits 示例: {logits[:5]}")
    print("✅ 前向传播成功")

    # 6. 测试生成
    print("\n测试生成（3个新token）...")
    current_tokens = prompt_tokens.copy()
    seq.token_ids = current_tokens

    for step in range(3):
        token_tensor = torch.tensor([current_tokens[-1]], device='cuda')
        positions = torch.tensor([len(current_tokens) - 1], device='cuda')

        with torch.no_grad():
            logits = model.forward(token_tensor, positions, block_manager, seq)

        next_token = torch.argmax(logits).item()
        current_tokens.append(next_token)
        seq.append_token(next_token)  # ✅ 使用 Sequence 的方法

        print(f"  Step {step + 1}: 生成 token {next_token}")

    print(f"\n最终序列: {current_tokens}")

    # 7. 清理
    print("\n清理资源...")
    block_manager.free_block(seq.block_table)
    print("✅ 清理完成")

    print("\n" + "=" * 50)
    print("✅ 所有测试通过！")
    print("=" * 50)


if __name__ == "__main__":
    # config = AutoConfig.from_pretrained("/home/xhk/model/Qwen3-0.6B/")
    # print(f"hidden_size: {config.hidden_size}")
    # print(f"num_heads: {config.num_attention_heads}")
    # print(f"num_layers: {config.num_hidden_layers}")
    # print(f"intermediate_size: {config.intermediate_size}")
    # from transformers import AutoModelForCausalLM
    # model = AutoModelForCausalLM.from_pretrained("/home/xhk/model/Qwen3-0.6B/")
    # layer = model.model.layers[0].self_attn
    #
    # print("可用属性:", [attr for attr in dir(layer) if 'proj' in attr])
    test()