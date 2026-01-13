"""Gemma 模型 Mock 工厂

提供 Gemma transformer 层和完整模型的 mock 对象。
使用真实的 PyTorch nn.Module (Linear, LayerNorm) 确保数学正确性。
"""

from unittest.mock import MagicMock
import torch
import torch.nn as nn


class MockGemmaModelFactory:
    """Gemma 模型结构 mock 工厂"""

    @staticmethod
    def create_transformer_layer(num_heads=8, num_kv_heads=4, head_dim=128,
                                  device='cpu', dtype=torch.float32):
        """创建单个 Transformer 层的 mock

        Args:
            num_heads: 查询头数 (Q heads)
            num_kv_heads: KV 头数 (用于 GQA)
            head_dim: 每个头的维度
            device: 设备 ('cpu' 或 'cuda')
            dtype: 数据类型

        Returns:
            mock layer 对象
        """
        layer = MagicMock()
        hidden_size = num_heads * head_dim

        # 注意力层
        attn = MagicMock()
        attn.num_heads = num_heads
        attn.num_kv_heads = num_kv_heads
        attn.head_dim = head_dim

        # Q/K/V 投影层 (使用真实 Linear 以测试真实 tensor 操作)
        attn.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False).to(device=device, dtype=dtype)
        attn.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False).to(device=device, dtype=dtype)
        attn.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False).to(device=device, dtype=dtype)
        attn.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False).to(device=device, dtype=dtype)

        # Q/K 归一化 (Gemma3 特有)
        attn.q_norm = nn.LayerNorm(head_dim).to(device=device, dtype=dtype)
        attn.k_norm = nn.LayerNorm(head_dim).to(device=device, dtype=dtype)

        layer.self_attn = attn

        # LayerNorms
        layer.input_layernorm = nn.LayerNorm(hidden_size).to(device=device, dtype=dtype)
        layer.post_attention_layernorm = nn.LayerNorm(hidden_size).to(device=device, dtype=dtype)
        layer.pre_feedforward_layernorm = nn.LayerNorm(hidden_size).to(device=device, dtype=dtype)
        layer.post_feedforward_layernorm = nn.LayerNorm(hidden_size).to(device=device, dtype=dtype)

        # MLP (简化版 - 只有一个 Linear)
        layer.mlp = nn.Linear(hidden_size, hidden_size).to(device=device, dtype=dtype)

        # 滑动注意力标志 (用于 Gemma3 双 RoPE 测试)
        layer.sliding_attention = False

        # parameters() 方法用于迭代所有参数
        def get_all_params():
            params = []
            for module in [attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj,
                          attn.q_norm, attn.k_norm,
                          layer.input_layernorm, layer.post_attention_layernorm,
                          layer.pre_feedforward_layernorm, layer.post_feedforward_layernorm,
                          layer.mlp]:
                params.extend(list(module.parameters()))
            return iter(params)

        layer.parameters = get_all_params

        return layer

    @staticmethod
    def create_model(num_layers=2, num_heads=8, num_kv_heads=4, head_dim=128,
                    vocab_size=256000, device='cpu', dtype=torch.float32):
        """创建完整模型 mock

        Args:
            num_layers: Transformer 层数
            num_heads: 查询头数
            num_kv_heads: KV 头数
            head_dim: 每个头的维度
            vocab_size: 词表大小
            device: 设备
            dtype: 数据类型

        Returns:
            mock model 对象
        """
        model = MagicMock()
        hidden_size = num_heads * head_dim

        # Config
        model.config = MagicMock()
        model.config.hidden_size = hidden_size
        model.config.head_dim = head_dim
        model.config.rope_theta = 10000.0
        model.config.rope_scale = 1.0
        model.config.rope_dims = head_dim

        # Embedding
        model.embed_tokens = nn.Embedding(vocab_size, hidden_size).to(device=device, dtype=dtype)
        model.normalize_in = True

        # Layers
        model.layers = [
            MockGemmaModelFactory.create_transformer_layer(
                num_heads, num_kv_heads, head_dim, device, dtype
            )
            for _ in range(num_layers)
        ]

        # Output norm
        model.norm = nn.LayerNorm(hidden_size).to(device=device, dtype=dtype)

        # Parameters method
        def get_all_model_params():
            params = list(model.embed_tokens.parameters())
            for layer in model.layers:
                params.extend(list(layer.parameters()))
            params.extend(list(model.norm.parameters()))
            return iter(params)

        model.parameters = get_all_model_params

        return model

    @staticmethod
    def create_simple_tokenizer():
        """创建简化的 tokenizer mock

        支持 encode 和 decode 操作。
        """
        tokenizer = MagicMock()

        def encode(text, **kwargs):
            # 简化: 返回固定的 token ids
            return [2, 100, 200, 1]  # BOS + tokens + EOS

        def decode(token_ids, **kwargs):
            # 简化: 返回固定的文本
            return "Generated response"

        tokenizer.encode = MagicMock(side_effect=encode)
        tokenizer.decode = MagicMock(side_effect=decode)
        tokenizer.tokenizer = tokenizer  # 支持嵌套访问

        return tokenizer
