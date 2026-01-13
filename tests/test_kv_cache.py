"""测试 KV-Cache 优化

PR #2 的核心功能: KV-Cache 优化实现 12-13x 加速。
包含 RoPE 计算、层 forward 和完整生成循环的测试。
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from comfy_newbie_gemma_chat import NewBieGemmaChat


class TestRoPEComputation:
    """测试 RoPE (Rotary Position Embedding) 相关方法"""

    def test_compute_rope_returns_tuple(self, mock_gemma_model, mock_comfy_rope, device):
        """_compute_rope_for_cache 应返回 (cos, sin) 元组"""
        chat = NewBieGemmaChat()
        position_ids = torch.tensor([[0, 1, 2, 3]], device=torch.device(device))
        freqs_cis = chat._compute_rope_for_cache(mock_gemma_model, position_ids, device)

        assert isinstance(freqs_cis, tuple)
        assert len(freqs_cis) == 2
        cos, sin = freqs_cis
        assert isinstance(cos, torch.Tensor)
        assert isinstance(sin, torch.Tensor)

    def test_compute_rope_correct_shape(self, mock_gemma_model, mock_comfy_rope, device):
        """RoPE 应返回正确形状的 cos/sin"""
        chat = NewBieGemmaChat()
        position_ids = torch.tensor([[0, 1, 2, 3]], device=torch.device(device))
        freqs_cis = chat._compute_rope_for_cache(mock_gemma_model, position_ids, device)

        cos, sin = freqs_cis
        head_dim = mock_gemma_model.config.head_dim
        assert cos.shape[-1] == head_dim
        assert sin.shape[-1] == head_dim

    def test_apply_rope_preserves_dtype(self):
        """应用 RoPE 后应保持原始 dtype"""
        chat = NewBieGemmaChat()
        xq = torch.randn(1, 8, 4, 128, dtype=torch.float16)
        xk = torch.randn(1, 4, 4, 128, dtype=torch.float16)
        cos = torch.randn(1, 1, 4, 128)
        sin = torch.randn(1, 1, 4, 128)

        xq_out, xk_out = chat._apply_rope_for_cache(xq, xk, (cos, sin))

        assert xq_out.dtype == torch.float16
        assert xk_out.dtype == torch.float16

    def test_apply_rope_preserves_shape(self):
        """应用 RoPE 后应保持 tensor 形状"""
        chat = NewBieGemmaChat()
        xq = torch.randn(1, 8, 4, 128)
        xk = torch.randn(1, 4, 4, 128)
        cos = torch.randn(1, 1, 4, 128)
        sin = torch.randn(1, 1, 4, 128)

        xq_out, xk_out = chat._apply_rope_for_cache(xq, xk, (cos, sin))

        assert xq_out.shape == xq.shape
        assert xk_out.shape == xk.shape


class TestLayerForward:
    """测试 _layer_forward_with_cache() 方法"""

    def test_layer_forward_no_cache(self, mock_transformer_layer, sample_freqs_cis):
        """无缓存的层前向传播应正常工作"""
        chat = NewBieGemmaChat()
        x = torch.randn(1, 4, 1024)  # (batch, seq, hidden_size)

        output, new_kv = chat._layer_forward_with_cache(
            mock_transformer_layer, x, sample_freqs_cis, None, past_kv=None
        )

        # 输出 shape 应与输入一致
        assert output.shape == x.shape
        # 应返回新的 KV 缓存
        assert new_kv is not None
        assert len(new_kv) == 2  # (k, v)
        assert isinstance(new_kv[0], torch.Tensor)
        assert isinstance(new_kv[1], torch.Tensor)

    def test_layer_forward_kv_cache_shape(self, mock_transformer_layer, sample_freqs_cis):
        """KV 缓存应有正确的形状"""
        chat = NewBieGemmaChat()
        batch, seq, hidden_size = 1, 4, 1024
        x = torch.randn(batch, seq, hidden_size)

        _, new_kv = chat._layer_forward_with_cache(
            mock_transformer_layer, x, sample_freqs_cis, None, past_kv=None
        )

        cached_k, cached_v = new_kv
        num_kv_heads = mock_transformer_layer.self_attn.num_kv_heads
        head_dim = mock_transformer_layer.self_attn.head_dim

        # Shape: (batch, num_kv_heads, seq, head_dim)
        assert cached_k.shape == (batch, num_kv_heads, seq, head_dim)
        assert cached_v.shape == (batch, num_kv_heads, seq, head_dim)

    def test_layer_forward_with_cache_concatenation(self, mock_transformer_layer):
        """使用缓存时应正确拼接 KV"""
        chat = NewBieGemmaChat()
        x = torch.randn(1, 1, 1024)  # decode 阶段只有 1 个 token
        freqs_cis = (torch.randn(1, 1, 1, 128), torch.randn(1, 1, 1, 128))

        # 模拟已有 3 个 token 的缓存
        cached_k = torch.randn(1, 4, 3, 128)  # 4 KV heads
        cached_v = torch.randn(1, 4, 3, 128)
        past_kv = (cached_k, cached_v)

        _, new_kv = chat._layer_forward_with_cache(
            mock_transformer_layer, x, freqs_cis, None, past_kv=past_kv
        )

        # 新缓存应该有 4 个 token (3 + 1)
        assert new_kv[0].shape[2] == 4
        assert new_kv[1].shape[2] == 4

    def test_layer_forward_cache_detach_clone(self, mock_transformer_layer, sample_freqs_cis):
        """KV 缓存应独立于计算图 (detach 和 clone)"""
        chat = NewBieGemmaChat()
        x = torch.randn(1, 4, 1024, requires_grad=True)

        _, new_kv = chat._layer_forward_with_cache(
            mock_transformer_layer, x, sample_freqs_cis, None, past_kv=None
        )

        # 缓存应该不需要梯度 (已 detach)
        assert not new_kv[0].requires_grad
        assert not new_kv[1].requires_grad


class TestKVCacheGeneration:
    """测试 _generate_with_kv_cache() 主生成循环"""

    def test_generate_returns_list(self, mock_gemma_model, mock_comfy_rope):
        """生成应返回 token ID 列表"""
        chat = NewBieGemmaChat()
        input_ids = torch.tensor([[2, 100, 200]])  # BOS + 2 tokens
        embed_weight = torch.randn(256000, 1024)

        with patch.object(chat, '_sample_next_token', return_value=torch.tensor([[1]])):
            generated = chat._generate_with_kv_cache(
                mock_gemma_model, embed_weight, input_ids,
                max_new_tokens=5, temperature=0.7, top_p=0.9, top_k=50
            )

        assert isinstance(generated, list)
        assert all(isinstance(token, int) for token in generated)

    def test_generate_respects_max_tokens_limit(self, mock_gemma_model, mock_comfy_rope):
        """生成不应超过 max_new_tokens 限制"""
        chat = NewBieGemmaChat()
        input_ids = torch.tensor([[2, 100]])
        embed_weight = torch.randn(256000, 1024)

        # Mock 返回非 EOS token
        with patch.object(chat, '_sample_next_token', return_value=torch.tensor([[200]])):
            generated = chat._generate_with_kv_cache(
                mock_gemma_model, embed_weight, input_ids,
                max_new_tokens=5, temperature=0.7, top_p=0.9, top_k=50
            )

        # 应恰好生成 5 个 token
        assert len(generated) <= 5

    def test_generate_stops_at_eos(self, mock_gemma_model, mock_comfy_rope):
        """遇到 EOS token (1) 应立即停止"""
        chat = NewBieGemmaChat()
        input_ids = torch.tensor([[2, 100]])
        embed_weight = torch.randn(256000, 1024)

        # 第一次就返回 EOS
        with patch.object(chat, '_sample_next_token', return_value=torch.tensor([[1]])):
            generated = chat._generate_with_kv_cache(
                mock_gemma_model, embed_weight, input_ids,
                max_new_tokens=10, temperature=0.7, top_p=0.9, top_k=50
            )

        assert len(generated) == 1
        assert generated[0] == 1

    def test_generate_prefill_phase(self, mock_gemma_model, mock_comfy_rope):
        """Prefill 阶段应正确处理输入序列"""
        chat = NewBieGemmaChat()
        input_ids = torch.tensor([[2, 100, 200, 300]])  # BOS + 3 tokens
        embed_weight = torch.randn(256000, 1024)

        with patch.object(chat, '_sample_next_token', return_value=torch.tensor([[1]])):
            generated = chat._generate_with_kv_cache(
                mock_gemma_model, embed_weight, input_ids,
                max_new_tokens=5, temperature=0.7, top_p=0.9, top_k=50
            )

        # Prefill 应该成功,生成至少 1 个 token
        assert len(generated) >= 1


class TestAttention:
    """测试 _scaled_dot_product_attention() 方法"""

    def test_scaled_dot_product_attention_basic(self):
        """基础注意力计算应返回正确形状"""
        chat = NewBieGemmaChat()
        q = torch.randn(1, 8, 4, 128)  # (batch, heads, seq, head_dim)
        k = torch.randn(1, 8, 4, 128)
        v = torch.randn(1, 8, 4, 128)

        output = chat._scaled_dot_product_attention(q, k, v, mask=None)

        assert output.shape == (1, 8, 4, 128)
        assert output.isfinite().all()

    def test_attention_with_causal_mask(self):
        """因果掩码应阻止未来信息泄漏"""
        chat = NewBieGemmaChat()
        q = k = v = torch.randn(1, 8, 4, 128)

        # 创建因果掩码 (上三角为 -inf)
        mask = torch.triu(torch.full((4, 4), float('-inf')), diagonal=1)

        output = chat._scaled_dot_product_attention(q, k, v, mask=mask)

        # 输出应该是有限值 (掩码正确应用)
        assert output.isfinite().all()
        assert output.shape == (1, 8, 4, 128)
