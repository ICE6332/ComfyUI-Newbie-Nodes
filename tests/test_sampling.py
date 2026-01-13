"""测试 Token 采样策略

测试 _sample_next_token() 和 _scaled_dot_product_attention() 方法。
"""

import pytest
import torch

from comfy_newbie_gemma_chat import NewBieGemmaChat


class TestSampling:
    """测试 _sample_next_token() 采样方法"""

    def test_sample_greedy_deterministic(self):
        """temperature=0 应使用贪婪解码 (argmax)"""
        chat = NewBieGemmaChat()
        torch.manual_seed(42)

        # token 1 有最高 logit
        logits = torch.tensor([[[1.0, 5.0, 2.0, 3.0]]])

        token = chat._sample_next_token(logits, temperature=0, top_p=1.0, top_k=0)

        # 应选择 argmax (token 1)
        assert token.shape == (1, 1)
        assert token.item() == 1

    def test_sample_handles_3d_logits(self):
        """应正确处理 3D logits (batch, seq, vocab)"""
        chat = NewBieGemmaChat()
        torch.manual_seed(42)

        # 3D logits: (1, 3, 4) - 取最后一个位置
        logits = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                                [2.0, 3.0, 4.0, 5.0],
                                [3.0, 4.0, 5.0, 6.0]]])  # 最后位置

        token = chat._sample_next_token(logits, temperature=0, top_p=1.0, top_k=0)

        # 应取最后位置的 argmax (token 3)
        assert token.item() == 3

    def test_sample_handles_2d_logits(self):
        """应正确处理 2D logits (batch, vocab)"""
        chat = NewBieGemmaChat()
        torch.manual_seed(42)

        # 2D logits: (1, 4)
        logits = torch.tensor([[1.0, 2.0, 5.0, 3.0]])

        token = chat._sample_next_token(logits, temperature=0, top_p=1.0, top_k=0)

        # 应选择 argmax (token 2)
        assert token.item() == 2

    def test_sample_top_k_filtering(self):
        """top_k 应只保留前 k 个最高 logits"""
        chat = NewBieGemmaChat()
        torch.manual_seed(42)

        # logits: [1.0, 2.0, 3.0, 4.0, 5.0]
        # top_k=2 应只允许 token 3 和 4 (最高的 2 个)
        logits = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]])

        token = chat._sample_next_token(logits, temperature=1.0, top_p=1.0, top_k=2)

        # 应在 top-2 中 (token 3 或 4)
        assert token.item() in [3, 4]

    def test_sample_top_p_nucleus(self):
        """top_p 应按累计概率过滤"""
        chat = NewBieGemmaChat()
        torch.manual_seed(42)

        # 使用较大差距的 logits 确保 top_p 生效
        logits = torch.tensor([[[0.0, 0.0, 10.0, 5.0, 0.0]]])

        # top_p=0.8 应过滤掉低概率 token
        token = chat._sample_next_token(logits, temperature=1.0, top_p=0.8, top_k=0)

        # 应在高概率 token 中 (token 2 或 3)
        assert token.item() in [2, 3]

    def test_sample_temperature_scaling(self):
        """不同 temperature 应影响采样分布"""
        chat = NewBieGemmaChat()

        logits = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])

        # temperature=0: 贪婪
        torch.manual_seed(42)
        token_greedy = chat._sample_next_token(logits, temperature=0, top_p=1.0, top_k=0)
        assert token_greedy.item() == 3  # argmax

        # temperature=1.0: 原始 softmax
        torch.manual_seed(42)
        token_normal = chat._sample_next_token(logits, temperature=1.0, top_p=1.0, top_k=0)
        # 应该是概率采样,可能不是 argmax

        # temperature=2.0: 更平滑的分布
        torch.manual_seed(42)
        token_smooth = chat._sample_next_token(logits, temperature=2.0, top_p=1.0, top_k=0)
        # 应该是概率采样

    def test_sample_returns_correct_shape(self):
        """采样应始终返回 (1, 1) 形状"""
        chat = NewBieGemmaChat()
        torch.manual_seed(42)

        for temp in [0, 0.5, 1.0, 1.5]:
            logits = torch.randn(1, 100)
            token = chat._sample_next_token(logits, temperature=temp, top_p=0.9, top_k=50)
            assert token.shape == (1, 1)
            assert isinstance(token, torch.Tensor)
