"""测试 EOS token 检测逻辑

PR #2 的关键修复: 移除 token 107 (<end_of_turn>) 从 EOS_TOKENS，
防止过早停止生成 (如生成 "```xml" 时)。
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from comfy_newbie_gemma_chat import NewBieGemmaChat


class TestEOSDetection:
    """测试 EOS token 常量和检测逻辑"""

    def test_eos_tokens_constant_value(self):
        """EOS_TOKENS 应只包含 token 1"""
        chat = NewBieGemmaChat()
        assert chat.EOS_TOKENS == frozenset({1})
        assert isinstance(chat.EOS_TOKENS, frozenset)

    def test_eos_tokens_excludes_107(self):
        """PR #2 关键修复: token 107 不应在 EOS_TOKENS 中

        token 107 (<end_of_turn>) 在某些情况下可能是其他含义，
        导致过早停止 (如 "```xml" 的 3 个 token)。
        """
        chat = NewBieGemmaChat()
        assert 107 not in chat.EOS_TOKENS, "token 107 不应在 EOS_TOKENS 中"
        assert 1 in chat.EOS_TOKENS, "token 1 应在 EOS_TOKENS 中"

    def test_extended_eos_tokens_includes_both(self):
        """EXTENDED_EOS_TOKENS 应包含 1 和 107 (用于其他用途)"""
        chat = NewBieGemmaChat()
        assert chat.EXTENDED_EOS_TOKENS == frozenset({1, 107})
        assert 1 in chat.EXTENDED_EOS_TOKENS
        assert 107 in chat.EXTENDED_EOS_TOKENS

    def test_token_constants_defined(self):
        """所有 token 常量应正确定义"""
        chat = NewBieGemmaChat()
        assert chat.EOS_TOKEN == 1
        assert chat.END_OF_TURN_TOKEN == 107
        assert chat.PAD_TOKEN == 0
        assert chat.BOS_TOKEN == 2

    def test_generation_stops_at_token_1(self, mock_gemma_model, mock_comfy_rope):
        """生成应在遇到 token 1 时停止"""
        chat = NewBieGemmaChat()
        input_ids = torch.tensor([[2, 100]])  # BOS + 1 token
        embed_weight = torch.randn(256000, 1024)

        # Mock _sample_next_token 返回 EOS (token 1)
        with patch.object(chat, '_sample_next_token', return_value=torch.tensor([[1]])):
            generated = chat._generate_with_kv_cache(
                mock_gemma_model, embed_weight, input_ids,
                max_new_tokens=10, temperature=0.7, top_p=0.9, top_k=50
            )

        # 应只生成一个 token 并停止
        assert len(generated) == 1
        assert generated[0] == 1

    def test_generation_continues_after_token_107(self, mock_gemma_model, mock_comfy_rope):
        """PR #2 修复验证: 生成应在遇到 token 107 后继续"""
        chat = NewBieGemmaChat()
        input_ids = torch.tensor([[2, 100]])  # BOS + 1 token
        embed_weight = torch.randn(256000, 1024)

        # Mock _sample_next_token: 先返回 107,  再返回其他 token, 最后返回 EOS
        mock_tokens = [
            torch.tensor([[107]]),  # end_of_turn - 不应停止
            torch.tensor([[200]]),  # 继续生成
            torch.tensor([[1]])     # EOS - 应停止
        ]

        with patch.object(chat, '_sample_next_token', side_effect=mock_tokens):
            generated = chat._generate_with_kv_cache(
                mock_gemma_model, embed_weight, input_ids,
                max_new_tokens=10, temperature=0.7, top_p=0.9, top_k=50
            )

        # 应生成 3 个 token: 107, 200, 1
        assert len(generated) == 3
        assert generated[0] == 107  # 包含 end_of_turn
        assert generated[1] == 200  # 继续生成
        assert generated[2] == 1    # 最后是 EOS
