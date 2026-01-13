"""测试 CLIP 类型检测和 Gemma 提取

PR #2 的核心功能: 支持 ComfyUI DualCLIPLoader，
实现 All-in-One 模式 (无需额外下载模型)。
"""

import pytest
import torch
from unittest.mock import MagicMock

from comfy_newbie_gemma_chat import NewBieGemmaChat


class TestCLIPDetection:
    """测试 _detect_clip_type() 方法"""

    def test_detect_newbie_clip_valid(self, mock_newbie_clip):
        """NewBie CLIP 应该被正确检测"""
        chat = NewBieGemmaChat()
        clip_type = chat._detect_clip_type(mock_newbie_clip)
        assert clip_type == "newbie_clip"

    def test_detect_comfyui_dual_clip_valid(self, mock_comfyui_dual_clip):
        """ComfyUI DualCLIPLoader 应该被正确检测"""
        chat = NewBieGemmaChat()
        clip_type = chat._detect_clip_type(mock_comfyui_dual_clip)
        assert clip_type == "comfyui_dual_clip"

    def test_detect_unknown_clip_missing_all_attributes(self):
        """缺少所有必需属性应返回 unknown"""
        chat = NewBieGemmaChat()
        clip = MagicMock()
        clip_type = chat._detect_clip_type(clip)
        assert clip_type == "unknown"

    def test_detect_newbie_clip_missing_text_encoder(self):
        """NewBie CLIP 缺少 text_encoder 应返回 unknown"""
        chat = NewBieGemmaChat()
        clip = MagicMock()
        clip.tokenizer = MagicMock()
        clip.tokenizer.name_or_path = "google/gemma-3-4b"
        # 缺少 text_encoder

        clip_type = chat._detect_clip_type(clip)
        assert clip_type == "unknown"

    def test_detect_comfyui_dual_clip_missing_gemma(self):
        """ComfyUI DualCLIPLoader 缺少 gemma 应返回 unknown"""
        chat = NewBieGemmaChat()
        clip = MagicMock()
        clip.cond_stage_model = MagicMock(spec=['other_attr'])
        clip.tokenizer = MagicMock()
        # cond_stage_model 存在但没有 gemma 属性

        clip_type = chat._detect_clip_type(clip)
        assert clip_type == "unknown"


class TestGemmaExtraction:
    """测试 _extract_gemma_from_clip() 方法"""

    def test_extract_gemma_valid_structure(self, mock_comfyui_dual_clip):
        """正确的结构应成功提取所有组件"""
        chat = NewBieGemmaChat()
        transformer, tokenizer, embed_weight, device, dtype = \
            chat._extract_gemma_from_clip(mock_comfyui_dual_clip)

        assert transformer is not None
        assert tokenizer is not None
        assert isinstance(embed_weight, torch.Tensor)
        assert device == torch.device('cpu')
        assert dtype == torch.bfloat16

    def test_extract_gemma_embed_weight_shape(self, mock_comfyui_dual_clip):
        """embed_weight shape 应正确"""
        chat = NewBieGemmaChat()
        _, _, embed_weight, _, _ = chat._extract_gemma_from_clip(mock_comfyui_dual_clip)

        # 默认: vocab_size=256000, hidden_size=2304
        assert embed_weight.shape[0] == 256000  # vocab_size
        assert embed_weight.shape[1] == 2304    # hidden_size
        assert embed_weight.dim() == 2

    def test_extract_gemma_device_dtype(self, dtype):
        """device 和 dtype 应正确提取"""
        chat = NewBieGemmaChat()
        from tests.fixtures.mock_clip import MockCLIPFactory

        # 创建自定义 device/dtype 的 CLIP
        clip = MockCLIPFactory.create_comfyui_dual_clip(device='cpu', dtype=torch.float32)
        _, _, _, device, extracted_dtype = chat._extract_gemma_from_clip(clip)

        assert device == torch.device('cpu')
        assert extracted_dtype == torch.float32

    def test_extract_gemma_missing_cond_stage_model(self):
        """缺少 cond_stage_model 应抛出 ValueError"""
        chat = NewBieGemmaChat()
        clip = MagicMock(spec=['other_attr'])

        with pytest.raises(ValueError, match="不包含 Gemma 模型"):
            chat._extract_gemma_from_clip(clip)

    def test_extract_gemma_missing_transformer(self, mock_incomplete_clip):
        """缺少 transformer 应抛出 ValueError"""
        chat = NewBieGemmaChat()
        clip = mock_incomplete_clip(missing_attr='transformer')

        with pytest.raises((ValueError, AttributeError)):
            chat._extract_gemma_from_clip(clip)

    def test_extract_gemma_missing_embed_tokens(self, mock_incomplete_clip):
        """缺少 embed_tokens 应抛出 ValueError"""
        chat = NewBieGemmaChat()
        clip = mock_incomplete_clip(missing_attr='embed_tokens')

        with pytest.raises((ValueError, AttributeError)):
            chat._extract_gemma_from_clip(clip)

    def test_extract_gemma_tokenizer_unwrapping(self, mock_comfyui_dual_clip):
        """应正确解包 SPieceTokenizer"""
        chat = NewBieGemmaChat()
        _, tokenizer, _, _, _ = chat._extract_gemma_from_clip(mock_comfyui_dual_clip)

        # tokenizer 应被正确解包
        assert tokenizer is not None
        # 检查是否有 tokenizer.tokenizer 属性 (嵌套结构)
        assert hasattr(tokenizer, 'tokenizer')
