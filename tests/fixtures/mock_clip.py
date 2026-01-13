"""CLIP 对象 Mock 工厂

提供 NewBie CLIP Loader 和 ComfyUI DualCLIPLoader 的 mock 对象。
"""

from unittest.mock import MagicMock
import torch


class MockCLIPFactory:
    """CLIP 对象 mock 工厂"""

    @staticmethod
    def create_newbie_clip(device='cpu', dtype=torch.bfloat16):
        """创建 NewBie CLIP Loader 的 mock

        属性结构:
        - text_encoder: 文本编码器
        - tokenizer: 分词器
        - tokenizer.name_or_path: 模型路径
        """
        clip = MagicMock()
        clip.text_encoder = MagicMock()
        clip.tokenizer = MagicMock()
        clip.tokenizer.name_or_path = "google/gemma-3-4b"

        # 设置参数以便 next(parameters()) 工作
        param = torch.zeros(1, dtype=dtype, device=torch.device(device))
        clip.text_encoder.parameters = MagicMock(return_value=iter([param]))

        return clip

    @staticmethod
    def create_comfyui_dual_clip(device='cpu', dtype=torch.bfloat16,
                                  vocab_size=256000, hidden_size=2304):
        """创建 ComfyUI DualCLIPLoader 的 mock

        属性结构:
        - clip.cond_stage_model.gemma: Gemma 模型
        - clip.cond_stage_model.gemma.transformer: transformer
        - clip.cond_stage_model.gemma.transformer.model: 核心模型
        - clip.cond_stage_model.gemma.transformer.model.embed_tokens.weight: embedding 权重
        - clip.tokenizer.gemma: tokenizer
        """
        clip = MagicMock()

        # 构建嵌套的 Gemma 结构
        gemma = MagicMock()
        transformer = MagicMock()
        model_core = MagicMock()

        # embed_tokens.weight (用于 weight tying)
        model_core.embed_tokens = MagicMock()
        model_core.embed_tokens.weight = torch.randn(
            vocab_size, hidden_size, dtype=dtype, device=torch.device(device)
        )

        # 组装层次结构
        transformer.model = model_core
        gemma.transformer = transformer
        clip.cond_stage_model = MagicMock()
        clip.cond_stage_model.gemma = gemma

        # Mock tokenizer (带 SPieceTokenizer 包装)
        spiece_tokenizer = MagicMock()
        spiece_tokenizer.tokenizer = MagicMock()
        clip.tokenizer = MagicMock()
        clip.tokenizer.gemma = spiece_tokenizer

        # 设置参数 (用于 device 和 dtype 检测)
        param = torch.zeros(1, dtype=dtype, device=torch.device(device))
        model_core.parameters = MagicMock(return_value=iter([param]))

        return clip

    @staticmethod
    def create_incomplete_clip(missing_attr='cond_stage_model'):
        """创建不完整的 CLIP mock (用于测试错误处理)

        Args:
            missing_attr: 要缺失的属性名称
        """
        clip = MagicMock()

        if missing_attr == 'cond_stage_model':
            # 缺少 cond_stage_model
            pass
        elif missing_attr == 'gemma':
            # 有 cond_stage_model 但缺少 gemma
            clip.cond_stage_model = MagicMock(spec=['other_attr'])
        elif missing_attr == 'transformer':
            # 有 gemma 但缺少 transformer
            clip.cond_stage_model = MagicMock()
            clip.cond_stage_model.gemma = MagicMock(spec=['other_attr'])
        elif missing_attr == 'embed_tokens':
            # 有 transformer 但缺少 embed_tokens
            clip.cond_stage_model = MagicMock()
            clip.cond_stage_model.gemma = MagicMock()
            clip.cond_stage_model.gemma.transformer = MagicMock()
            clip.cond_stage_model.gemma.transformer.model = MagicMock(spec=['other_attr'])

        return clip
