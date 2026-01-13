"""pytest 全局配置和 fixtures

提供所有测试共享的 fixtures，包括 CLIP mock、Gemma model mock 和测试数据。
"""

import pytest
import torch
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# 添加项目根目录到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.fixtures.mock_clip import MockCLIPFactory
from tests.fixtures.mock_gemma import MockGemmaModelFactory


# ==================== 设备和 dtype fixtures ====================

@pytest.fixture
def device():
    """默认测试设备 (CPU)"""
    return 'cpu'


@pytest.fixture
def dtype():
    """默认测试数据类型"""
    return torch.bfloat16


# ==================== CLIP fixtures ====================

@pytest.fixture
def mock_newbie_clip(device, dtype):
    """NewBie CLIP Loader mock"""
    return MockCLIPFactory.create_newbie_clip(device, dtype)


@pytest.fixture
def mock_comfyui_dual_clip(device, dtype):
    """ComfyUI DualCLIPLoader mock"""
    return MockCLIPFactory.create_comfyui_dual_clip(device, dtype)


@pytest.fixture
def mock_incomplete_clip():
    """不完整的 CLIP mock (用于测试错误处理)"""
    return MockCLIPFactory.create_incomplete_clip


# ==================== Gemma 模型 fixtures ====================

@pytest.fixture
def mock_gemma_model(device, dtype):
    """完整 Gemma 模型 mock"""
    return MockGemmaModelFactory.create_model(
        num_layers=2,
        num_heads=8,
        num_kv_heads=4,
        head_dim=128,
        device=device,
        dtype=dtype
    )


@pytest.fixture
def mock_transformer_layer(device, dtype):
    """单个 Transformer 层 mock"""
    return MockGemmaModelFactory.create_transformer_layer(
        num_heads=8,
        num_kv_heads=4,
        head_dim=128,
        device=device,
        dtype=dtype
    )


@pytest.fixture
def mock_simple_tokenizer():
    """简化的 tokenizer mock"""
    return MockGemmaModelFactory.create_simple_tokenizer()


# ==================== ComfyUI RoPE mock ====================

@pytest.fixture
def mock_comfy_rope(monkeypatch):
    """Mock ComfyUI precompute_freqs_cis

    提供简化的 RoPE 频率计算。
    """
    def fake_precompute_freqs_cis(head_dim, position_ids, theta, scale, dims, device):
        """简化的 RoPE 计算"""
        seq_len = position_ids.shape[1]

        # 生成频率
        freqs = torch.arange(0, head_dim, 2, device=device).float()
        freqs = 1.0 / (theta ** (freqs / head_dim))

        # 计算位置编码
        t = position_ids.float()
        freqs = torch.outer(t.squeeze(), freqs)

        # 生成 cos 和 sin
        cos = torch.cos(freqs).unsqueeze(0).unsqueeze(0)  # (1, 1, seq, dim/2)
        sin = torch.sin(freqs).unsqueeze(0).unsqueeze(0)

        # 扩展到完整维度 (repeat_interleave 模拟)
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)

        return (cos, sin)

    # Mock COMFY_ROPE_AVAILABLE 为 True
    import comfy_newbie_gemma_chat
    monkeypatch.setattr(comfy_newbie_gemma_chat, 'COMFY_ROPE_AVAILABLE', True)

    # Mock precompute_freqs_cis 函数
    monkeypatch.setattr(
        "comfy.text_encoders.llama.precompute_freqs_cis",
        fake_precompute_freqs_cis
    )

    return fake_precompute_freqs_cis


# ==================== 测试数据 fixtures ====================

@pytest.fixture
def sample_input_ids(device):
    """示例输入 token IDs

    格式: [BOS, token1, token2, token3]
    """
    return torch.tensor([[2, 100, 200, 300]], device=torch.device(device))


@pytest.fixture
def sample_embed_weight(device, dtype):
    """示例 embedding 权重

    形状: (vocab_size, hidden_size)
    """
    torch.manual_seed(42)
    return torch.randn(256000, 1024, device=torch.device(device), dtype=dtype)


@pytest.fixture
def sample_freqs_cis(device):
    """示例 RoPE 频率 (cos, sin)

    形状: (1, 1, seq_len, head_dim)
    """
    seq_len = 4
    head_dim = 128
    cos = torch.randn(1, 1, seq_len, head_dim, device=torch.device(device))
    sin = torch.randn(1, 1, seq_len, head_dim, device=torch.device(device))
    return (cos, sin)


# ==================== Hooks ====================

def pytest_collection_modifyitems(config, items):
    """自动标记慢速测试"""
    for item in items:
        if "integration" in item.nodeid or "slow" in item.nodeid:
            item.add_marker(pytest.mark.slow)


# ==================== 测试前后钩子 ====================

@pytest.fixture(autouse=True)
def reset_random_seed():
    """每个测试前重置随机种子以确保可重复性"""
    torch.manual_seed(42)
    yield
    # 测试后清理 (如果需要)
