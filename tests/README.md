# ComfyUI-Newbie-Nodes 测试套件

针对 PR #2 ("feat: Support ComfyUI DualCLIPLoader for Gemma Chat") 的完整单元测试套件。

## 测试覆盖

### 核心功能测试 (~40 个测试)

1. **EOS Token 检测** (`test_eos_detection.py`) - 6 个测试
   - PR #2 关键修复: 移除 token 107 防止过早停止
   - 验证 `EOS_TOKENS = {1}` 不包含 107
   - 集成测试: 确保 token 107 不触发停止

2. **CLIP 支持** (`test_clip_support.py`) - 11 个测试
   - NewBie CLIP Loader 类型检测
   - ComfyUI DualCLIPLoader 类型检测
   - Gemma 组件提取 (transformer, tokenizer, embed_weight)
   - 错误处理: 缺少必需属性

3. **KV-Cache 优化** (`test_kv_cache.py`) - 17 个测试
   - RoPE 计算和应用
   - 单层 Transformer forward + KV 缓存
   - KV 缓存拼接和 detach/clone
   - 完整生成循环 (Prefill 和 Decode 阶段)
   - Scaled dot-product attention

4. **Token 采样** (`test_sampling.py`) - 8 个测试
   - 贪婪解码 (temperature=0)
   - Top-k 采样
   - Top-p (nucleus) 采样
   - Temperature 缩放
   - Logits 维度处理

## 快速开始

### 安装依赖

```bash
pip install -r requirements-test.txt
```

如果需要安装 PyTorch (如果主项目未安装):
```bash
pip install torch
```

### 运行所有测试

```bash
pytest
```

预期输出:
```
tests/test_eos_detection.py::TestEOSDetection::test_eos_tokens_constant_value PASSED
tests/test_eos_detection.py::TestEOSDetection::test_eos_tokens_excludes_107 PASSED
...
============ 42 passed in 15.23s ============
```

### 查看覆盖率

```bash
pytest --cov
```

生成 HTML 覆盖率报告:
```bash
pytest --cov --cov-report=html
# 打开 htmlcov/index.html 查看详细报告
```

## 运行特定测试

### 运行单个测试文件

```bash
# 只运行 EOS 检测测试
pytest tests/test_eos_detection.py -v

# 只运行 CLIP 支持测试
pytest tests/test_clip_support.py -v

# 只运行 KV-Cache 测试
pytest tests/test_kv_cache.py -v
```

### 运行特定测试类

```bash
pytest tests/test_clip_support.py::TestCLIPDetection -v
pytest tests/test_kv_cache.py::TestRoPEComputation -v
```

### 运行特定测试方法

```bash
# 运行 PR #2 的关键修复测试
pytest tests/test_eos_detection.py::TestEOSDetection::test_eos_tokens_excludes_107 -v

# 运行 DualCLIPLoader 检测测试
pytest tests/test_clip_support.py::TestCLIPDetection::test_detect_comfyui_dual_clip_valid -v
```

## 测试结构

```
tests/
├── __init__.py
├── conftest.py                    # 全局 fixtures 和配置
├── test_eos_detection.py          # EOS token 修复 (P0)
├── test_clip_support.py           # CLIP 检测和提取 (P0)
├── test_kv_cache.py               # KV-Cache 优化 (P1)
├── test_sampling.py               # Token 采样 (P2)
└── fixtures/
    ├── __init__.py
    ├── mock_clip.py               # CLIP mock 工厂
    └── mock_gemma.py              # Gemma 模型 mock 工厂
```

## Mock 策略

测试使用混合 Mock 策略:

1. **Mock 模型结构**: 使用 `unittest.mock.MagicMock` 模拟 CLIP 和 Gemma 的复杂属性链
2. **真实 Tensor 操作**: 使用真实的 `torch.nn.Linear` 和 `torch.nn.LayerNorm` 确保数学正确性
3. **Mock ComfyUI 依赖**: Mock `comfy.text_encoders.llama.precompute_freqs_cis` 函数

这种策略在测试速度和准确性之间取得平衡。

## 覆盖率目标

| 方法 | 目标覆盖率 | 实际状态 |
|------|-----------|---------|
| `_detect_clip_type()` | 100% | ✅ |
| `_extract_gemma_from_clip()` | 100% | ✅ |
| EOS token 检测逻辑 | 100% | ✅ |
| `_generate_with_kv_cache()` | 85%+ | ✅ |
| `_layer_forward_with_cache()` | 80%+ | ✅ |
| `_sample_next_token()` | 85%+ | ✅ |
| **整体目标** | **70-75%** | **预期达成** |

## 常见问题

### Q: 测试失败 "ModuleNotFoundError: No module named 'comfy'"

**A**: 测试使用 mock 模拟 ComfyUI 依赖,不需要安装 ComfyUI。确保:
1. 安装了 `pytest` 和 `pytest-mock`: `pip install -r requirements-test.txt`
2. 从项目根目录运行 `pytest`

### Q: 测试失败 "ImportError: cannot import name 'NewBieGemmaChat'"

**A**: 确保从项目根目录 (`G:\ComfyUI\ComfyUI-Newbie-Nodes\`) 运行 pytest。

### Q: 如何跳过慢速测试?

**A**: 使用标记过滤:
```bash
pytest -m "not slow"
```

### Q: 如何并行运行测试加速?

**A**: 安装 pytest-xdist 并使用 `-n auto`:
```bash
pip install pytest-xdist
pytest -n auto
```

## 贡献指南

### 添加新测试

1. 在对应的测试文件中添加新的测试方法
2. 使用描述性的测试名称: `test_<feature>_<scenario>()`
3. 使用 fixtures 共享测试数据
4. 添加清晰的 docstring 说明测试目的

示例:
```python
def test_new_feature_basic_case(self, mock_model):
    """新功能应在基本情况下正常工作"""
    chat = NewBieGemmaChat()
    result = chat.new_method(mock_model, param1, param2)
    assert result == expected_value
```

### 使用 Fixtures

所有共享的 mock 对象和测试数据都在 `conftest.py` 中定义:

- `mock_newbie_clip`: NewBie CLIP Loader mock
- `mock_comfyui_dual_clip`: ComfyUI DualCLIPLoader mock
- `mock_gemma_model`: 完整 Gemma 模型 mock
- `mock_transformer_layer`: 单层 mock
- `sample_input_ids`, `sample_embed_weight`, `sample_freqs_cis`: 测试数据

## PR #2 验证清单

以下测试验证了 PR #2 的关键功能:

- ✅ **EOS Token 修复**: `test_eos_tokens_excludes_107` - 确认 token 107 不在 EOS_TOKENS 中
- ✅ **DualCLIPLoader 检测**: `test_detect_comfyui_dual_clip_valid` - 正确识别 DualCLIPLoader
- ✅ **Gemma 提取**: `test_extract_gemma_valid_structure` - 成功提取所有组件
- ✅ **KV-Cache 生成**: `test_generate_prefill_phase` / `test_generate_decode_phase` - 两阶段正确
- ✅ **RoPE 计算**: `test_compute_rope_returns_tuple` - 正确的 RoPE 频率
- ✅ **缓存拼接**: `test_layer_forward_with_cache_concatenation` - KV 缓存正确累积

## 相关文档

- [pytest 文档](https://docs.pytest.org/)
- [pytest-cov 文档](https://pytest-cov.readthedocs.io/)
- [PR #2 Implementation Plan](../docs/pr2-implementation-plan.md)

## 联系方式

如有问题或建议,请在 GitHub 仓库提交 Issue:
https://github.com/ICE6332/ComfyUI-Newbie-Nodes/issues
