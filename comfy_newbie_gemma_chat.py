"""
NewBie Gemma Chat 节点
使用 NewBie CLIP 中加载的 Gemma 模型进行聊天/文本生成

支持两种模式：
1. All-in-One 模式：直接使用 CLIP 内嵌的 Gemma 模型（推荐，节省显存）
2. 外部模型模式：加载独立的 HuggingFace 模型
"""

from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

# ComfyUI RoPE 计算（All-in-One 模式必需）
try:
    from comfy.text_encoders.llama import precompute_freqs_cis
    COMFY_ROPE_AVAILABLE = True
except ImportError:
    COMFY_ROPE_AVAILABLE = False

# HuggingFace transformers（外部模型模式必需）
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# 生成模型缓存，避免重复加载
_gemma_gen_cache: Dict[str, Any] = {
    "model": None,
    "tokenizer": None,
    "model_path": None,
    "device": None,
    "dtype": None,
}


def _clear_gemma_cache() -> None:
    """清除缓存的 Gemma 生成模型以释放显存"""
    global _gemma_gen_cache
    if _gemma_gen_cache["model"] is not None:
        print("[NewBie Gemma Chat] 正在卸载生成模型以释放显存...")
        del _gemma_gen_cache["model"]
        del _gemma_gen_cache["tokenizer"]
        _gemma_gen_cache["model"] = None
        _gemma_gen_cache["tokenizer"] = None
        _gemma_gen_cache["model_path"] = None
        _gemma_gen_cache["device"] = None
        _gemma_gen_cache["dtype"] = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("[NewBie Gemma Chat] 显存已释放")


class NewBieGemmaChat:
    """
    与 NewBie CLIP 中使用的 Gemma 模型进行聊天。
    此节点加载一个独立的 Gemma 实例用于文本生成。
    支持 NewBie CLIP Loader 和 ComfyUI DualCLIPLoader (type="newbie")。
    """

    # Gemma 特殊 token 常量: BOS=2, EOS=1, PAD=0
    EOS_TOKEN = 1           # <eos> 真正的结束符
    END_OF_TURN_TOKEN = 107 # <end_of_turn> 对话轮次结束符
    PAD_TOKEN = 0           # <pad> 填充符
    BOS_TOKEN = 2           # <bos> 开始符

    # EOS token 集合（仅用真正的 EOS token，避免过早终止）
    EOS_TOKENS = frozenset({1})
    # 扩展的结束 token 集合（包含 <end_of_turn>）
    EXTENDED_EOS_TOKENS = frozenset({1, 107})

    def _detect_clip_type(self, clip) -> str:
        """检测 CLIP 对象类型。

        返回值:
            'newbie_clip': NewBie CLIP Loader 加载的模型
            'comfyui_dual_clip': ComfyUI DualCLIPLoader (type="newbie") 加载的模型
            'unknown': 未知类型
        """
        # NewBie CLIP Loader: 具有 text_encoder + tokenizer.name_or_path
        if (hasattr(clip, 'text_encoder') and
            hasattr(clip, 'tokenizer') and
            hasattr(clip.tokenizer, 'name_or_path')):
            return "newbie_clip"

        # ComfyUI DualCLIPLoader (type="newbie"): 具有 cond_stage_model.gemma + tokenizer.gemma
        if (hasattr(clip, 'cond_stage_model') and
            hasattr(clip, 'tokenizer') and
            hasattr(clip.cond_stage_model, 'gemma') and
            hasattr(clip.tokenizer, 'gemma')):
            return "comfyui_dual_clip"

        return "unknown"

    def _extract_gemma_from_clip(self, clip) -> Tuple[Any, Any, torch.Tensor, torch.device, torch.dtype]:
        """从 ComfyUI DualCLIPLoader 中提取 Gemma 组件用于 All-in-One 模式。

        直接复用 DualCLIPLoader 加载的模型，无需额外下载或加载 HuggingFace 模型。

        返回值:
            tuple: (transformer, tokenizer, embed_weight, device, dtype)
        """
        # 验证 CLIP 结构
        if not hasattr(clip, 'cond_stage_model') or not hasattr(clip.cond_stage_model, 'gemma'):
            raise ValueError("CLIP 对象不包含 Gemma 模型。请使用 ComfyUI DualCLIPLoader (type=newbie)。")

        gemma_model = clip.cond_stage_model.gemma

        if not hasattr(gemma_model, 'transformer'):
            raise ValueError("Gemma 模型结构异常：缺少 transformer")

        transformer = gemma_model.transformer
        model_core = transformer.model if hasattr(transformer, 'model') else transformer

        # 获取 embed_tokens.weight 用于 weight tying（作为 lm_head）
        if not hasattr(model_core, 'embed_tokens'):
            raise ValueError("无法在 Gemma transformer 中找到 embed_tokens")
        embed_weight = model_core.embed_tokens.weight

        # 获取 tokenizer（SPieceTokenizer 将实际 tokenizer 存储在 .tokenizer 属性中）
        if not hasattr(clip, 'tokenizer') or not hasattr(clip.tokenizer, 'gemma'):
            raise ValueError("CLIP 对象不包含 Gemma tokenizer")
        tokenizer = clip.tokenizer.gemma
        if hasattr(tokenizer, 'tokenizer'):
            tokenizer = tokenizer.tokenizer

        # 从模型参数获取 device 和 dtype
        param = next(model_core.parameters())
        device, dtype = param.device, param.dtype

        print(f"[NewBie Gemma Chat] All-in-One 模式：使用 CLIP 内嵌的 Gemma 模型")
        print(f"[NewBie Gemma Chat] Device: {device}, Dtype: {dtype}")
        print(f"[NewBie Gemma Chat] Embed weight shape: {embed_weight.shape}")

        return transformer, tokenizer, embed_weight, device, dtype

    def _tokenize_with_spiece(self, tokenizer, text: str, device: torch.device) -> torch.Tensor:
        """使用 ComfyUI SPieceTokenizer 进行分词。

        参数:
            tokenizer: SPieceTokenizer 实例
            text: 要分词的文本
            device: 目标设备

        返回值:
            input_ids tensor，形状为 (1, seq_len)
        """
        if callable(tokenizer):
            result = tokenizer(text)
            tokens = result["input_ids"] if isinstance(result, dict) and "input_ids" in result else result
        elif hasattr(tokenizer, 'tokenizer') and hasattr(tokenizer.tokenizer, 'encode'):
            tokens = tokenizer.tokenizer.encode(text)
        else:
            raise ValueError(f"无法使用此 tokenizer 进行分词: {type(tokenizer)}")

        return torch.tensor([tokens], dtype=torch.long, device=device)

    def _decode_with_spiece(self, tokenizer, token_ids: List[int]) -> str:
        """使用 ComfyUI SPieceTokenizer 进行解码。

        参数:
            tokenizer: SPieceTokenizer 实例
            token_ids: token ID 列表

        返回值:
            解码后的文本字符串
        """
        # 过滤掉 PAD tokens
        clean_ids = [t for t in token_ids if t != self.PAD_TOKEN]

        if hasattr(tokenizer, 'tokenizer') and hasattr(tokenizer.tokenizer, 'decode'):
            return tokenizer.tokenizer.decode(clean_ids)
        if hasattr(tokenizer, 'decode'):
            return tokenizer.decode(clean_ids)
        raise ValueError(f"无法使用此 tokenizer 进行解码: {type(tokenizer)}")

    # ==================== KV-Cache 优化方法 ====================

    def _compute_rope_for_cache(self, model, position_ids: torch.Tensor, device: torch.device):
        """计算 KV-Cache 生成所需的 RoPE 频率。

        参数:
            model: Llama2_ 模型实例
            position_ids: 位置 ID tensor，形状为 (1, seq_len)
            device: 目标设备

        返回值:
            RoPE 频率（tuple 或 Gemma3 的 tuple 列表）
        """
        if not COMFY_ROPE_AVAILABLE:
            raise RuntimeError(
                "All-in-One 模式需要 ComfyUI 的 precompute_freqs_cis。"
                "请安装 ComfyUI 或使用外部模型模式 (use_clip_model=False)。"
            )

        config = model.config
        return precompute_freqs_cis(
            config.head_dim,
            position_ids,
            config.rope_theta,
            config.rope_scale,
            config.rope_dims,
            device=device
        )

    def _apply_rope_for_cache(self, xq: torch.Tensor, xk: torch.Tensor, freqs_cis: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """对 query 和 key tensor 应用旋转位置编码 (RoPE)。

        参数:
            xq: Query tensor，形状为 (batch, heads, seq, head_dim)
            xk: Key tensor，形状为 (batch, heads, seq, head_dim)
            freqs_cis: (cos, sin) 元组

        返回值:
            (rotated_query, rotated_key) 元组
        """
        cos, sin = freqs_cis

        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        org_dtype = xq.dtype
        xq_out = (xq * cos) + (rotate_half(xq) * sin)
        xk_out = (xk * cos) + (rotate_half(xk) * sin)
        return xq_out.to(org_dtype), xk_out.to(org_dtype)

    def _scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算缩放点积注意力。

        参数:
            q: Query tensor，形状为 (batch, heads, seq_q, head_dim)
            k: Key tensor，形状为 (batch, heads, seq_k, head_dim)
            v: Value tensor，形状为 (batch, heads, seq_k, head_dim)
            mask: 可选的注意力掩码，形状为 (seq_q, seq_k)

        返回值:
            注意力输出，形状为 (batch, heads, seq_q, head_dim)
        """
        scale = q.shape[-1] ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            attn_weights = attn_weights + mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        return torch.matmul(attn_weights, v)

    def _sample_next_token(self, logits: torch.Tensor, temperature: float, top_p: float, top_k: int) -> torch.Tensor:
        """从 logits 中采样下一个 token，支持 top-k/top-p 过滤。

        参数:
            logits: Logits tensor，形状为 (1, seq, vocab_size) 或 (1, vocab_size)
            temperature: 采样温度
            top_p: nucleus sampling 阈值
            top_k: top-k sampling 值

        返回值:
            下一个 token tensor，形状为 (1, 1)
        """
        # 确保形状为 (1, vocab_size)
        if logits.dim() == 3:
            logits = logits[:, -1, :]

        if temperature > 0:
            logits = logits / temperature

        # Top-k 过滤
        if top_k > 0:
            top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            threshold = top_k_vals[:, -1:]
            logits = torch.where(
                logits < threshold,
                torch.full_like(logits, float('-inf')),
                logits
            )

        # Top-p (nucleus) 过滤
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove_mask = cumsum > top_p
            remove_mask[:, 1:] = remove_mask[:, :-1].clone()
            remove_mask[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(remove_mask, float('-inf'))
            logits = torch.zeros_like(logits).scatter(1, sorted_idx, sorted_logits)

        if temperature > 0:
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1)
        return logits.argmax(dim=-1, keepdim=True)

    def _layer_forward_with_cache(
        self,
        layer,
        x: torch.Tensor,
        freqs_cis,
        attention_mask: Optional[torch.Tensor],
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """带 KV 缓存的单层 Transformer forward pass。

        参数:
            layer: TransformerBlockGemma2 层
            x: 输入 tensor，形状为 (batch, seq, hidden)
            freqs_cis: RoPE 频率
            attention_mask: 因果掩码或 None
            past_kv: 缓存的 (key, value) 元组或 None

        返回值:
            (输出 tensor, 新的 KV 缓存) 元组
        """
        attn = layer.self_attn

        # Gemma3 双 RoPE 处理：常规层使用 global RoPE，滑动注意力层使用 local RoPE
        # precompute_freqs_cis 对 Gemma3 返回 [(cos1, sin1), (cos2, sin2)]
        layer_freqs = freqs_cis
        is_dual_rope = (isinstance(freqs_cis, list) and
                        len(freqs_cis) == 2 and
                        isinstance(freqs_cis[0], tuple))

        if is_dual_rope:
            use_local_rope = hasattr(layer, 'sliding_attention') and layer.sliding_attention
            layer_freqs = freqs_cis[1] if use_local_rope else freqs_cis[0]

        # 自注意力
        residual = x
        x = layer.input_layernorm(x)
        batch, seq, _ = x.shape

        # Q, K, V 投影
        xq = attn.q_proj(x)
        xk = attn.k_proj(x)
        xv = attn.v_proj(x)

        # 重塑为 (batch, num_heads, seq, head_dim)
        xq = xq.view(batch, seq, attn.num_heads, attn.head_dim).transpose(1, 2)
        xk = xk.view(batch, seq, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
        xv = xv.view(batch, seq, attn.num_kv_heads, attn.head_dim).transpose(1, 2)

        # Q, K 归一化（Gemma3 特有）
        if attn.q_norm is not None:
            xq = attn.q_norm(xq)
        if attn.k_norm is not None:
            xk = attn.k_norm(xk)

        # 应用 RoPE
        xq, xk = self._apply_rope_for_cache(xq, xk, layer_freqs)

        # 与缓存的 KV 拼接（如果有）
        if past_kv is not None:
            cached_k, cached_v = past_kv
            xk = torch.cat([cached_k, xk], dim=2)
            xv = torch.cat([cached_v, xv], dim=2)

        # 存储新的 KV 缓存
        new_kv = (xk.detach().clone(), xv.detach().clone())

        # GQA: 为分组查询注意力扩展 K, V
        n_rep = attn.num_heads // attn.num_kv_heads
        if n_rep > 1:
            xk_expanded = xk.repeat_interleave(n_rep, dim=1)
            xv_expanded = xv.repeat_interleave(n_rep, dim=1)
        else:
            xk_expanded, xv_expanded = xk, xv

        # 计算注意力
        attn_output = self._scaled_dot_product_attention(
            xq, xk_expanded, xv_expanded, attention_mask
        )

        # 输出投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq, -1)
        attn_output = attn.o_proj(attn_output)

        # 后注意力层归一化 + 残差
        x = layer.post_attention_layernorm(attn_output)
        x = residual + x

        # MLP 块
        residual = x
        x = layer.pre_feedforward_layernorm(x)
        x = layer.mlp(x)
        x = layer.post_feedforward_layernorm(x)
        x = residual + x

        return x, new_kv

    def _generate_with_kv_cache(
        self,
        model,
        embed_weight: torch.Tensor,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> List[int]:
        """带 KV-Cache 优化的文本生成。

        使用 KV-Cache 缓存先前计算的 key 和 value，避免重复计算，
        显著提升生成速度（约 12-13x 加速）。

        参数:
            model: Llama2_ 模型实例
            embed_weight: LM head 权重，形状为 (vocab_size, hidden_size)，用于 weight tying
            input_ids: 输入 token ID tensor，形状为 (1, seq_len)
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: nucleus sampling 阈值
            top_k: top-k sampling 值

        返回值:
            生成的 token ID 列表
        """
        device = input_ids.device
        dtype = next(model.parameters()).dtype

        # 初始化所有层的 KV 缓存
        num_layers = len(model.layers)
        kv_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * num_layers

        with torch.no_grad():
            # 阶段 1: Prefill - 处理整个输入序列
            print(f"[KV-Cache] Prefill 阶段: {input_ids.shape[1]} tokens")

            x = model.embed_tokens(input_ids)
            if model.normalize_in:
                x = x * (model.config.hidden_size ** 0.5)

            seq_len = x.shape[1]
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            freqs_cis = self._compute_rope_for_cache(model, position_ids, device)

            # 构建 prefill 的因果掩码
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype),
                diagonal=1
            )

            # 遍历所有层，构建 KV 缓存
            for i, layer in enumerate(model.layers):
                x, kv_cache[i] = self._layer_forward_with_cache(
                    layer, x, freqs_cis, causal_mask, past_kv=None
                )

            if model.norm is not None:
                x = model.norm(x)

            # 采样第一个 token
            last_hidden = x[:, -1:, :]
            logits = last_hidden @ embed_weight.T
            next_token = self._sample_next_token(logits, temperature, top_p, top_k)
            generated = [next_token.item()]

            # 阶段 2: Decode - 使用缓存的 KV 逐个生成 token
            print(f"[KV-Cache] Decode 阶段: 最多生成 {max_new_tokens - 1} 个 token")

            for step in range(max_new_tokens - 1):
                x = model.embed_tokens(next_token)
                if model.normalize_in:
                    x = x * (model.config.hidden_size ** 0.5)

                current_pos = seq_len + step + 1
                position_ids = torch.tensor([[current_pos - 1]], device=device)
                freqs_cis = self._compute_rope_for_cache(model, position_ids, device)

                # 解码阶段不需要因果掩码（单个查询 token）
                for i, layer in enumerate(model.layers):
                    x, kv_cache[i] = self._layer_forward_with_cache(
                        layer, x, freqs_cis, None, past_kv=kv_cache[i]
                    )

                if model.norm is not None:
                    x = model.norm(x)

                logits = x @ embed_weight.T
                next_token = self._sample_next_token(logits, temperature, top_p, top_k)
                generated.append(next_token.item())

                # 检查 EOS token
                if next_token.item() in self.EOS_TOKENS:
                    print(f"[KV-Cache] 在第 {step + 1} 步生成了 EOS token")
                    break

                if (step + 1) % 50 == 0:
                    print(f"[KV-Cache] 已生成 {step + 1} 个 token...")

        print(f"[KV-Cache] 生成完成: {len(generated)} 个 token")

        # 显式清理 KV 缓存以释放显存
        del kv_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return generated

    # ==================== KV-Cache 方法结束 ====================

    def _chat_with_clip_model(
        self,
        clip,
        user_message: str,
        system_prompt: str,
        temperature: float,
        max_new_tokens: int,
        top_p: float,
        top_k: int,
        conversation_history: str,
    ) -> Tuple[str, str]:
        """使用 CLIP 内嵌的 Gemma 模型进行聊天（All-in-One 模式）。

        直接复用 DualCLIPLoader 加载的模型，无需额外下载。
        使用 KV-Cache 优化显著提升生成速度。
        """
        transformer, tokenizer, embed_weight, device, _ = self._extract_gemma_from_clip(clip)
        messages = self._format_messages(system_prompt, user_message, conversation_history)
        prompt = self._apply_chat_template_manual(messages)

        print(f"[NewBie Gemma Chat] All-in-One (KV-Cache): Generating response...")
        print(f"[NewBie Gemma Chat] Temperature: {temperature}, Max tokens: {max_new_tokens}")

        input_ids = self._tokenize_with_spiece(tokenizer, prompt, device)
        print(f"[NewBie Gemma Chat] Input tokens: {input_ids.shape[1]}")

        model = transformer.model if hasattr(transformer, 'model') else transformer

        generated_ids = self._generate_with_kv_cache(
            model, embed_weight, input_ids,
            max_new_tokens, temperature, top_p, top_k
        )

        response = self._decode_with_spiece(tokenizer, generated_ids)
        response = response.replace("<end_of_turn>", "").strip()
        full_conversation = self._build_conversation_history(user_message, response, conversation_history)

        print(f"[NewBie Gemma Chat] All-in-One (KV-Cache): Response generated ({len(response)} chars)")
        return response, full_conversation

    def _apply_chat_template_manual(self, messages: List[Dict[str, str]]) -> str:
        """手动应用 Gemma3 聊天模板。

        当 SPieceTokenizer 没有 apply_chat_template() 方法时使用。
        """
        result = ""
        for message in messages:
            role = message['role']
            content = message['content']
            if role == 'user':
                result += f"<start_of_turn>user\n{content}<end_of_turn>\n"
            elif role == 'assistant':
                result += f"<start_of_turn>model\n{content}<end_of_turn>\n"

        result += "<start_of_turn>model\n"
        return result

    default_sys_prompt = """
You are a Danbooru-to-XML prompt converter for AI image generation.

INPUT: Comma-separated Danbooru tags
OUTPUT: XML prompt in the exact format below

STRICT RULES:
1. PRESERVE tags exactly as given - do not modify, translate, or rephrase
2. ESCAPE parentheses with backslash: ( becomes \\( and ) becomes \\)
3. Character names go in <n> tags exactly as written (e.g., hatsune_miku, rem_\\(re:zero\\))
4. If no character name is specified, keep <n></n> empty but do not delete it
5. For single character, omit <character_2> entirely
6. Artist tags (artist:name or by_artist) go in <artists> as: artist:exact_name
7. Delete <clothing> tag entirely if no clothing specified
8. The <caption> section uses natural language WITHOUT underscores - convert blue_hair to "blue hair" etc.

CLASSIFICATION GUIDE:
- Gender: 1girl, 1boy, 2girls, multiple_boys, etc.
- Appearance: hair color, eye color, body features (blue_hair, red_eyes, long_hair, etc.)
- Clothing: outfit tags (dress, school_uniform, hat, etc.)
- Expression: emotions (smile, blush, crying, angry, etc.)
- Action: poses/activities (sitting, standing, holding, looking_at_viewer, etc.)
- Background: setting tags (outdoors, indoors, classroom, forest, etc.)
- Atmosphere: mood tags (dark, bright, romantic, etc.)

XML FORMAT:
<character_1>
  <n></n>
  <gender></gender>
  <appearance></appearance>
  <clothing></clothing>
  <expression></expression>
  <action></action>
  <interaction></interaction>
  <position></position>
</character_1>

<general_tags>
  <count></count>
  <artists></artists>
  <style>anime style</style>
  <background></background>
  <environment></environment>
  <perspective></perspective>
  <atmosphere></atmosphere>
  <lighting></lighting>
  <resolution>max_high_resolution</resolution>
  <quality>very aesthetic, masterpiece, no text</quality>
  <objects></objects>
  <other></other>
</general_tags>

<caption>Natural language description of the scene with lighting/shadow details. Escape parentheses with backslash here too.</caption>
"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {
                    "tooltip": "CLIP model from NewBie CLIP Loader or ComfyUI DualCLIPLoader (type=newbie)"
                }),
                "user_message": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Your message to Gemma"
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful AI assistant.",
                    "tooltip": "System prompt to set Gemma's behavior"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Sampling temperature (0=deterministic, higher=more creative)"
                }),
                "max_new_tokens": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Maximum number of tokens to generate"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Top-p (nucleus) sampling threshold"
                }),
                "top_k": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Top-k sampling (0=disabled)"
                }),
                "conversation_history": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Previous conversation history (optional, for multi-turn chat)"
                }),
                "unload_after": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Unload the generation model after response to free VRAM (only for external model mode)"
                }),
                "use_clip_model": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "All-in-One mode: directly use CLIP's embedded Gemma model (recommended, saves VRAM)"
                }),
                "gemma_model_path": ("STRING", {
                    "default": "",
                    "tooltip": "External Gemma model path (only used when use_clip_model=False, e.g. google/gemma-3-4b-it)"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("response", "full_conversation",)
    OUTPUT_TOOLTIPS = ("Gemma's response", "Full conversation history for chaining",)
    FUNCTION = "chat"
    CATEGORY = "NewBie/LLM"
    TITLE = "NewBie Gemma Chat"
    DESCRIPTION = "Chat with the Gemma model. Supports both NewBie CLIP Loader and ComfyUI DualCLIPLoader (type=newbie)."

    def _get_gemma_model_info(self, clip, gemma_model_path: str = ""):
        """Extract Gemma model path and settings from the CLIP object."""
        clip_type = self._detect_clip_type(clip)
        model_path = self._resolve_model_path(clip, clip_type, gemma_model_path)
        device = self._get_device(clip)
        dtype = self._get_dtype(clip, clip_type)
        return model_path, device, dtype

    def _resolve_model_path(self, clip, clip_type: str, gemma_model_path: str) -> str:
        """Resolve the Gemma model path based on clip type and user input."""
        # User-specified path takes priority
        if gemma_model_path and gemma_model_path.strip():
            model_path = gemma_model_path.strip()
            print(f"[NewBie Gemma Chat] Using user-specified model path: {model_path}")
            return model_path

        if clip_type == "newbie_clip":
            model_path = self._extract_newbie_clip_path(clip)
            print(f"[NewBie Gemma Chat] Detected: NewBie CLIP mode")
            return model_path

        if clip_type == "comfyui_dual_clip":
            raise ValueError(
                "ComfyUI DualCLIPLoader detected but gemma_model_path is empty.\n"
                "Please provide the Gemma generation model path (e.g., google/gemma-3-4b-it or local path).\n"
                "Note: The encoder model in DualCLIPLoader cannot be used for text generation."
            )

        raise ValueError(
            "Unknown CLIP type. Please use:\n"
            "1. NewBie CLIP Loader, or\n"
            "2. ComfyUI DualCLIPLoader (type=newbie) with gemma_model_path specified"
        )

    def _extract_newbie_clip_path(self, clip) -> str:
        """从 NewBie CLIP 对象中提取模型路径。"""
        # 优先尝试 tokenizer.name_or_path
        if hasattr(clip, 'tokenizer') and hasattr(clip.tokenizer, 'name_or_path'):
            return clip.tokenizer.name_or_path

        # 尝试 text_encoder.name_or_path
        if hasattr(clip, 'text_encoder'):
            if hasattr(clip.text_encoder, 'name_or_path'):
                return clip.text_encoder.name_or_path
            if hasattr(clip.text_encoder, 'config') and hasattr(clip.text_encoder.config, '_name_or_path'):
                return clip.text_encoder.config._name_or_path

        raise ValueError("无法从 NewBie CLIP 中获取 Gemma 模型路径。")

    def _get_device(self, clip) -> str:
        """确定模型使用的设备。"""
        if hasattr(clip, 'device'):
            return clip.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _get_dtype(self, clip, clip_type: str) -> torch.dtype:
        """确定模型使用的数据类型。"""
        if clip_type == "newbie_clip" and hasattr(clip, 'text_encoder'):
            return next(clip.text_encoder.parameters()).dtype

        if clip_type == "comfyui_dual_clip" and hasattr(clip, 'cond_stage_model'):
            gemma = getattr(clip.cond_stage_model, 'gemma', None)
            if gemma is not None and hasattr(gemma, 'dtype'):
                return gemma.dtype

        return torch.bfloat16

    def _load_generation_model(self, model_path: str, device: str, dtype: torch.dtype):
        """加载或获取缓存的 Gemma 生成模型。"""
        global _gemma_gen_cache

        # 检查是否可以复用缓存的模型
        if (_gemma_gen_cache["model"] is not None and
            _gemma_gen_cache["model_path"] == model_path and
            _gemma_gen_cache["device"] == device and
            _gemma_gen_cache["dtype"] == dtype):
            print(f"[NewBie Gemma Chat] 使用缓存的生成模型")
            return _gemma_gen_cache["model"], _gemma_gen_cache["tokenizer"]

        # 清除旧缓存
        if _gemma_gen_cache["model"] is not None:
            del _gemma_gen_cache["model"]
            del _gemma_gen_cache["tokenizer"]
            torch.cuda.empty_cache()

        print(f"[NewBie Gemma Chat] 正在从 {model_path} 加载 Gemma 生成模型")
        print(f"[NewBie Gemma Chat] Device: {device}, Dtype: {dtype}")

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Gemma Chat 需要 transformers 库")

        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        # 加载因果语言模型（用于生成）
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        model.eval()

        # 缓存以便复用
        _gemma_gen_cache["model"] = model
        _gemma_gen_cache["tokenizer"] = tokenizer
        _gemma_gen_cache["model_path"] = model_path
        _gemma_gen_cache["device"] = device
        _gemma_gen_cache["dtype"] = dtype

        print(f"[NewBie Gemma Chat] 模型加载成功")
        return model, tokenizer

    def _format_messages(self, system_prompt: str, user_message: str, conversation_history: str = "") -> List[Dict[str, str]]:
        """格式化消息以适配 Gemma 聊天模板。"""
        messages = []

        # 添加系统提示（如果有）
        if system_prompt and system_prompt.strip():
            messages.append({
                "role": "user",
                "content": f"[System Instructions]\n{system_prompt.strip()}\n\n[End System Instructions]\n\nAcknowledge that you understand these instructions."
            })
            messages.append({
                "role": "assistant",
                "content": "I understand and will follow these instructions."
            })
        
        # 解析并添加对话历史（如果有）
        if conversation_history and conversation_history.strip():
            # 解析格式: USER: ... ASSISTANT: ...
            history = conversation_history.strip()
            parts = []
            current_role = None
            current_content = []

            for line in history.split('\n'):
                if line.startswith('USER:'):
                    if current_role and current_content:
                        parts.append((current_role, '\n'.join(current_content)))
                    current_role = 'user'
                    current_content = [line[5:].strip()]
                elif line.startswith('ASSISTANT:'):
                    if current_role and current_content:
                        parts.append((current_role, '\n'.join(current_content)))
                    current_role = 'assistant'
                    current_content = [line[10:].strip()]
                elif current_role:
                    current_content.append(line)

            if current_role and current_content:
                parts.append((current_role, '\n'.join(current_content)))

            for role, content in parts:
                if content.strip():
                    messages.append({"role": role, "content": content.strip()})

        # 添加当前用户消息
        if user_message and user_message.strip():
            messages.append({
                "role": "user",
                "content": user_message.strip()
            })

        return messages

    def _apply_chat_template(self, tokenizer, messages: List[Dict[str, str]]) -> str:
        """应用 Gemma3 聊天模板到消息列表。"""
        # 尝试使用 tokenizer 内置的聊天模板
        if hasattr(tokenizer, 'apply_chat_template'):
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                print(f"[NewBie Gemma Chat] Tokenizer 聊天模板失败: {e}, 使用手动格式")

        # 回退到手动模板
        return self._apply_chat_template_manual(messages)

    def _build_generation_config(
        self,
        tokenizer,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
    ) -> Dict[str, Any]:
        """构建生成配置字典。"""
        use_sampling = do_sample and temperature > 0
        config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": use_sampling,
            "pad_token_id": tokenizer.eos_token_id,
        }

        if repetition_penalty != 1.0:
            config["repetition_penalty"] = repetition_penalty

        if use_sampling:
            config["temperature"] = temperature
            config["top_p"] = top_p
            if top_k > 0:
                config["top_k"] = top_k

        return config

    def _generate_response(
        self,
        model,
        tokenizer,
        prompt: str,
        device: str,
        generation_config: Dict[str, Any],
    ) -> str:
        """从模型生成并解码响应。"""
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_config)

        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.replace("<end_of_turn>", "").strip()

    def _build_conversation_history(
        self,
        user_message: str,
        response: str,
        conversation_history: str,
    ) -> str:
        """构建完整的对话历史字符串。"""
        new_exchange = f"USER: {user_message.strip()}\nASSISTANT: {response}"
        if conversation_history and conversation_history.strip():
            return f"{conversation_history.strip()}\n{new_exchange}"
        return new_exchange

    def chat(
        self,
        clip,
        user_message: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        top_p: float = 0.9,
        top_k: int = 50,
        conversation_history: str = "",
        unload_after: bool = True,
        use_clip_model: bool = True,
        gemma_model_path: str = "",
    ) -> Tuple[str, str]:
        """从 Gemma 生成聊天响应。

        支持两种模式：
        1. All-in-One 模式 (use_clip_model=True): 直接使用 CLIP 内嵌的 Gemma 模型
        2. 外部模型模式 (use_clip_model=False): 加载独立的 HuggingFace 模型
        """
        if not user_message or not user_message.strip():
            return "请提供消息。", conversation_history

        clip_type = self._detect_clip_type(clip)

        # All-in-One 模式：直接使用 CLIP 内嵌的 Gemma 模型
        if use_clip_model and clip_type == "comfyui_dual_clip":
            print(f"[NewBie Gemma Chat] 使用 All-in-One 模式（CLIP 内嵌模型）")
            return self._chat_with_clip_model(
                clip, user_message, system_prompt,
                temperature, max_new_tokens, top_p, top_k,
                conversation_history
            )

        # 外部模型模式：加载 HuggingFace 模型
        print(f"[NewBie Gemma Chat] 使用外部模型模式")
        model_path, device, dtype = self._get_gemma_model_info(clip, gemma_model_path)
        model, tokenizer = self._load_generation_model(model_path, device, dtype)

        messages = self._format_messages(system_prompt, user_message, conversation_history)
        prompt = self._apply_chat_template(tokenizer, messages)

        print(f"[NewBie Gemma Chat] 正在生成响应...")
        print(f"[NewBie Gemma Chat] Temperature: {temperature}, Max tokens: {max_new_tokens}")

        generation_config = self._build_generation_config(
            tokenizer, max_new_tokens, temperature, top_p, top_k
        )
        response = self._generate_response(model, tokenizer, prompt, device, generation_config)
        full_conversation = self._build_conversation_history(user_message, response, conversation_history)

        print(f"[NewBie Gemma Chat] Response generated ({len(response)} chars)")

        if unload_after:
            _clear_gemma_cache()

        return response, full_conversation


class NewBieGemmaChatAdvanced(NewBieGemmaChat):
    """
    高级版 Gemma Chat 节点。

    扩展基础版功能，提供更多生成参数控制：
    - repetition_penalty: 重复惩罚系数
    - do_sample: 是否启用采样（关闭则使用贪婪解码）
    - seed: 随机种子，用于结果复现
    """

    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()

        base_inputs["optional"]["repetition_penalty"] = ("FLOAT", {
            "default": 1.0,
            "min": 1.0,
            "max": 2.0,
            "step": 0.05,
            "tooltip": "Penalty for repeating tokens (1.0=no penalty)"
        })
        base_inputs["optional"]["do_sample"] = ("BOOLEAN", {
            "default": True,
            "tooltip": "Enable sampling (False=greedy decoding)"
        })
        base_inputs["optional"]["seed"] = ("INT", {
            "default": -1,
            "min": -1,
            "max": 2**31-1,
            "tooltip": "Random seed for reproducibility (-1=random)"
        })

        return base_inputs

    TITLE = "NewBie Gemma Chat (Advanced)"
    DESCRIPTION = "Advanced Gemma chat with additional generation parameters. Supports both NewBie CLIP Loader and ComfyUI DualCLIPLoader."

    def _set_seed(self, seed: int) -> None:
        """设置随机种子以确保结果可复现。"""
        if seed >= 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def chat(
        self,
        clip,
        user_message: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        top_p: float = 0.9,
        top_k: int = 50,
        conversation_history: str = "",
        unload_after: bool = True,
        use_clip_model: bool = True,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        seed: int = -1,
        gemma_model_path: str = "",
    ) -> Tuple[str, str]:
        """
        使用高级选项生成 Gemma 聊天响应。

        支持两种模式：
        1. All-in-One 模式 (use_clip_model=True): 直接使用 CLIP 内嵌的 Gemma 模型
           注意: repetition_penalty, do_sample, seed 在此模式下不生效
        2. 外部模型模式 (use_clip_model=False): 加载独立的 HuggingFace 模型
        """
        if not user_message or not user_message.strip():
            return "Please provide a message.", conversation_history

        clip_type = self._detect_clip_type(clip)

        # All-in-One 模式: 直接使用 CLIP 内嵌的 Gemma 模型
        if use_clip_model and clip_type == "comfyui_dual_clip":
            print(f"[NewBie Gemma Chat Advanced] 使用 All-in-One 模式 (CLIP 内嵌模型)")
            if repetition_penalty != 1.0 or not do_sample or seed >= 0:
                print(f"[NewBie Gemma Chat Advanced] 警告: repetition_penalty, do_sample, seed 在 All-in-One 模式下被忽略")
            return self._chat_with_clip_model(
                clip, user_message, system_prompt,
                temperature, max_new_tokens, top_p, top_k,
                conversation_history
            )

        # 外部模型模式: 加载 HuggingFace 模型
        print(f"[NewBie Gemma Chat Advanced] 使用外部模型模式")
        self._set_seed(seed)

        model_path, device, dtype = self._get_gemma_model_info(clip, gemma_model_path)
        model, tokenizer = self._load_generation_model(model_path, device, dtype)

        messages = self._format_messages(system_prompt, user_message, conversation_history)
        prompt = self._apply_chat_template(tokenizer, messages)

        print(f"[NewBie Gemma Chat Advanced] 正在生成响应...")
        print(f"[NewBie Gemma Chat Advanced] Temperature: {temperature}, 最大 token 数: {max_new_tokens}, Seed: {seed}")

        generation_config = self._build_generation_config(
            tokenizer, max_new_tokens, temperature, top_p, top_k,
            do_sample=do_sample, repetition_penalty=repetition_penalty
        )
        response = self._generate_response(model, tokenizer, prompt, device, generation_config)
        full_conversation = self._build_conversation_history(user_message, response, conversation_history)

        print(f"[NewBie Gemma Chat Advanced] 响应生成完成 ({len(response)} 字符)")

        if unload_after:
            _clear_gemma_cache()

        return response, full_conversation


NODE_CLASS_MAPPINGS = {
    "NewBieGemmaChat": NewBieGemmaChat,
    "NewBieGemmaChatAdvanced": NewBieGemmaChatAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NewBieGemmaChat": "NewBie Gemma Chat",
    "NewBieGemmaChatAdvanced": "NewBie Gemma Chat (Advanced)",
}
