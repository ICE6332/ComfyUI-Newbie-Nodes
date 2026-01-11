"""
NewBie Gemma Chat Node
Enables chat/text generation using the Gemma model loaded in NewBie CLIP.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List

# Import RoPE computation from ComfyUI
try:
    from comfy.text_encoders.llama import precompute_freqs_cis
    COMFY_ROPE_AVAILABLE = True
except ImportError:
    COMFY_ROPE_AVAILABLE = False
    print("Warning: Could not import precompute_freqs_cis from ComfyUI")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available for Gemma Chat")


# Cache for the generation model to avoid reloading
_gemma_gen_cache = {
    "model": None,
    "tokenizer": None,
    "model_path": None,
    "device": None,
    "dtype": None,
}


def _clear_gemma_cache():
    """Clear the cached Gemma generation model to free VRAM"""
    global _gemma_gen_cache
    if _gemma_gen_cache["model"] is not None:
        print("[NewBie Gemma Chat] Unloading generation model to free VRAM...")
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
        print("[NewBie Gemma Chat] VRAM cleared")


class NewBieGemmaChat:
    """
    Chat with the Gemma model that's used in NewBie CLIP.
    This node loads a separate instance of Gemma for text generation.
    Supports both NewBie CLIP Loader and ComfyUI DualCLIPLoader (type="newbie").
    """

    def _detect_clip_type(self, clip):
        """
        Detect CLIP object type.
        Returns: "newbie_clip" | "comfyui_dual_clip" | "unknown"
        """
        # NewBie CLIP Loader: has text_encoder + tokenizer with name_or_path
        has_text_encoder = hasattr(clip, 'text_encoder') and hasattr(clip, 'tokenizer')
        if has_text_encoder and hasattr(clip.tokenizer, 'name_or_path'):
            return "newbie_clip"

        # ComfyUI DualCLIPLoader (type="newbie"): has cond_stage_model.gemma + tokenizer.gemma
        has_cond_stage = hasattr(clip, 'cond_stage_model') and hasattr(clip, 'tokenizer')
        if has_cond_stage and hasattr(clip.cond_stage_model, 'gemma') and hasattr(clip.tokenizer, 'gemma'):
            return "comfyui_dual_clip"

        return "unknown"

    def _extract_gemma_from_clip(self, clip):
        """
        从 ComfyUI DualCLIPLoader 加载的 CLIP 中提取 Gemma 组件
        用于 All-in-One 模式，直接复用已加载的模型进行文本生成

        Returns: (transformer, tokenizer, embed_weight, device, dtype)
        """
        # 访问路径: clip.cond_stage_model.gemma.transformer.model
        if not hasattr(clip, 'cond_stage_model') or not hasattr(clip.cond_stage_model, 'gemma'):
            raise ValueError("CLIP object does not contain Gemma model. Use ComfyUI DualCLIPLoader (type=newbie).")

        gemma_model = clip.cond_stage_model.gemma

        # 获取 transformer (Llama2_)
        if not hasattr(gemma_model, 'transformer'):
            raise ValueError("Gemma model structure unexpected: missing transformer")

        transformer = gemma_model.transformer

        # 获取实际的模型层 (Llama2_.model)
        if hasattr(transformer, 'model'):
            model_core = transformer.model
        else:
            model_core = transformer

        # 获取 embed_tokens.weight 用于 weight tying (作为 lm_head)
        if not hasattr(model_core, 'embed_tokens'):
            raise ValueError("Cannot find embed_tokens in Gemma transformer")

        embed_weight = model_core.embed_tokens.weight  # (262208, 2560)

        # 获取 tokenizer
        if not hasattr(clip, 'tokenizer') or not hasattr(clip.tokenizer, 'gemma'):
            raise ValueError("CLIP object does not contain Gemma tokenizer")

        tokenizer = clip.tokenizer.gemma
        # SPieceTokenizer 在 .tokenizer 属性下
        if hasattr(tokenizer, 'tokenizer'):
            tokenizer = tokenizer.tokenizer

        # 获取 device 和 dtype
        device = next(model_core.parameters()).device
        dtype = next(model_core.parameters()).dtype

        print(f"[NewBie Gemma Chat] All-in-One mode: Using CLIP's embedded Gemma model")
        print(f"[NewBie Gemma Chat] Device: {device}, Dtype: {dtype}")
        print(f"[NewBie Gemma Chat] Embed weight shape: {embed_weight.shape}")

        return transformer, tokenizer, embed_weight, device, dtype

    def _tokenize_with_spiece(self, tokenizer, text: str, device) -> torch.Tensor:
        """
        使用 ComfyUI SPieceTokenizer 进行分词

        Args:
            tokenizer: SPieceTokenizer 实例
            text: 要分词的文本
            device: 目标设备

        Returns:
            input_ids: (1, seq_len) tensor
        """
        # SPieceTokenizer 使用 __call__ 方法，返回 {"input_ids": [...]}
        if callable(tokenizer):
            result = tokenizer(text)
            if isinstance(result, dict) and "input_ids" in result:
                tokens = result["input_ids"]
            else:
                tokens = result
        elif hasattr(tokenizer, 'tokenizer') and hasattr(tokenizer.tokenizer, 'encode'):
            # 访问内部的 SentencePieceProcessor
            tokens = tokenizer.tokenizer.encode(text)
        else:
            raise ValueError(f"Cannot tokenize with: {type(tokenizer)}")

        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        return input_ids

    def _decode_with_spiece(self, tokenizer, token_ids: list) -> str:
        """
        使用 ComfyUI SPieceTokenizer 进行解码

        Args:
            tokenizer: SPieceTokenizer 实例
            token_ids: token ID 列表

        Returns:
            解码后的文本
        """
        # Gemma special tokens: BOS=2, EOS=1, PAD=0
        # 清理特殊 tokens (保留更多 tokens 以确保输出完整)
        clean_ids = [t for t in token_ids if t not in [0]]

        # SPieceTokenizer 的内部 tokenizer 是 SentencePieceProcessor
        if hasattr(tokenizer, 'tokenizer') and hasattr(tokenizer.tokenizer, 'decode'):
            return tokenizer.tokenizer.decode(clean_ids)
        elif hasattr(tokenizer, 'decode'):
            return tokenizer.decode(clean_ids)
        else:
            raise ValueError(f"Cannot decode with: {type(tokenizer)}")

    # ==================== KV-Cache 优化方法 ====================

    def _compute_rope_for_cache(self, model, position_ids, device):
        """
        计算 RoPE frequencies for KV-cache generation

        Args:
            model: Llama2_ 模型
            position_ids: (1, seq_len) 位置 IDs
            device: 目标设备

        Returns:
            freqs_cis: RoPE frequencies (可能是 tuple for Gemma3)
        """
        if not COMFY_ROPE_AVAILABLE:
            raise RuntimeError(
                "All-in-One mode requires ComfyUI's precompute_freqs_cis. "
                "Please install ComfyUI or use external model mode (use_clip_model=False)."
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

    def _apply_rope_for_cache(self, xq, xk, freqs_cis):
        """
        应用 rotary position embedding

        Args:
            xq: Query tensor (batch, heads, seq, head_dim)
            xk: Key tensor (batch, heads, seq, head_dim)
            freqs_cis: (cos, sin) tuple

        Returns:
            (xq_rotated, xk_rotated)
        """
        cos, sin = freqs_cis

        def rotate_half(x):
            x1 = x[..., :x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        org_dtype = xq.dtype
        xq_out = (xq * cos) + (rotate_half(xq) * sin)
        xk_out = (xk * cos) + (rotate_half(xk) * sin)
        return xq_out.to(org_dtype), xk_out.to(org_dtype)

    def _scaled_dot_product_attention(self, q, k, v, mask=None):
        """
        标准 scaled dot-product attention

        Args:
            q: Query (batch, heads, seq_q, head_dim)
            k: Key (batch, heads, seq_k, head_dim)
            v: Value (batch, heads, seq_k, head_dim)
            mask: Attention mask (seq_q, seq_k) or None

        Returns:
            attention output (batch, heads, seq_q, head_dim)
        """
        scale = q.shape[-1] ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            # 扩展 mask 到正确的形状
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_q, seq_k)
            attn_weights = attn_weights + mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        return torch.matmul(attn_weights, v)

    def _sample_token_for_cache(self, logits, temperature, top_p, top_k):
        """
        采样下一个 token

        Args:
            logits: (1, seq, vocab_size) or (1, vocab_size)
            temperature: 采样温度
            top_p: nucleus sampling 阈值
            top_k: top-k 采样值

        Returns:
            next_token: (1, 1) tensor
        """
        # 确保是 (1, vocab_size)
        if logits.dim() == 3:
            logits = logits[:, -1, :]

        if temperature > 0:
            logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            threshold = top_k_vals[:, -1:]
            logits = torch.where(logits < threshold,
                                torch.full_like(logits, float('-inf')),
                                logits)

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove_mask = cumsum > top_p
            remove_mask[:, 1:] = remove_mask[:, :-1].clone()
            remove_mask[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(remove_mask, float('-inf'))
            # Scatter back
            logits = torch.zeros_like(logits).scatter(1, sorted_idx, sorted_logits)

        if temperature > 0:
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1)
        else:
            return logits.argmax(dim=-1, keepdim=True)

    def _layer_forward_with_cache(
        self,
        layer,
        x: torch.Tensor,
        freqs_cis,
        attention_mask,
        past_kv,
        layer_idx: int,
    ):
        """
        单层 forward，带 KV 缓存

        Args:
            layer: TransformerBlockGemma2 层
            x: (batch, seq, hidden) 输入
            freqs_cis: RoPE frequencies
            attention_mask: Causal mask or None
            past_kv: (cached_k, cached_v) or None
            layer_idx: 层索引

        Returns:
            (output, new_kv)
        """
        attn = layer.self_attn

        # Gemma3 特殊处理：选择正确的 RoPE
        # precompute_freqs_cis 对 Gemma3 返回 list: [(cos1, sin1), (cos2, sin2)]
        # 第一个是 global RoPE (theta=1000000), 第二个是 local RoPE (theta=10000)
        layer_freqs = freqs_cis
        is_dual_rope = isinstance(freqs_cis, list) and len(freqs_cis) == 2 and isinstance(freqs_cis[0], tuple)

        if is_dual_rope:
            if hasattr(layer, 'sliding_attention') and layer.sliding_attention:
                # Sliding attention 层用 local RoPE
                layer_freqs = freqs_cis[1]
            else:
                # 非 sliding 层用 global RoPE
                layer_freqs = freqs_cis[0]

        # === Self Attention ===
        residual = x
        x = layer.input_layernorm(x)

        batch, seq, hidden = x.shape

        # Q, K, V projection
        xq = attn.q_proj(x)
        xk = attn.k_proj(x)
        xv = attn.v_proj(x)

        # Reshape: (batch, seq, num_heads * head_dim) -> (batch, num_heads, seq, head_dim)
        xq = xq.view(batch, seq, attn.num_heads, attn.head_dim).transpose(1, 2)
        xk = xk.view(batch, seq, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
        xv = xv.view(batch, seq, attn.num_kv_heads, attn.head_dim).transpose(1, 2)

        # Q, K norm (Gemma3 特有)
        if attn.q_norm is not None:
            xq = attn.q_norm(xq)
        if attn.k_norm is not None:
            xk = attn.k_norm(xk)

        # RoPE
        xq, xk = self._apply_rope_for_cache(xq, xk, layer_freqs)

        # === KV Cache ===
        if past_kv is not None:
            cached_k, cached_v = past_kv
            xk = torch.cat([cached_k, xk], dim=2)
            xv = torch.cat([cached_v, xv], dim=2)

        # 保存新的 KV cache
        new_kv = (xk.detach().clone(), xv.detach().clone())

        # GQA: expand K,V for grouped query attention
        n_rep = attn.num_heads // attn.num_kv_heads
        if n_rep > 1:
            xk_expanded = xk.repeat_interleave(n_rep, dim=1)
            xv_expanded = xv.repeat_interleave(n_rep, dim=1)
        else:
            xk_expanded = xk
            xv_expanded = xv

        # Attention
        attn_output = self._scaled_dot_product_attention(
            xq, xk_expanded, xv_expanded, attention_mask
        )

        # Output projection: (batch, heads, seq, head_dim) -> (batch, seq, hidden)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq, -1)
        attn_output = attn.o_proj(attn_output)

        # Post attention layernorm + residual
        x = layer.post_attention_layernorm(attn_output)
        x = residual + x

        # === MLP ===
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
        """
        带 KV-Cache 的文本生成

        Args:
            model: Llama2_ 模型
            embed_weight: lm_head 权重 (vocab_size, hidden_size)
            input_ids: (1, seq_len) 输入 token IDs
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: nucleus sampling 阈值
            top_k: top-k 采样值

        Returns:
            generated_ids: 生成的 token ID 列表
        """
        device = input_ids.device
        dtype = next(model.parameters()).dtype

        # 初始化 KV cache
        num_layers = len(model.layers)
        kv_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * num_layers

        # EOS tokens - 只用真正的 EOS token
        # 注意：107 是 <end_of_turn> 但在某些情况下可能误触发
        # Gemma special tokens: BOS=2, EOS=1, PAD=0
        eos_tokens = {1}  # 只用 <eos>

        with torch.no_grad():
            # ==================== Phase 1: Prefill ====================
            print(f"[KV-Cache] Prefill phase: {input_ids.shape[1]} tokens")

            # Embedding
            x = model.embed_tokens(input_ids)
            if model.normalize_in:
                x = x * (model.config.hidden_size ** 0.5)

            seq_len = x.shape[1]
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

            # 预计算 RoPE
            freqs_cis = self._compute_rope_for_cache(model, position_ids, device)

            # Causal mask for prefill
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype),
                diagonal=1
            )

            # 逐层 forward，构建 KV cache
            for i, layer in enumerate(model.layers):
                x, kv_cache[i] = self._layer_forward_with_cache(
                    layer, x, freqs_cis, causal_mask,
                    past_kv=None, layer_idx=i
                )

            # Final norm
            if model.norm is not None:
                x = model.norm(x)

            # 计算第一个 token
            last_hidden = x[:, -1:, :]
            logits = last_hidden @ embed_weight.T
            next_token = self._sample_token_for_cache(logits, temperature, top_p, top_k)

            generated = [next_token.item()]

            # ==================== Phase 2: Decode ====================
            print(f"[KV-Cache] Decode phase: generating up to {max_new_tokens - 1} more tokens")

            for step in range(max_new_tokens - 1):
                # 只处理新 token
                x = model.embed_tokens(next_token)
                if model.normalize_in:
                    x = x * (model.config.hidden_size ** 0.5)

                # 当前位置
                current_pos = seq_len + step + 1
                position_ids = torch.tensor([[current_pos - 1]], device=device)
                freqs_cis = self._compute_rope_for_cache(model, position_ids, device)

                # Decode 时不需要 causal mask（只有1个query token）
                # 但需要确保 attention 只看之前的 tokens
                for i, layer in enumerate(model.layers):
                    x, kv_cache[i] = self._layer_forward_with_cache(
                        layer, x, freqs_cis, None,
                        past_kv=kv_cache[i], layer_idx=i
                    )

                # Final norm
                if model.norm is not None:
                    x = model.norm(x)

                # 采样
                logits = x @ embed_weight.T
                next_token = self._sample_token_for_cache(logits, temperature, top_p, top_k)

                generated.append(next_token.item())

                # EOS check
                if next_token.item() in eos_tokens:
                    print(f"[KV-Cache] EOS token generated at step {step + 1}")
                    break

                # 进度显示
                if (step + 1) % 50 == 0:
                    print(f"[KV-Cache] Generated {step + 1} tokens...")

        print(f"[KV-Cache] Generation complete: {len(generated)} tokens")
        return generated

    # ==================== End of KV-Cache 方法 ====================

    def _custom_generate(
        self,
        transformer,
        embed_weight: torch.Tensor,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        eos_token_id: int = 1,  # Gemma EOS token
    ) -> torch.Tensor:
        """
        自定义文本生成循环，使用 embed_tokens.weight 作为 lm_head

        Args:
            transformer: Gemma transformer 模型
            embed_weight: embed_tokens.weight，用于 weight tying
            input_ids: (1, seq_len) 输入 token IDs
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: nucleus sampling 阈值
            top_k: top-k sampling 值
            eos_token_id: 结束 token ID

        Returns:
            generated: 完整的 token ID 序列
        """
        generated = input_ids.clone()
        device = input_ids.device

        # 检测 EOS token - Gemma 通常使用 <end_of_turn> (107) 或 <eos> (1)
        # 实际的 EOS tokens
        eos_tokens = {1, 107}  # <eos> 和 <end_of_turn>

        with torch.no_grad():
            for step in range(max_new_tokens):
                # 1. Forward pass 获取 hidden states
                # ComfyUI 的 Llama2_.forward() 接受 (tokens, intermediate_output) 或只是 tokens
                try:
                    # 尝试标准调用方式
                    if hasattr(transformer, 'model'):
                        # Llama2_ 结构
                        hidden_states = transformer(generated, None)
                        if isinstance(hidden_states, tuple):
                            hidden_states = hidden_states[0]
                    else:
                        hidden_states = transformer(generated)
                        if isinstance(hidden_states, tuple):
                            hidden_states = hidden_states[0]
                except Exception as e:
                    print(f"[NewBie Gemma Chat] Forward pass error: {e}")
                    raise

                # 2. 取最后一个位置的 hidden state
                last_hidden = hidden_states[:, -1, :]  # (1, hidden_size)

                # 3. 计算 logits (weight tying: hidden @ embed_weight.T)
                logits = last_hidden @ embed_weight.T  # (1, vocab_size)

                # 4. 应用温度
                if temperature > 0:
                    logits = logits / temperature

                # 5. Top-k 过滤
                if top_k > 0:
                    top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    min_top_k = top_k_values[:, -1].unsqueeze(-1)
                    logits = torch.where(logits < min_top_k,
                                        torch.full_like(logits, float('-inf')),
                                        logits)

                # 6. Top-p (nucleus) 过滤
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # 移除累积概率超过 top_p 的 tokens
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # 保留第一个超过阈值的 token
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False

                    # 将移除的 token 设为 -inf
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits = logits.masked_fill(indices_to_remove, float('-inf'))

                # 7. 采样
                if temperature > 0:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = logits.argmax(dim=-1, keepdim=True)

                # 8. 拼接
                generated = torch.cat([generated, next_token], dim=1)

                # 9. 检查 EOS
                if next_token.item() in eos_tokens:
                    break

        return generated

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
        """
        使用 CLIP 内置 Gemma 模型的完整聊天流程 (All-in-One 模式)

        直接复用 DualCLIPLoader 加载的模型，无需额外下载/加载 HuggingFace 模型
        使用 KV-Cache 优化，大幅提升生成速度
        """
        # 1. 提取组件
        transformer, tokenizer, embed_weight, device, dtype = self._extract_gemma_from_clip(clip)

        # 2. 格式化消息
        messages = self._format_messages(system_prompt, user_message, conversation_history)

        # 3. 手工应用 Gemma3 chat template (SPieceTokenizer 没有 apply_chat_template)
        prompt = self._apply_chat_template_manual(messages)

        print(f"[NewBie Gemma Chat] All-in-One (KV-Cache): Generating response...")
        print(f"[NewBie Gemma Chat] Temperature: {temperature}, Max tokens: {max_new_tokens}")

        # 4. 分词
        input_ids = self._tokenize_with_spiece(tokenizer, prompt, device)
        print(f"[NewBie Gemma Chat] Input tokens: {input_ids.shape[1]}")

        # 5. 获取实际的 Llama2_ 模型用于 KV-Cache 生成
        if hasattr(transformer, 'model'):
            model = transformer.model
        else:
            model = transformer

        # 6. 使用 KV-Cache 版本生成 (大幅提速)
        generated_ids = self._generate_with_kv_cache(
            model, embed_weight, input_ids,
            max_new_tokens, temperature, top_p, top_k
        )

        # 7. 解码 (generated_ids 已经只是新生成的 tokens)
        response = self._decode_with_spiece(tokenizer, generated_ids)

        # 8. 清理响应
        response = response.replace("<end_of_turn>", "").strip()

        # 9. 构建对话历史
        full_conversation = self._build_conversation_history(user_message, response, conversation_history)

        print(f"[NewBie Gemma Chat] All-in-One (KV-Cache): Response generated ({len(response)} chars)")

        return response, full_conversation

    def _apply_chat_template_manual(self, messages: list) -> str:
        """
        手工实现 Gemma3 chat template
        用于 SPieceTokenizer 没有 apply_chat_template() 的情况
        """
        result = ""
        for message in messages:
            role = message['role']
            content = message['content']

            if role == 'user':
                result += f"<start_of_turn>user\n{content}<end_of_turn>\n"
            elif role == 'assistant':
                result += f"<start_of_turn>model\n{content}<end_of_turn>\n"

        # 添加生成提示
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
        """Extract model path from NewBie CLIP object."""
        # Try tokenizer.name_or_path first
        if hasattr(clip, 'tokenizer') and hasattr(clip.tokenizer, 'name_or_path'):
            return clip.tokenizer.name_or_path

        # Try text_encoder.name_or_path
        if hasattr(clip, 'text_encoder'):
            if hasattr(clip.text_encoder, 'name_or_path'):
                return clip.text_encoder.name_or_path
            if hasattr(clip.text_encoder, 'config') and hasattr(clip.text_encoder.config, '_name_or_path'):
                return clip.text_encoder.config._name_or_path

        raise ValueError("Cannot determine Gemma model path from NewBie CLIP.")

    def _get_device(self, clip) -> str:
        """Determine the device to use for the model."""
        if hasattr(clip, 'device'):
            return clip.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _get_dtype(self, clip, clip_type: str) -> torch.dtype:
        """Determine the dtype to use for the model."""
        if clip_type == "newbie_clip" and hasattr(clip, 'text_encoder'):
            return next(clip.text_encoder.parameters()).dtype

        if clip_type == "comfyui_dual_clip" and hasattr(clip, 'cond_stage_model'):
            gemma = getattr(clip.cond_stage_model, 'gemma', None)
            if gemma is not None and hasattr(gemma, 'dtype'):
                return gemma.dtype

        return torch.bfloat16

    def _load_generation_model(self, model_path: str, device: str, dtype: torch.dtype):
        """Load or retrieve cached Gemma model for generation"""
        global _gemma_gen_cache
        
        # Check if we can reuse cached model
        if (_gemma_gen_cache["model"] is not None and 
            _gemma_gen_cache["model_path"] == model_path and
            _gemma_gen_cache["device"] == device and
            _gemma_gen_cache["dtype"] == dtype):
            print(f"[NewBie Gemma Chat] Using cached generation model")
            return _gemma_gen_cache["model"], _gemma_gen_cache["tokenizer"]
        
        # Clear old cache
        if _gemma_gen_cache["model"] is not None:
            del _gemma_gen_cache["model"]
            del _gemma_gen_cache["tokenizer"]
            torch.cuda.empty_cache()
        
        print(f"[NewBie Gemma Chat] Loading Gemma generation model from: {model_path}")
        print(f"[NewBie Gemma Chat] Device: {device}, Dtype: {dtype}")
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for Gemma Chat")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        
        # Load model for causal LM (generation)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        model.eval()
        
        # Cache for reuse
        _gemma_gen_cache["model"] = model
        _gemma_gen_cache["tokenizer"] = tokenizer
        _gemma_gen_cache["model_path"] = model_path
        _gemma_gen_cache["device"] = device
        _gemma_gen_cache["dtype"] = dtype
        
        print(f"[NewBie Gemma Chat] Model loaded successfully")
        return model, tokenizer

    def _format_messages(self, system_prompt: str, user_message: str, conversation_history: str = "") -> list:
        """Format messages for Gemma chat template"""
        messages = []
        
        # Add system prompt if provided
        if system_prompt and system_prompt.strip():
            messages.append({
                "role": "user",
                "content": f"[System Instructions]\n{system_prompt.strip()}\n\n[End System Instructions]\n\nAcknowledge that you understand these instructions."
            })
            messages.append({
                "role": "assistant", 
                "content": "I understand and will follow these instructions."
            })
        
        # Parse and add conversation history if provided
        if conversation_history and conversation_history.strip():
            # Parse the history format: USER: ... ASSISTANT: ...
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
        
        # Add current user message
        if user_message and user_message.strip():
            messages.append({
                "role": "user",
                "content": user_message.strip()
            })
        
        return messages

    def _apply_chat_template(self, tokenizer, messages: list) -> str:
        """Apply Gemma3 chat template to messages"""
        # Try to use tokenizer's built-in chat template if available
        if hasattr(tokenizer, 'apply_chat_template'):
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                print(f"[NewBie Gemma Chat] Tokenizer chat template failed: {e}, using manual format")
        
        # Manual Gemma3 chat template
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

    def _build_generation_config(
        self,
        tokenizer,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
    ) -> dict:
        """Build generation configuration dictionary."""
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
        generation_config: dict,
    ) -> str:
        """Generate and decode response from the model."""
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
        """Build the full conversation history string."""
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
        """Generate a chat response from Gemma.

        Supports two modes:
        1. All-in-One mode (use_clip_model=True): Directly uses CLIP's embedded Gemma model
        2. External model mode (use_clip_model=False): Loads separate HuggingFace model
        """
        if not user_message or not user_message.strip():
            return "Please provide a message.", conversation_history

        clip_type = self._detect_clip_type(clip)

        # All-in-One mode: directly use CLIP's embedded Gemma model
        if use_clip_model and clip_type == "comfyui_dual_clip":
            print(f"[NewBie Gemma Chat] Using All-in-One mode (CLIP embedded model)")
            return self._chat_with_clip_model(
                clip, user_message, system_prompt,
                temperature, max_new_tokens, top_p, top_k,
                conversation_history
            )

        # External model mode: load HuggingFace model
        print(f"[NewBie Gemma Chat] Using external model mode")
        model_path, device, dtype = self._get_gemma_model_info(clip, gemma_model_path)
        model, tokenizer = self._load_generation_model(model_path, device, dtype)

        messages = self._format_messages(system_prompt, user_message, conversation_history)
        prompt = self._apply_chat_template(tokenizer, messages)

        print(f"[NewBie Gemma Chat] Generating response...")
        print(f"[NewBie Gemma Chat] Temperature: {temperature}, Max tokens: {max_new_tokens}")

        generation_config = self._build_generation_config(
            tokenizer, max_new_tokens, temperature, top_p, top_k
        )
        response = self._generate_response(model, tokenizer, prompt, device, generation_config)
        full_conversation = self._build_conversation_history(user_message, response, conversation_history)

        print(f"[NewBie Gemma Chat] Response generated ({len(response)} chars)")

        if unload_after:
            _clear_gemma_cache()

        return (response, full_conversation)


class NewBieGemmaChatAdvanced(NewBieGemmaChat):
    """
    Advanced Gemma Chat with more generation parameters and optional model override.
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
        """Set random seed for reproducibility."""
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
        """Generate a chat response from Gemma with advanced options.

        Supports two modes:
        1. All-in-One mode (use_clip_model=True): Directly uses CLIP's embedded Gemma model
           Note: repetition_penalty, do_sample, seed are not supported in All-in-One mode
        2. External model mode (use_clip_model=False): Loads separate HuggingFace model
        """
        if not user_message or not user_message.strip():
            return "Please provide a message.", conversation_history

        clip_type = self._detect_clip_type(clip)

        # All-in-One mode: directly use CLIP's embedded Gemma model
        if use_clip_model and clip_type == "comfyui_dual_clip":
            print(f"[NewBie Gemma Chat Advanced] Using All-in-One mode (CLIP embedded model)")
            if repetition_penalty != 1.0 or not do_sample or seed >= 0:
                print(f"[NewBie Gemma Chat Advanced] Warning: repetition_penalty, do_sample, seed are ignored in All-in-One mode")
            return self._chat_with_clip_model(
                clip, user_message, system_prompt,
                temperature, max_new_tokens, top_p, top_k,
                conversation_history
            )

        # External model mode: load HuggingFace model
        print(f"[NewBie Gemma Chat Advanced] Using external model mode")
        self._set_seed(seed)

        model_path, device, dtype = self._get_gemma_model_info(clip, gemma_model_path)
        model, tokenizer = self._load_generation_model(model_path, device, dtype)

        messages = self._format_messages(system_prompt, user_message, conversation_history)
        prompt = self._apply_chat_template(tokenizer, messages)

        print(f"[NewBie Gemma Chat Advanced] Generating response...")
        print(f"[NewBie Gemma Chat Advanced] Temperature: {temperature}, Max tokens: {max_new_tokens}, Seed: {seed}")

        generation_config = self._build_generation_config(
            tokenizer, max_new_tokens, temperature, top_p, top_k,
            do_sample=do_sample, repetition_penalty=repetition_penalty
        )
        response = self._generate_response(model, tokenizer, prompt, device, generation_config)
        full_conversation = self._build_conversation_history(user_message, response, conversation_history)

        print(f"[NewBie Gemma Chat Advanced] Response generated ({len(response)} chars)")

        if unload_after:
            _clear_gemma_cache()

        return (response, full_conversation)


NODE_CLASS_MAPPINGS = {
    "NewBieGemmaChat": NewBieGemmaChat,
    "NewBieGemmaChatAdvanced": NewBieGemmaChatAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NewBieGemmaChat": "NewBie Gemma Chat",
    "NewBieGemmaChatAdvanced": "NewBie Gemma Chat (Advanced)",
}
