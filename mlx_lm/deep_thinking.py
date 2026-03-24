# Copyright © 2026 Apple Inc.

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .models.base import create_attention_mask


@dataclass
class DeepThinkingConfig:
    jsd_threshold: float = 0.1
    deep_layer_ratio: float = 0.8
    min_prefix_tokens: int = 64
    dtr_threshold: float = 0.15
    early_stop_patience: int = 3
    sample_layers: Optional[List[int]] = None


@dataclass
class DeepThinkingResult:
    deep_thinking_ratio: float
    deep_thinking_tokens: List[int]
    layer_convergence: List[int]
    jsd_scores: List[List[float]]
    total_tokens: int


def js_divergence(p: mx.array, q: mx.array, eps: float = 1e-8) -> mx.array:
    p = mx.clip(p, eps, 1.0)
    q = mx.clip(q, eps, 1.0)
    p = p / mx.sum(p, axis=-1, keepdims=True)
    q = q / mx.sum(q, axis=-1, keepdims=True)
    m = 0.5 * (p + q)
    kl_pm = mx.sum(p * mx.log(p / m), axis=-1)
    kl_qm = mx.sum(q * mx.log(q / m), axis=-1)
    return 0.5 * (kl_pm + kl_qm)


def extract_layer_hidden_states(
    model: nn.Module,
    inputs: mx.array,
    cache: Optional[Any] = None,
    layer_indices: Optional[List[int]] = None,
) -> List[mx.array]:
    hidden_states = []
    
    if hasattr(model, 'model'):
        inner_model = model.model
    else:
        inner_model = model
    
    if len(inputs.shape) == 1:
        inputs = inputs[None, :]
    
    if hasattr(inner_model, 'embed_tokens'):
        h = inner_model.embed_tokens(inputs)
    elif hasattr(inner_model, 'model') and hasattr(inner_model.model, 'embed_tokens'):
        h = inner_model.model.embed_tokens(inputs)
    else:
        raise ValueError("Cannot find embedding layer in model")
    
    if cache is None:
        cache = [None] * len(inner_model.layers)
    
    all_layer_indices = layer_indices or list(range(len(inner_model.layers)))
    max_layer = max(all_layer_indices) if all_layer_indices else 0
    
    for i, (layer, c) in enumerate(zip(inner_model.layers, cache)):
        mask = create_attention_mask(h, c)
        h = layer(h, mask=mask, cache=c)
        if i in all_layer_indices or i == len(inner_model.layers) - 1:
            hidden_states.append(h)
    
    if hasattr(inner_model, 'norm'):
        final_hidden = inner_model.norm(h)
    else:
        final_hidden = h
    
    if not hidden_states or hidden_states[-1] is not final_hidden:
        hidden_states.append(final_hidden)
    
    return hidden_states


def hidden_states_to_logits(
    hidden_states: mx.array,
    model: nn.Module,
) -> mx.array:
    if hasattr(model, 'lm_head'):
        return model.lm_head(hidden_states)
    elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        return model.model.embed_tokens.as_linear(hidden_states)
    else:
        raise ValueError("Cannot find lm_head in model")


def compute_token_convergence_layer(
    jsd_scores: List[float],
    threshold: float,
    total_layers: int,
    deep_layer_ratio: float = 0.8,
) -> Tuple[int, bool]:
    deep_layer_start = int(total_layers * deep_layer_ratio)
    
    for layer_idx, jsd in enumerate(jsd_scores):
        if jsd < threshold:
            return layer_idx, layer_idx >= deep_layer_start
    
    return total_layers - 1, True


def compute_deep_thinking_ratio(
    model: nn.Module,
    tokens: mx.array,
    config: DeepThinkingConfig,
    cache: Optional[Any] = None,
) -> DeepThinkingResult:
    num_layers = len(model.model.layers) if hasattr(model, 'model') else len(model.layers)
    
    if config.sample_layers is None:
        sample_interval = max(1, num_layers // 8)
        config.sample_layers = list(range(0, num_layers, sample_interval)) + [num_layers - 1]
    
    hidden_states = extract_layer_hidden_states(
        model, tokens, cache, config.sample_layers
    )
    
    final_logits = hidden_states_to_logits(hidden_states[-1], model)
    final_probs = mx.softmax(final_logits, axis=-1)
    
    all_jsd_scores = []
    deep_thinking_tokens = []
    layer_convergence = []
    
    seq_len = tokens.shape[-1] if len(tokens.shape) > 1 else 1
    
    for layer_idx, h in enumerate(hidden_states[:-1]):
        layer_logits = hidden_states_to_logits(h, model)
        layer_probs = mx.softmax(layer_logits, axis=-1)
        
        jsd = js_divergence(
            layer_probs.reshape(-1, layer_probs.shape[-1]),
            final_probs.reshape(-1, final_probs.shape[-1])
        )
        all_jsd_scores.append(jsd.tolist() if hasattr(jsd, 'tolist') else [float(jsd)])
    
    all_jsd_scores = list(zip(*all_jsd_scores))
    
    for token_idx, token_jsd_scores in enumerate(all_jsd_scores):
        conv_layer, is_deep = compute_token_convergence_layer(
            token_jsd_scores,
            config.jsd_threshold,
            num_layers,
            config.deep_layer_ratio
        )
        layer_convergence.append(conv_layer)
        if is_deep:
            deep_thinking_tokens.append(token_idx)
    
    dtr = len(deep_thinking_tokens) / len(all_jsd_scores) if all_jsd_scores else 0.0
    
    return DeepThinkingResult(
        deep_thinking_ratio=dtr,
        deep_thinking_tokens=deep_thinking_tokens,
        layer_convergence=layer_convergence,
        jsd_scores=all_jsd_scores,
        total_tokens=len(all_jsd_scores)
    )


class DeepThinkingTracker:
    def __init__(self, config: DeepThinkingConfig):
        self.config = config
        self.prefix_dtr_scores: List[float] = []
        self.current_dtr = 0.0
        self.low_dtr_count = 0
        self.should_stop = False
    
    def update(self, dtr: float, tokens_generated: int) -> bool:
        self.current_dtr = dtr
        
        if tokens_generated >= self.config.min_prefix_tokens:
            self.prefix_dtr_scores.append(dtr)
            
            if dtr < self.config.dtr_threshold:
                self.low_dtr_count += 1
                if self.low_dtr_count >= self.config.early_stop_patience:
                    self.should_stop = True
            else:
                self.low_dtr_count = 0
        
        return self.should_stop
    
    def get_priority_score(self) -> float:
        if not self.prefix_dtr_scores:
            return 0.0
        return sum(self.prefix_dtr_scores) / len(self.prefix_dtr_scores)


class ThinkAtN:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: DeepThinkingConfig,
        num_samples: int = 8,
        prefix_tokens: int = 128,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.num_samples = num_samples
        self.prefix_tokens = prefix_tokens
        self.trackers: List[DeepThinkingTracker] = []
    
    def generate_with_dtr(
        self,
        prompt: Union[str, mx.array, List[int]],
        max_tokens: int = 512,
        temp: float = 0.7,
        **kwargs,
    ) -> Tuple[str, DeepThinkingResult]:
        from .generate import generate_step, make_sampler
        
        if isinstance(prompt, str):
            tokens = mx.array(self.tokenizer.encode(prompt))
        elif isinstance(prompt, list):
            tokens = mx.array(prompt)
        else:
            tokens = prompt
        
        sampler = make_sampler(temp=temp)
        
        generated_tokens = []
        all_logits = []
        
        for token, logprobs in generate_step(
            tokens,
            self.model,
            max_tokens=max_tokens,
            sampler=sampler,
            **kwargs
        ):
            generated_tokens.append(token)
            all_logits.append(logprobs)
        
        full_tokens = mx.array(generated_tokens)
        
        dtr_result = compute_deep_thinking_ratio(
            self.model,
            full_tokens,
            self.config
        )
        
        generated_text = self.tokenizer.decode(generated_tokens)
        
        return generated_text, dtr_result
    
    def generate_batch_with_priority(
        self,
        prompt: Union[str, mx.array, List[int]],
        max_tokens: int = 512,
        temp: float = 0.7,
        **kwargs,
    ) -> List[Tuple[str, DeepThinkingResult, float]]:
        results = []
        
        for _ in range(self.num_samples):
            text, dtr_result = self.generate_with_dtr(
                prompt, max_tokens, temp, **kwargs
            )
            priority = dtr_result.deep_thinking_ratio
            results.append((text, dtr_result, priority))
        
        results.sort(key=lambda x: x[2], reverse=True)
        
        return results
    
    def early_stop_generate(
        self,
        prompt: Union[str, mx.array, List[int]],
        max_tokens: int = 512,
        temp: float = 0.7,
        **kwargs,
    ) -> Tuple[str, DeepThinkingResult, bool]:
        from .generate import generate_step, make_sampler
        
        if isinstance(prompt, str):
            tokens = mx.array(self.tokenizer.encode(prompt))
        elif isinstance(prompt, list):
            tokens = mx.array(prompt)
        else:
            tokens = prompt
        
        sampler = make_sampler(temp=temp)
        tracker = DeepThinkingTracker(self.config)
        
        generated_tokens = []
        early_stopped = False
        
        for i, (token, logprobs) in enumerate(
            generate_step(tokens, self.model, max_tokens=max_tokens, sampler=sampler, **kwargs)
        ):
            generated_tokens.append(token)
            
            if len(generated_tokens) >= self.config.min_prefix_tokens:
                if len(generated_tokens) % 32 == 0:
                    prefix_tokens = mx.array(generated_tokens)
                    dtr_result = compute_deep_thinking_ratio(
                        self.model,
                        prefix_tokens,
                        self.config
                    )
                    
                    if tracker.update(dtr_result.deep_thinking_ratio, len(generated_tokens)):
                        early_stopped = True
                        break
        
        full_tokens = mx.array(generated_tokens)
        final_dtr = compute_deep_thinking_ratio(self.model, full_tokens, self.config)
        
        generated_text = self.tokenizer.decode(generated_tokens)
        
        return generated_text, final_dtr, early_stopped


def create_deep_thinking_config(
    jsd_threshold: float = 0.1,
    deep_layer_ratio: float = 0.8,
    min_prefix_tokens: int = 64,
    dtr_threshold: float = 0.15,
    early_stop_patience: int = 3,
) -> DeepThinkingConfig:
    return DeepThinkingConfig(
        jsd_threshold=jsd_threshold,
        deep_layer_ratio=deep_layer_ratio,
        min_prefix_tokens=min_prefix_tokens,
        dtr_threshold=dtr_threshold,
        early_stop_patience=early_stop_patience,
    )
