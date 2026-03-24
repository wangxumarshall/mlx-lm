# Copyright © 2026 Apple Inc.

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .deep_thinking import (
    DeepThinkingConfig,
    DeepThinkingResult,
    DeepThinkingTracker,
    ThinkAtN,
    compute_deep_thinking_ratio,
    compute_token_convergence_layer,
    create_deep_thinking_config,
    extract_layer_hidden_states,
    hidden_states_to_logits,
    js_divergence,
)
from .generate import generate_step, stream_generate
from .sample_utils import make_sampler
from .utils import load


@dataclass
class OptimizedGenerationResult:
    text: str
    tokens: List[int]
    deep_thinking_ratio: float
    generation_time: float
    tokens_per_second: float
    early_stopped: bool
    priority_score: float


class Qwen35DeepThinkingOptimizer:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: Optional[DeepThinkingConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or create_deep_thinking_config(
            jsd_threshold=0.08,
            deep_layer_ratio=0.75,
            min_prefix_tokens=48,
            dtr_threshold=0.12,
            early_stop_patience=2,
        )
        
        self._num_layers = self._get_num_layers()
        self._sample_layers = self._compute_sample_layers()
    
    def _get_num_layers(self) -> int:
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return len(self.model.model.layers)
        elif hasattr(self.model, 'layers'):
            return len(self.model.layers)
        else:
            raise ValueError("Cannot determine number of layers")
    
    def _compute_sample_layers(self) -> List[int]:
        num_layers = self._num_layers
        deep_start = int(num_layers * self.config.deep_layer_ratio)
        
        sample_layers = [
            0,
            num_layers // 4,
            num_layers // 2,
            3 * num_layers // 4,
            deep_start,
            num_layers - 2,
            num_layers - 1,
        ]
        
        return sorted(list(set(sample_layers)))
    
    def _extract_sampled_hidden_states(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> Dict[int, mx.array]:
        hidden_states = {}
        
        if hasattr(self.model, 'model'):
            inner_model = self.model.model
        else:
            inner_model = self.model
        
        if hasattr(inner_model, 'embed_tokens'):
            h = inner_model.embed_tokens(inputs)
        elif hasattr(inner_model, 'model') and hasattr(inner_model.model, 'embed_tokens'):
            h = inner_model.model.embed_tokens(inputs)
        else:
            raise ValueError("Cannot find embedding layer")
        
        if cache is None:
            cache = [None] * len(inner_model.layers)
        
        for i, (layer, c) in enumerate(zip(inner_model.layers, cache)):
            h = layer(h, cache=c)
            if i in self._sample_layers:
                hidden_states[i] = h
        
        if hasattr(inner_model, 'norm'):
            hidden_states[self._num_layers - 1] = inner_model.norm(h)
        
        return hidden_states
    
    def _compute_fast_dtr(
        self,
        tokens: mx.array,
        cache: Optional[Any] = None,
    ) -> float:
        hidden_states = self._extract_sampled_hidden_states(tokens, cache)
        
        if not hidden_states:
            return 0.0
        
        final_layer = max(hidden_states.keys())
        final_h = hidden_states[final_layer]
        final_logits = hidden_states_to_logits(final_h, self.model)
        final_probs = mx.softmax(final_logits[0, -1, :], axis=-1)
        
        deep_tokens = 0
        total_tokens = tokens.shape[-1] if len(tokens.shape) > 1 else 1
        
        deep_layer_start = int(self._num_layers * self.config.deep_layer_ratio)
        sample_deep_layers = [l for l in self._sample_layers if l >= deep_layer_start]
        
        if not sample_deep_layers:
            sample_deep_layers = [final_layer]
        
        for layer_idx in sample_deep_layers[:-1]:
            if layer_idx not in hidden_states:
                continue
            
            h = hidden_states[layer_idx]
            layer_logits = hidden_states_to_logits(h, self.model)
            layer_probs = mx.softmax(layer_logits[0, -1, :], axis=-1)
            
            jsd = js_divergence(layer_probs, final_probs)
            
            if jsd > self.config.jsd_threshold:
                deep_tokens += 1
                break
        
        return deep_tokens / max(1, len(sample_deep_layers))
    
    def generate_optimized(
        self,
        prompt: Union[str, mx.array, List[int]],
        max_tokens: int = 512,
        temp: float = 0.7,
        top_p: float = 0.9,
        enable_early_stop: bool = True,
        verbose: bool = False,
        **kwargs,
    ) -> OptimizedGenerationResult:
        start_time = time.time()
        
        if isinstance(prompt, str):
            tokens = mx.array(self.tokenizer.encode(prompt))
        elif isinstance(prompt, list):
            tokens = mx.array(prompt)
        else:
            tokens = prompt
        
        sampler = make_sampler(temp=temp, top_p=top_p)
        tracker = DeepThinkingTracker(self.config)
        
        generated_tokens = []
        early_stopped = False
        dtr_check_interval = 32
        
        for i, (token, logprobs) in enumerate(
            generate_step(
                tokens,
                self.model,
                max_tokens=max_tokens,
                sampler=sampler,
                **kwargs
            )
        ):
            generated_tokens.append(token)
            
            if enable_early_stop and len(generated_tokens) >= self.config.min_prefix_tokens:
                if len(generated_tokens) % dtr_check_interval == 0:
                    prefix_tokens = mx.array(generated_tokens)
                    dtr = self._compute_fast_dtr(prefix_tokens)
                    
                    if tracker.update(dtr, len(generated_tokens)):
                        early_stopped = True
                        if verbose:
                            print(f"[Early stop] DTR={dtr:.3f} at token {len(generated_tokens)}")
                        break
        
        full_tokens = mx.array(generated_tokens)
        final_dtr = self._compute_fast_dtr(full_tokens)
        
        generated_text = self.tokenizer.decode(generated_tokens)
        
        generation_time = time.time() - start_time
        tokens_per_second = len(generated_tokens) / generation_time if generation_time > 0 else 0
        
        return OptimizedGenerationResult(
            text=generated_text,
            tokens=generated_tokens,
            deep_thinking_ratio=final_dtr,
            generation_time=generation_time,
            tokens_per_second=tokens_per_second,
            early_stopped=early_stopped,
            priority_score=tracker.get_priority_score(),
        )
    
    def generate_with_self_consistency(
        self,
        prompt: Union[str, mx.array, List[int]],
        num_samples: int = 8,
        max_tokens: int = 512,
        temp: float = 0.7,
        top_p: float = 0.9,
        use_dtr_priority: bool = True,
        verbose: bool = False,
        **kwargs,
    ) -> Tuple[str, List[OptimizedGenerationResult]]:
        results = []
        
        for i in range(num_samples):
            if verbose:
                print(f"\n[Sample {i+1}/{num_samples}]")
            
            result = self.generate_optimized(
                prompt,
                max_tokens=max_tokens,
                temp=temp,
                top_p=top_p,
                enable_early_stop=use_dtr_priority,
                verbose=verbose,
                **kwargs
            )
            results.append(result)
        
        if use_dtr_priority:
            results.sort(key=lambda x: x.deep_thinking_ratio, reverse=True)
        
        answer_counts: Dict[str, int] = {}
        for r in results:
            answer = r.text.strip()
            answer_counts[answer] = answer_counts.get(answer, 0) + 1
        
        most_common = max(answer_counts.items(), key=lambda x: x[1])[0]
        
        return most_common, results


def setup_arg_parser():
    parser = argparse.ArgumentParser(
        description="Optimized generation with Deep-Thinking Tokens for Qwen3.5"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen3-8B-4bit",
        help="Path to the model or Hugging Face repo",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of samples for self-consistency",
    )
    parser.add_argument(
        "--jsd-threshold",
        type=float,
        default=0.08,
        help="JSD threshold for deep-thinking token detection",
    )
    parser.add_argument(
        "--dtr-threshold",
        type=float,
        default=0.12,
        help="DTR threshold for early stopping",
    )
    parser.add_argument(
        "--min-prefix-tokens",
        type=int,
        default=48,
        help="Minimum tokens before DTR evaluation",
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable early stopping based on DTR",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark comparison",
    )
    return parser


def benchmark_comparison(
    optimizer: Qwen35DeepThinkingOptimizer,
    prompt: str,
    max_tokens: int = 512,
    temp: float = 0.7,
    num_runs: int = 3,
):
    import statistics
    
    print("\n" + "="*60)
    print("BENCHMARK: Standard vs DTR-Optimized Generation")
    print("="*60)
    
    standard_times = []
    standard_tokens = []
    
    print("\n[Standard Generation]")
    for i in range(num_runs):
        start = time.time()
        result = optimizer.generate_optimized(
            prompt,
            max_tokens=max_tokens,
            temp=temp,
            enable_early_stop=False,
        )
        elapsed = time.time() - start
        standard_times.append(elapsed)
        standard_tokens.append(len(result.tokens))
        print(f"  Run {i+1}: {elapsed:.2f}s, {len(result.tokens)} tokens, DTR={result.deep_thinking_ratio:.3f}")
    
    optimized_times = []
    optimized_tokens = []
    early_stops = 0
    
    print("\n[DTR-Optimized Generation]")
    for i in range(num_runs):
        result = optimizer.generate_optimized(
            prompt,
            max_tokens=max_tokens,
            temp=temp,
            enable_early_stop=True,
        )
        optimized_times.append(result.generation_time)
        optimized_tokens.append(len(result.tokens))
        if result.early_stopped:
            early_stops += 1
        print(f"  Run {i+1}: {result.generation_time:.2f}s, {len(result.tokens)} tokens, DTR={result.deep_thinking_ratio:.3f}, early_stop={result.early_stopped}")
    
    avg_standard = statistics.mean(standard_times)
    avg_optimized = statistics.mean(optimized_times)
    speedup = avg_standard / avg_optimized if avg_optimized > 0 else 1.0
    
    print("\n" + "-"*60)
    print("RESULTS SUMMARY")
    print("-"*60)
    print(f"Standard Generation:")
    print(f"  Avg Time: {avg_standard:.2f}s")
    print(f"  Avg Tokens: {statistics.mean(standard_tokens):.1f}")
    print(f"\nDTR-Optimized Generation:")
    print(f"  Avg Time: {avg_optimized:.2f}s")
    print(f"  Avg Tokens: {statistics.mean(optimized_tokens):.1f}")
    print(f"  Early Stops: {early_stops}/{num_runs}")
    print(f"\n🚀 Speedup: {speedup:.2f}x ({(speedup-1)*100:.1f}% faster)")
    print("="*60)
    
    return speedup


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)
    
    config = create_deep_thinking_config(
        jsd_threshold=args.jsd_threshold,
        deep_layer_ratio=0.75,
        min_prefix_tokens=args.min_prefix_tokens,
        dtr_threshold=args.dtr_threshold,
        early_stop_patience=2,
    )
    
    optimizer = Qwen35DeepThinkingOptimizer(model, tokenizer, config)
    
    if args.benchmark:
        benchmark_comparison(
            optimizer,
            args.prompt,
            max_tokens=args.max_tokens,
            temp=args.temp,
            num_runs=3,
        )
        return
    
    if args.num_samples > 1:
        print(f"\nGenerating {args.num_samples} samples with self-consistency...")
        answer, results = optimizer.generate_with_self_consistency(
            args.prompt,
            num_samples=args.num_samples,
            max_tokens=args.max_tokens,
            temp=args.temp,
            top_p=args.top_p,
            use_dtr_priority=not args.no_early_stop,
            verbose=args.verbose,
        )
        
        print("\n" + "="*60)
        print("FINAL ANSWER (Self-Consistency)")
        print("="*60)
        print(answer)
        
        print("\n" + "-"*60)
        print("SAMPLE STATISTICS")
        print("-"*60)
        for i, r in enumerate(results[:3]):
            print(f"Sample {i+1}: DTR={r.deep_thinking_ratio:.3f}, "
                  f"Tokens={len(r.tokens)}, Time={r.generation_time:.2f}s, "
                  f"EarlyStop={r.early_stopped}")
    else:
        result = optimizer.generate_optimized(
            args.prompt,
            max_tokens=args.max_tokens,
            temp=args.temp,
            top_p=args.top_p,
            enable_early_stop=not args.no_early_stop,
            verbose=args.verbose,
        )
        
        print("\n" + "="*60)
        print("GENERATION RESULT")
        print("="*60)
        print(result.text)
        print("\n" + "-"*60)
        print("STATISTICS")
        print("-"*60)
        print(f"Deep-Thinking Ratio: {result.deep_thinking_ratio:.3f}")
        print(f"Tokens Generated: {len(result.tokens)}")
        print(f"Generation Time: {result.generation_time:.2f}s")
        print(f"Tokens/Second: {result.tokens_per_second:.1f}")
        print(f"Early Stopped: {result.early_stopped}")


if __name__ == "__main__":
    main()
