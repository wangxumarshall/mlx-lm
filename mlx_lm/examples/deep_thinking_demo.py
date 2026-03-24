#!/usr/bin/env python3
# Copyright © 2026 Apple Inc.

import argparse
import time

import mlx.core as mx

from mlx_lm import load, Qwen35DeepThinkingOptimizer, create_deep_thinking_config


def run_benchmark(model_path: str, prompt: str, max_tokens: int = 256):
    print(f"\n{'='*70}")
    print("Qwen3.5 Deep-Thinking Optimization Benchmark")
    print(f"{'='*70}")
    
    print(f"\n[1/3] Loading model: {model_path}")
    model, tokenizer = load(model_path)
    
    config = create_deep_thinking_config(
        jsd_threshold=0.08,
        deep_layer_ratio=0.75,
        min_prefix_tokens=48,
        dtr_threshold=0.12,
        early_stop_patience=2,
    )
    
    optimizer = Qwen35DeepThinkingOptimizer(model, tokenizer, config)
    
    print(f"\n[2/3] Standard Generation (baseline)")
    print("-" * 50)
    
    start_time = time.time()
    standard_result = optimizer.generate_optimized(
        prompt,
        max_tokens=max_tokens,
        temp=0.7,
        enable_early_stop=False,
    )
    standard_time = time.time() - start_time
    
    print(f"Generated {len(standard_result.tokens)} tokens in {standard_time:.2f}s")
    print(f"Deep-Thinking Ratio: {standard_result.deep_thinking_ratio:.3f}")
    print(f"Tokens/Second: {len(standard_result.tokens) / standard_time:.1f}")
    
    print(f"\n[3/3] DTR-Optimized Generation")
    print("-" * 50)
    
    start_time = time.time()
    optimized_result = optimizer.generate_optimized(
        prompt,
        max_tokens=max_tokens,
        temp=0.7,
        enable_early_stop=True,
    )
    optimized_time = time.time() - start_time
    
    print(f"Generated {len(optimized_result.tokens)} tokens in {optimized_time:.2f}s")
    print(f"Deep-Thinking Ratio: {optimized_result.deep_thinking_ratio:.3f}")
    print(f"Tokens/Second: {len(optimized_result.tokens) / optimized_time:.1f}")
    print(f"Early Stopped: {optimized_result.early_stopped}")
    
    speedup = standard_time / optimized_time if optimized_time > 0 else 1.0
    
    print(f"\n{'='*70}")
    print("BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"Standard Generation:  {standard_time:.2f}s ({len(standard_result.tokens)} tokens)")
    print(f"Optimized Generation: {optimized_time:.2f}s ({len(optimized_result.tokens)} tokens)")
    print(f"\n🚀 Speedup: {speedup:.2f}x ({(speedup-1)*100:.1f}% improvement)")
    print(f"{'='*70}")
    
    print("\n[Sample Output]")
    print("-" * 50)
    print(optimized_result.text[:500] + "..." if len(optimized_result.text) > 500 else optimized_result.text)
    
    return speedup


def run_self_consistency(model_path: str, prompt: str, num_samples: int = 8, max_tokens: int = 256):
    print(f"\n{'='*70}")
    print("Qwen3.5 Think@n Self-Consistency Demo")
    print(f"{'='*70}")
    
    print(f"\n[1/2] Loading model: {model_path}")
    model, tokenizer = load(model_path)
    
    config = create_deep_thinking_config(
        jsd_threshold=0.08,
        deep_layer_ratio=0.75,
        min_prefix_tokens=48,
        dtr_threshold=0.12,
    )
    
    optimizer = Qwen35DeepThinkingOptimizer(model, tokenizer, config)
    
    print(f"\n[2/2] Generating {num_samples} samples with DTR-based priority...")
    print("-" * 50)
    
    start_time = time.time()
    answer, results = optimizer.generate_with_self_consistency(
        prompt,
        num_samples=num_samples,
        max_tokens=max_tokens,
        temp=0.7,
        use_dtr_priority=True,
    )
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print("SELF-CONSISTENCY RESULTS")
    print(f"{'='*70}")
    
    print("\n[Top 3 Samples by DTR]")
    for i, r in enumerate(results[:3]):
        print(f"\nSample {i+1}:")
        print(f"  DTR: {r.deep_thinking_ratio:.3f}")
        print(f"  Tokens: {len(r.tokens)}")
        print(f"  Time: {r.generation_time:.2f}s")
        print(f"  Early Stopped: {r.early_stopped}")
    
    print(f"\n[Final Answer]")
    print("-" * 50)
    print(answer)
    
    print(f"\n[Statistics]")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Samples Generated: {len(results)}")
    print(f"Early Stops: {sum(1 for r in results if r.early_stopped)}")
    
    return answer


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5 Deep-Thinking Optimization Demo")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen3-8B-4bit",
        help="Model path or HuggingFace repo",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Solve this math problem step by step: If a train travels at 60 mph for 2.5 hours, then stops for 30 minutes, and then travels at 80 mph for 1.5 hours, what is the total distance traveled?",
        help="Input prompt",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["benchmark", "self-consistency", "both"],
        default="benchmark",
        help="Run mode",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of samples for self-consistency",
    )
    
    args = parser.parse_args()
    
    if args.mode in ["benchmark", "both"]:
        run_benchmark(args.model, args.prompt, args.max_tokens)
    
    if args.mode in ["self-consistency", "both"]:
        run_self_consistency(args.model, args.prompt, args.num_samples, args.max_tokens)


if __name__ == "__main__":
    main()
