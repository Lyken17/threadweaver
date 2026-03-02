"""Utility functions for generating multiplication examples with chain-of-thought reasoning."""

import random
from typing import List, Tuple


def multiply_parallel_chains(numbers: List[int], parallel: bool = False, p: float = None) -> Tuple[int, str]:
    """
    Multiply a list of numbers with chain-of-thought reasoning.

    Args:
        numbers: List of integers to multiply
        parallel: Whether to use parallel chain of thought style
        p: Probability of parallelizing steps (unused in simple version)

    Returns:
        Tuple of (final_product, reasoning_text)
    """
    if len(numbers) == 0:
        return 1, ""

    if len(numbers) == 1:
        return numbers[0], f"The number is simply {numbers[0]}."

    # Build chain of thought
    lines = []
    lines.append(f"I need to multiply: {' × '.join(map(str, numbers))}")

    # Compute step by step
    current = numbers[0]
    steps = []

    for i in range(1, len(numbers)):
        next_num = numbers[i]
        product = current * next_num

        if parallel:
            # Parallel style: show all intermediate steps
            steps.append(f"Step {i}: {current} × {next_num} = {product}")
        else:
            # Sequential style
            steps.append(f"{current} × {next_num} = {product}")

        current = product

    if parallel:
        lines.append("I'll compute this step by step:")
        lines.extend(steps)
    else:
        lines.append("Let me multiply sequentially:")
        lines.extend(steps)

    final_product = current
    lines.append(f"\nTherefore, the final answer is \\boxed{{{final_product}}}.")

    return final_product, "\n".join(lines)


def make_example(min_value: int, max_value: int, min_len: int, max_len: int,
                 rng: random.Random, parallel: bool = False, **kwargs) -> dict:
    """
    Generate a single multiplication example with chain-of-thought reasoning.

    Args:
        min_value: Minimum integer value (inclusive)
        max_value: Maximum integer value (inclusive)
        min_len: Minimum number of integers to multiply
        max_len: Maximum number of integers to multiply
        rng: Random number generator
        parallel: Whether to use parallel chain of thought style
        **kwargs: Additional arguments (e.g., p for probability)

    Returns:
        Dictionary with conversation format
    """
    # Generate random list of numbers
    chain_len = rng.randint(min_len, max_len)
    numbers = [rng.randint(min_value, max_value) for _ in range(chain_len)]

    # Get the chain of thought reasoning
    p = kwargs.get('p', None)
    final_product, reasoning = multiply_parallel_chains(numbers, parallel=parallel, p=p)

    # Format question
    question = f"What is {' × '.join(map(str, numbers))}?"

    # Format response
    response = reasoning

    # Return in conversation format
    return {
        "conversations": [
            {"from": "human", "value": question},
            {"from": "assistant", "value": response}
        ]
    }
