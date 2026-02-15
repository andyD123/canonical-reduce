---
title: "Canonical Reduce"
document: DxxxxR0
date: today
audience: LEWG
author:
  - name: Andy Dawes
toc: false
---

# Abstract

This paper proposes `canonical_reduce`, a standardized interface for reduction operations on ranges in C++.

# Motivation

Reduction operations are a fundamental pattern in programming, yet C++ lacks a unified, composable interface for expressing them. The current standard library provides `std::accumulate`, `std::reduce`, and various specialized algorithms, but they are disconnected and don't work seamlessly with modern ranges.

`canonical_reduce` aims to provide:

1. A consistent interface for reduction operations
2. Full compatibility with C++20 ranges
3. Support for parallel execution policies
4. Composability with other range algorithms

# Proposed Solution

We propose adding `std::ranges::canonical_reduce` to the standard library:

```cpp
namespace std::ranges {
  template<input_range R, class T, class BinaryOp>
  constexpr auto canonical_reduce(R&& r, T init, BinaryOp binary_op);
  
  template<input_range R, class BinaryOp>
  constexpr auto canonical_reduce(R&& r, BinaryOp binary_op);
}
```

# Design Considerations

The design follows these principles:

- **Range-based**: Works naturally with C++20 ranges
- **Composable**: Can be piped with other range operations
- **Efficient**: Supports parallel execution where appropriate
- **Flexible**: Handles different reduction patterns

# Examples

```cpp
// Simple sum
std::vector<int> nums = {1, 2, 3, 4, 5};
auto sum = std::ranges::canonical_reduce(nums, 0, std::plus{});

// With ranges
auto result = nums 
  | std::views::filter([](int x) { return x > 2; })
  | std::views::transform([](int x) { return x * 2; })
  | std::ranges::to<std::vector>()
  | std::ranges::canonical_reduce(0, std::plus{});
```

# Impact on the Standard

This proposal adds new functionality without breaking existing code. It integrates naturally with the ranges library introduced in C++20.

# Acknowledgments

Thanks to the C++ community for their input and feedback on this proposal.
