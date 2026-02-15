# Canonical Parallel Reduction

WG21 proposal for a deterministic parallel reduction facility in C++.

**Document:** DxxxxR0 (P-number pending)
**Author:** Andrew Drakeford
**Audience:** LEWG

## Summary

This paper specifies a canonical reduction expression structure — a fixed parenthesization and operand order for parallel reduction, parameterised by a topology coordinate. It closes the gap between `std::accumulate` (deterministic but sequential) and `std::reduce` (parallel but non-deterministic).

Semantics only. API design is deferred.

## Paper

[Read the paper (HTML)](https://andyD123.github.io/canonical-reduce/generated/DxxxxR0.html) | [PDF](generated/DxxxxR0.pdf)

## Building

Requires [pandoc](https://pandoc.org) and Python 3.

```
make DxxxxR0.html
```

## Demonstrators

Working implementations with Compiler Explorer links are documented in Appendix K of the paper, covering x86 AVX2, ARM NEON, and CUDA.
