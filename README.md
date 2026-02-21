# Canonical Parallel Reduction

WG21 proposal **P4016R0** — a deterministic parallel reduction facility for C++.

**Document:** P4016R0  
**Date:** 2026-02-19  
**Author:** Andrew Drakeford  
**Audience:** SG6 (Numerics), LEWG, SG1 (Concurrency), SG14 (Low Latency/Games/Embedded/Financial Trading)

## Summary

This paper specifies a **canonical reduction expression structure**: for a given input order and topology coordinate (lane count `L`), the expression — its parenthesization and operand order — is unique and fully specified. Implementations are free to schedule evaluation using parallelization, vectorization, or any other strategy, provided the returned value matches that of the specified expression.

The proposal closes the gap between `std::accumulate` (deterministic but sequential) and `std::reduce` (parallel but non-deterministic for floating-point operations).

**Semantics only.** API design is deferred; this paper seeks LEWG validation of the expression structure before committing to API surface.

## Paper

| Format | Link |
|--------|------|
| **HTML** | [P4016R0.html](https://andyD123.github.io/canonical-reduce/P4016R0.html) |
| **PDF** | [P4016R0.pdf](P4016R0.pdf) |
| **Markdown source** | [P4016R0.md](P4016R0.md) |

## Key Design Points

- **Two-stage reduction**: input distributed across `L` interleaved lanes by `i mod L`; each lane reduced by iterative pairwise (shift-reduce) tree; lane results combined by same tree rule.
- **Expression ownership**: the algorithm "owns" the tree — unlike `std::reduce` which permits arbitrary reassociation via `GENERALIZED_SUM`.
- **Lane count L** (not byte-span M) is the sole topology coordinate — avoids the portability trap where `sizeof(V)` varies across platforms.
- **Init placement**: `op(init, R)` — init combined once after tree reduction, not folded into the tree.

## Demonstrators

Working implementations with Compiler Explorer links are documented in Appendix K of the paper:

| Demonstrator | Platform | Link |
|---|---|---|
| Sequential reference | Portable | [GB-SEQ](https://godbolt.org/z/8EEhEqrz6) |
| x86 AVX2 | x86-64 | [GB-x86-AVX2](https://godbolt.org/z/Eaa3vWYqb) |
| Multi-threaded x86 | x86-64 | [GB-x86-MT](https://godbolt.org/z/7a11r9o95) |
| MT with thread pool | x86-64 | [GB-x86-MT-PERF](https://godbolt.org/z/sdxMohT48) |
| ARM NEON | AArch64 | [GB-NEON](https://godbolt.org/z/Pxzc3YM7q) |
| NEON 8-block unroll | AArch64 | [GB-NEON-PERF](https://godbolt.org/z/sY9W78rze) |
| CUDA/NVCC | GPU | [GB-CUDA](https://godbolt.org/z/5n9EvGoeb) |

**Golden reference values** (N=1M doubles, fixed seed):
- L=16 (NARROW): `0x40618f71f6379380`
- L=128 (WIDE): `0x40618f71f6379397`

## Building from source

Requires [pandoc](https://pandoc.org) and Python 3.

```bash
make P4016R0.html
```

## License

This document is submitted to WG21 (ISO/IEC JTC1/SC22/WG21) for consideration as a C++ standards proposal.
