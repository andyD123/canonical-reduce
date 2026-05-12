# Canonical Parallel Reduction and Scan

WG21 proposals for deterministic parallel reduction (**P4016R0**) and parallel scan (**P4229R0**) facilities for C++.

**Author:** Andrew Drakeford
**Audience:** SG6 (Numerics), LEWG, SG1 (Concurrency), SG14 (Low Latency/Games/Embedded/Financial Trading)

## Overview

These two papers address the determinism gap in C++ parallel numeric algorithms. P4016R0 specifies a canonical expression structure for parallel reduction. P4229R0 extends the model to parallel scan, introducing the distinction between expression contracts and observation contracts.

---

## P4016R0 — Canonical Parallel Reduction

**Document:** P4016R0
**Date:** 2026-02-19

### Summary

This paper specifies a **canonical reduction expression structure**: for a given input order and topology coordinate (lane count `L`), the expression — its parenthesization and operand order — is unique and fully specified. Implementations are free to schedule evaluation using parallelization, vectorization, or any other strategy, provided the returned value matches that of the specified expression.

The proposal closes the gap between `std::accumulate` (deterministic but sequential) and `std::reduce` (parallel but non-deterministic for floating-point operations).

**Semantics only.** API design is deferred; this paper seeks LEWG validation of the expression structure before committing to API surface.

### Paper

| Format | Link |
|--------|------|
| **HTML** | [P4016R0.html](https://andyD123.github.io/canonical-reduce/P4016R0.html) |
| **PDF** | [P4016R0.pdf](P4016R0.pdf) |
| **Markdown source** | [P4016R0.md](P4016R0.md) |

### Key Design Points

- **Two-stage reduction**: input distributed across `L` interleaved lanes by `i mod L`; each lane reduced by iterative pairwise (shift-reduce) tree; lane results combined by same tree rule.
- **Expression ownership**: the algorithm "owns" the tree — unlike `std::reduce` which permits arbitrary reassociation via `GENERALIZED_SUM`.
- **Lane count L** (not byte-span M) is the sole topology coordinate — avoids the portability trap where `sizeof(V)` varies across platforms.
- **Init placement**: `op(init, R)` — init combined once after tree reduction, not folded into the tree.

### Demonstrators

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

---

## P4229R0 — Canonical Parallel Scan

**Document:** P4229R0
**Date:** 2026-05-12

### Summary

This paper extends the P4016R0 model from parallel reduction to parallel scan. Where reduction observes the **root** of a named expression, scan observes the **prefixes** of the same expression family. A reproducible scan therefore requires two contracts: an expression contract (what abstract expression defines the values) and an observation contract (which values of that expression are returned).

The paper proposes named, selectable expression policies — `left_fold`, `pairwise`, `iterated_pairwise`, `block_dyadic<B>`, and implementation-defined deterministic policies — and distinguishes **reproducibility** (same algorithm, same result) from **consistency** (`scan[i] == reduce(prefix)` under the same named expression).

**Direction paper for SG6 and SG1.** API and wording are deferred pending feedback on the expression/observation abstraction.

### Paper

| Format | Link |
|--------|------|
| **HTML** | [P4229R0.html](https://andyD123.github.io/canonical-reduce/P4229R0.html) |
| **PDF** | [P4229R0.pdf](P4229R0.pdf) |
| **Markdown source** | [P4229R0.md](P4229R0.md) |

### Key Design Points

- **Expression is not execution**: a named expression specifies the value to be observed; the implementation chooses scheduling, tiling, vectorization, and buffering strategy freely.
- **Reduce observes the root; scan observes prefixes**: `reduce<E>(xs) = root_observation(E(xs))`, `scan<E>(xs) = prefix_observations(E(xs))`.
- **Scan/reduce consistency** is a separate, selectable property: for policy `E`, `inclusive_scan<E>(xs)[i] == reduce<E>(xs[0..i+1))`.
- **Returned values are not expression state**: a scan output is an observation, not necessarily composable internal state.
- **Portable vs implementation-defined policies**: portable named expressions support cross-implementation reproducibility; implementation-defined policies support target-shaped deterministic throughput.

### Demonstrators

Working implementations and witness artifacts for the expression/observation model, documented in Appendices C through F of the paper:

| Demonstrator | Platform | Expression | Appendix | Link |
|---|---|---|---|---|
| P4016 canonical CUDA scan/reduce witness | CUDA | `iterated_pairwise` | C | [GB-CUDA-PAIRWISE](https://godbolt.org/z/qMa9bs3zh) |
| `canonical_block_dyadic` CUDA witness | CUDA (Tesla T4) | `canonical_block_dyadic` | D | [GB-CUDA-BLOCKDYADIC](https://godbolt.org/z/aqfY6TWas) |
| `canonical_block_dyadic` CPU (AVX2) | x86-64 | `canonical_block_dyadic` | E | [GB-x86-BLOCKDYADIC](https://godbolt.org/z/nEae7MGW7) |
| `canonical_block_dyadic` CPU (NEON) | AArch64 | `canonical_block_dyadic` | E | [GB-NEON-BLOCKDYADIC](https://godbolt.org/z/qEv6YnG54) |
| Cross-platform witness (GCC 16.1) | x86-64 | `pairwise` + `block_dyadic` | F | [GB-XPLAT-x86](https://godbolt.org/z/WovKvEh1b) |
| Cross-platform witness (Clang) | AArch64 | `pairwise` + `block_dyadic` | F | [GB-XPLAT-arm](https://godbolt.org/z/bjd5sfMKv) |
| Cross-platform witness (NVCC) | CUDA | `pairwise` + `block_dyadic` | F | [GB-XPLAT-cuda](https://godbolt.org/z/9ME9dYeen) |

**Key results** (from Appendix B.8 and Appendix F):

- **Scan/reduce consistency** (Tesla T4, N = 1,048,576 doubles, B = 256): canonical scan back element bit-exact with canonical reduce at `0x406fef4dbe54a0f8`; 22 of 22 sampled mid-range prefixes bit-exact with host reference.
- **CUB contrast**: CUB's `inclusive_scan` and `reduce` are each individually deterministic but disagree on the same input — 975,698 of 1,048,576 scan elements differ from canonical, and CUB's own scan back element differs from CUB's reduce.
- **Cross-platform reproducibility** (Appendix F): 84 PASS / 0 FAIL on each of x86-64, AArch64, and CUDA — bit-identical scan output hashes and 24 mid-block probe values across all three platforms for both `canonical_pairwise` and `canonical_block_dyadic` (B = 16 and B = 256), under matched FP environment (`-ffp-contract=off` host, `--fmad=false` CUDA).

---

## Building from source

Requires [pandoc](https://pandoc.org) and Python 3.

```bash
make P4016R0.html
make P4229R0.html
```

## License

These documents are submitted to WG21 (ISO/IEC JTC1/SC22/WG21) for consideration as C++ standards proposals.