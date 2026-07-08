#!/usr/bin/env python3
"""
tools/mlir_to_python.py

A probe: pull one MLIR snapshot from the Parquet corpus and ask a local
CPU-hosted code model to write Python that captures what the IR does.

Generative experiment, not lossless recovery. Includes a signal-extraction
step that isolates user functions (main, kernel stubs) from the library
boilerplate that dominates raw ClangIR output.
"""

import argparse
import re
import sys
import time
from pathlib import Path


def load_snapshot(parquet_dir, benchmark, pass_name):
    import pyarrow.dataset as ds
    import pyarrow.compute as pc

    dset = ds.dataset(parquet_dir, format='parquet', partitioning='hive')
    flt = (pc.field('benchmark_name') == benchmark)
    if pass_name:
        flt = flt & (pc.field('pass_name') == pass_name)
    table = dset.to_table(filter=flt)

    if table.num_rows == 0:
        raise SystemExit(f'No snapshot found for benchmark={benchmark} '
                         f'pass={pass_name}. Check names.')

    row = {col: table.column(col)[0].as_py() for col in table.column_names}
    return row['ir_text'], row


def split_functions(ir_text):
    """
    Split an MLIR module into (header, [functions]) using plain string
    matching (no regex, to stay heredoc-safe).

    A function definition line contains 'cir.func' or 'llvm.func'.
    """
    def is_func_line(s):
        return ('cir.func' in s) or ('llvm.func' in s)

    lines = ir_text.splitlines()
    header_lines = []
    functions = []

    i = 0
    n = len(lines)
    while i < n and not is_func_line(lines[i]):
        header_lines.append(lines[i])
        i += 1

    while i < n:
        line = lines[i]
        if is_func_line(line):
            block = [line]
            depth = line.count('{') - line.count('}')
            i += 1
            while i < n and depth > 0:
                block.append(lines[i])
                depth += lines[i].count('{') - lines[i].count('}')
                i += 1
            functions.append((line.strip(), '\n'.join(block)))
        else:
            i += 1

    return '\n'.join(header_lines), functions


def is_user_function(signature):
    if not signature or not isinstance(signature, str):
        return False
    """
    Heuristic: is this a user-defined function worth showing the model,
    vs. C++ standard library boilerplate?

    Keep:  @main, CUDA device stubs (__device_stub), demangled kernel
           names, anything not obviously std::.
    Drop:  std:: library instantiations, chrono, random internals.
    """
    s = signature.lower()

    # Always keep main and kernel-related functions.
    if '@main' in s:
        return True
    if 'device_stub' in s or 'cudalaunch' in s or 'kernel' in s:
        return True

    # Drop obvious C++ standard library functions.
    drop_markers = ['std', 'chrono', 'random', '_zst', '__gnu',
                    'basic_string', 'allocator', 'char_traits',
                    'ratio', 'duration', 'operator']
    if any(m in s for m in drop_markers):
        return False

    # Default: keep. Better to show the model a user function we weren't
    # sure about than to hide the actual computation.
    return True


def extract_signal(ir_text, max_lines):
    """
    Return a reduced version of the IR that keeps the signal (user functions)
    and drops the library-boilerplate noise. Prioritizes main and kernel
    functions, includes as many user functions as fit in max_lines.
    """
    header, functions = split_functions(ir_text)

    user_funcs = [(sig, body) for sig, body in functions if is_user_function(sig)]
    if not user_funcs:
        # Fallback: nothing matched; just clip the top.
        return '\n'.join(ir_text.splitlines()[:max_lines]), 0, len(functions)

    # Order: main first, then device stubs/kernels, then the rest.
    def priority(item):
        s = item[0].lower()
        if '@main' in s:
            return 0
        if 'device_stub' in s or 'kernel' in s:
            return 1
        return 2
    user_funcs.sort(key=priority)

    # A trimmed header: keep only a few lines so the model knows the module
    # has type declarations, without drowning in them.
    header_lines = header.splitlines()
    trimmed_header = header_lines[:8]
    if len(header_lines) > 8:
        trimmed_header.append(f'// ... ({len(header_lines) - 8} more type/attr declarations)')

    out = list(trimmed_header)
    out.append('')
    # Hard line budget. We add functions in priority order and clip INSIDE
    # a function if it would overflow, so a single large function (like main)
    # can't blow the whole budget.
    budget = max_lines - len(out)
    included = 0
    for sig, body in user_funcs:
        if budget <= 0:
            break
        blk = body.splitlines()
        if len(blk) > budget:
            # Clip this function to what fits, note the truncation.
            blk = blk[:budget]
            blk.append('    // ... (function body truncated to fit budget)')
        out.extend(blk)
        out.append('')
        budget -= (len(blk) + 1)
        included += 1
    if included < len(user_funcs):
        out.append(f'// ... ({len(user_funcs) - included} more user functions omitted for length)')

    return '\n'.join(out), len(user_funcs), len(functions)


def build_prompt(ir_text, meta, max_ir_lines):
    signal, n_user, n_total = extract_signal(ir_text, max_ir_lines)

    system = (
        "You are a compiler engineer fluent in MLIR and Python. "
        "You are given the SIGNAL portion of an MLIR module (ClangIR/LLVM "
        "dialects) produced by lowering a CUDA/C++ program. Library boilerplate "
        "has been stripped; what remains is the user's own functions (main, "
        "CUDA kernel stubs, helpers). Write a short, readable Python program "
        "that performs the same high-level computation. Focus on observable "
        "behavior, not literal op-by-op translation. "
        "Output only a Python code block and one or two sentences of explanation."
    )

    user = (
        f"Benchmark: {meta['benchmark_name']}\n"
        f"Pass stage: {meta['pass_name']} (index {meta['pass_index']})\n"
        f"Kept {n_user} user functions out of {n_total} total in the module.\n\n"
        f"MLIR (signal only):\n```mlir\n{signal}\n```\n\n"
        f"Write equivalent Python."
    )
    return system, user, n_user, n_total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--parquet', required=True, type=Path)
    ap.add_argument('--benchmark', required=True)
    ap.add_argument('--pass-name', default=None)
    ap.add_argument('--model', default='Qwen/Qwen2.5-Coder-1.5B-Instruct')
    ap.add_argument('--max-ir-lines', default=200, type=int)
    ap.add_argument('--max-new-tokens', default=512, type=int)
    ap.add_argument('--threads', default=None, type=int)
    ap.add_argument('--show-ir', action='store_true',
                    help='Print the extracted signal IR before generating.')
    args = ap.parse_args()

    print(f'[1/4] Loading snapshot: {args.benchmark} / {args.pass_name or "(first)"}')
    ir_text, meta = load_snapshot(args.parquet, args.benchmark, args.pass_name)
    print(f'    IR is {meta["ir_lines"]} lines, {meta["ir_bytes"]} bytes.')

    system, user, n_user, n_total = build_prompt(ir_text, meta, args.max_ir_lines)
    print(f'    Extracted signal: {n_user} user functions (of {n_total} total).')

    if args.show_ir:
        print('-' * 70)
        print(user)
        print('-' * 70)

    print(f'[2/4] Loading {args.model} ...')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.threads:
        torch.set_num_threads(args.threads)

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, device_map='cpu',
    )
    print(f'    Model loaded in {time.time() - t0:.1f}s.')

    print('[3/4] Tokenizing...')
    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': user},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors='pt')
    prompt_tokens = inputs['input_ids'].shape[1]
    print(f'    Prompt is {prompt_tokens} tokens.')

    print(f'[4/4] Generating up to {args.max_new_tokens} tokens (CPU)...')
    t0 = time.time()
    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens,
            do_sample=False, temperature=None, top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    new_tokens = generated[0][prompt_tokens:]
    output = tokenizer.decode(new_tokens, skip_special_tokens=True)
    n_new = len(new_tokens)

    print()
    print('=' * 70)
    print(f'MODEL OUTPUT  ({n_new} tokens in {elapsed:.1f}s = {n_new/elapsed:.2f} tok/s)')
    print('=' * 70)
    print(output)
    print('=' * 70)


if __name__ == '__main__':
    main()
