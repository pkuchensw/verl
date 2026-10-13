"""Evaluate a merged model checkpoint on AIME-2024 with pass@8."""
import json, re, gc, os, sys
import torch
import numpy as np
from datasets import Dataset
from vllm import LLM, SamplingParams

SUBSTITUTIONS = [
    ('an ', ''), ('a ', ''), ('$', ''), (r'\ ', ''),
    (' ', ''), ('mbox', 'text'), (',\\text{and}', ','),
    ('\\text{and}', ','), ('\\text{m}', '\\text{}'),
]
REMOVED_EXPRESSIONS = [
    'square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'hours', 'km',
    'units', '\\ldots', 'sue', 'points', 'feet', 'minutes', 'digits', 'cents',
    'degrees', 'cm', 'gm', 'pounds', 'meters', 'meals', 'edges', 'students',
    'childrentickets', 'multiples', '\\text{s}', '\\text{.}', '\\text{\ns}',
    '\\text{}^2', '\\text{}^3', '\\text{\n}', '\\text{}',
    r'^\circ', r'^{\circ}', r'\,', '{,}',
    '"', '\\dots',
]

def normalize_final_answer(final_answer):
    final_answer = final_answer.split('=')[-1]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, '')
    final_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(frac)([^{])(.)', 'frac{\\2}{\\3}', final_answer)
    final_answer = re.sub(r'(sqrt)([^{])', 'sqrt{\\2}', final_answer)
    final_answer = re.sub(r'\$', '', final_answer)
    if final_answer.replace(',', '').isdigit():
        final_answer = final_answer.replace(',', '')
    return final_answer.strip()

def is_correct(solution_str, gt):
    match = re.findall(r'(?i)Answer\s*:\s*([^\n]+)', solution_str)
    if match:
        extracted_answer = match[-1]
    else:
        match = re.findall(r'\\boxed\{([^}]+)\}', solution_str)
        extracted_answer = match[-1] if match else '[INVALID]'
    pred = normalize_final_answer(extracted_answer)
    gt_norm = normalize_final_answer(gt)
    return pred == gt_norm, pred


def main():
    experiment = os.environ.get('EXPERIMENT', 'unknown')
    step = os.environ.get('STEP', 'unknown')
    target_dir = os.environ.get('TARGET_DIR', '')
    aime_parquet = os.environ.get('AIME_PARQUET', '')

    data = Dataset.from_parquet(aime_parquet)
    prompts = [d['prompt'][0]['content'] for d in data]
    ground_truths = [d['reward_model']['ground_truth'] for d in data]
    print(f'Loaded {len(prompts)} AIME questions')

    llm = LLM(
        model=target_dir,
        tokenizer=target_dir,
        tensor_parallel_size=4,
        dtype='bfloat16',
        gpu_memory_utilization=0.9,
        max_model_len=10368,
        enforce_eager=False,
        disable_custom_all_reduce=True,
    )

    sampling_params = SamplingParams(
        n=8,
        temperature=0.7,
        top_p=1.0,
        max_tokens=8192,
        stop=None,
    )

    outputs = llm.generate(prompts, sampling_params)

    del llm
    torch.cuda.empty_cache()
    gc.collect()

    all_pass1 = []
    all_pass8 = []

    for idx, output in enumerate(outputs):
        gt = ground_truths[idx]
        responses = [o.text for o in output.outputs]
        scores = []
        for resp in responses:
            correct, pred = is_correct(resp, gt)
            scores.append(1.0 if correct else 0.0)
        pass1 = np.mean(scores)
        pass8 = 1.0 if any(scores) else 0.0
        all_pass1.append(pass1)
        all_pass8.append(pass8)
        if idx < 3:
            print(f'  Q{idx}: pass@1={pass1:.3f} pass@8={pass8:.3f} pred={pred} gt={gt}')

    print(f'')
    print(f'  === Results for {experiment} {step} ===')
    print(f'  pass@1 (mean@8): {np.mean(all_pass1)*100:.1f}%')
    print(f'  pass@8 (best@8): {np.mean(all_pass8)*100:.1f}%')
    print(f'  pass@1 std: {np.std(all_pass1)*100:.1f}%')


if __name__ == '__main__':
    main()
