"""Verify the Score Centering engine-side primitives without a GPU.

Covers:
  * logprobs_at_ids_from_logits: full-vocab normalization, invalid-slot masking,
    fp32 output, gradient flowing through the logits, and equivalence to
    log_softmax+gather.
  * response_field_to_rmpad_label_positions: the scatter matches the response window
    that no_padding_2_padding reads, so the gathered logprobs line up with the
    response axis (round-trip test against no_padding_2_padding).
"""

import importlib.util
import sys

import torch
import torch.nn.functional as F

ROOT = "/jizhicfs/pkuhetu/siwei/verl/verl"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# torch_functional imports verl.utils.* -> load via the real package path.
sys.path.insert(0, ROOT)
from verl.utils.torch_functional import logprobs_at_ids_from_logits  # noqa: E402

PASS = True


def check(name, cond, extra=""):
    global PASS
    if not cond:
        PASS = False
    print(("OK  " if cond else "FAIL") + "  " + name + (("  " + extra) if extra else ""))


# ---------------------------------------------------------------- primitive ----
torch.manual_seed(0)
N, V, K = 6, 11, 3
logits = torch.randn(N, V, dtype=torch.float64)  # emulate model logits (cp to fp32)
logits32 = logits.float()

ids = torch.randint(0, V, (N, K))
ids[0, 1] = -1  # invalid slot
ids[3, 2] = -1

out = logprobs_at_ids_from_logits(logits32, ids, chunk_size=2)
ref = F.log_softmax(logits32, dim=-1).gather(-1, ids.clamp(min=0))
valid_mask = ids >= 0

check("engine: matches log_softmax+gather (valid slots)",
      torch.allclose(out[valid_mask], ref[valid_mask], atol=1e-5),
      f"max err={(out[valid_mask] - ref[valid_mask]).abs().max().item():.2e}")
check("engine: output is fp32", out.dtype == torch.float32)
check("engine: invalid slots are 0.0",
      out[0, 1].item() == 0.0 and out[3, 2].item() == 0.0)

# gradient flows through the logits and matches the analytic value.
x = logits32.clone().requires_grad_(True)
o = logprobs_at_ids_from_logits(x, ids, chunk_size=2)
o.sum().backward()
# d/dlogits[i,v] = sum_k [1{v==ids[i,k]} valid] - K_i * softmax(logits[i])[v]
valid = (ids >= 0).float()
count = valid.sum(-1, keepdim=True)  # (N,1)
onehot = torch.zeros(N, V)
for i in range(N):
    for k in range(K):
        if ids[i, k] >= 0:
            onehot[i, ids[i, k]] += 1.0
expected = onehot - count * F.softmax(logits32, dim=-1)
check("engine: gradient flows through logits (analytic match)",
      torch.allclose(x.grad, expected, atol=1e-4),
      f"max err={(x.grad - expected).abs().max().item():.2e}")

# chunk_size invariance
for cs in (1, 3, 100):
    o2 = logprobs_at_ids_from_logits(logits32, ids, chunk_size=cs)
    check(f"engine: chunk_size={cs} invariant", torch.allclose(o2, out, atol=1e-6))

# ------------------------------------------------------------- rmpad scatter ----
# Build a fake NO_PADDING micro-batch: prompts/responses nested, sc_topk_ids padded.
padding = load("verl_padding", "/jizhicfs/pkuhetu/siwei/verl/verl/workers/utils/padding.py")

from tensordict import TensorDict  # noqa: E402


def make_batch(prompt_lens, resp_lens, max_resp, K, V=11):
    prompts = [torch.arange(p) for p in prompt_lens]
    responses = [torch.arange(r) for r in resp_lens]
    n = len(prompt_lens)
    g = torch.Generator().manual_seed(1234)
    sc_ids = torch.full((n, max_resp, K), -1, dtype=torch.int64)
    for i, r in enumerate(resp_lens):
        sc_ids[i, :r] = torch.randint(0, V, (r, K), generator=g)
    td = TensorDict(
        {
            "prompts": torch.nested.as_nested_tensor(prompts, layout=torch.jagged),
            "responses": torch.nested.as_nested_tensor(responses, layout=torch.jagged),
            "sc_topk_ids": sc_ids,
        },
        batch_size=n,
    )
    return td, sc_ids


prompt_lens = [4, 3, 5]
resp_lens = [3, 2, 4]
max_resp, K = 4, 3
td, sc_ids = make_batch(prompt_lens, resp_lens, max_resp, K)
total_nnz = sum(prompt_lens) + sum(resp_lens)

ids_rmpad = padding.response_field_to_rmpad_label_positions(td["sc_topk_ids"], td, total_nnz=total_nnz)

# Direct check: the rmpad row we wrote for sequence i, response position j must be the
# same row no_padding_2_padding reads for that (i, j).
offs = td["prompts"].offsets()
pl = offs.diff()
rl = td["responses"].offsets().diff()
s = (pl + rl).cumsum(0)
ok = True
for i in range(len(prompt_lens)):
    for j in range(resp_lens[i]):
        row = int(s[i]) - int(rl[i]) - 1 + j
        if not torch.equal(ids_rmpad[row], sc_ids[i, j]):
            ok = False
check("rmpad: response position j maps to the right rmpad row", ok)

# Rows outside any response window keep -1.
mask_resp = torch.zeros(total_nnz, dtype=torch.bool)
for i in range(len(prompt_lens)):
    mask_resp[int(s[i]) - int(rl[i]) - 1 : int(s[i]) - 1] = True
check("rmpad: non-response rows stay invalid",
      bool((ids_rmpad[~mask_resp] == -1).all()))

# End-to-end shape check: gathered logprobs padded back to the response window.
logits = torch.randn(total_nnz, V)
gathered = logprobs_at_ids_from_logits(logits, ids_rmpad)
check("rmpad: gathered shape == (total_nnz, K)", gathered.shape == (total_nnz, K))
padded = padding.no_padding_2_padding(gathered, td)  # (n, max_resp, K)
check("rmpad: padded shape == (n, max_resp, K)", padded.shape == (len(prompt_lens), max_resp, K))
# and the (i,j) entry equals the full-vocab logprob of sc_ids[i,j] under logits[row]
ref_lp = F.log_softmax(logits, dim=-1)
ok = True
for i in range(len(prompt_lens)):
    for j in range(resp_lens[i]):
        row = int(s[i]) - int(rl[i]) - 1 + j
        for k in range(K):
            if not torch.isclose(padded[i, j, k], ref_lp[row, sc_ids[i, j, k]], atol=1e-5):
                ok = False
check("rmpad: padded magnitudes equal full-vocab logp at the right row", ok)

# ------------------------------------------------------- nested-field parity ----
# The real TransferQueue pipeline stores sc_topk_ids as a jagged nested tensor
# [bsz, j1, K] (list_of_dict_to_tensordict nests 2-D sequence fields). Confirm the
# scatter equals the padded-field path, i.e. field[i] indexes sample i not flat row i.
g = torch.Generator().manual_seed(1234)
nested_rows = [torch.randint(0, V, (r, K), generator=g) for r in resp_lens]
sc_ids_nested = torch.nested.as_nested_tensor(nested_rows, layout=torch.jagged)
sc_ids_padded = torch.full((len(prompt_lens), max_resp, K), -1, dtype=torch.int64)
for i, r in enumerate(resp_lens):
    sc_ids_padded[i, :r] = nested_rows[i]
prompts_nt = torch.nested.as_nested_tensor([torch.arange(p) for p in prompt_lens], layout=torch.jagged)
responses_nt = torch.nested.as_nested_tensor([torch.arange(r) for r in resp_lens], layout=torch.jagged)
td_nested = TensorDict(
    {"prompts": prompts_nt, "responses": responses_nt, "sc_topk_ids": sc_ids_nested},
    batch_size=len(prompt_lens),
)
td_padded = TensorDict(
    {"prompts": prompts_nt, "responses": responses_nt, "sc_topk_ids": sc_ids_padded},
    batch_size=len(prompt_lens),
)
ids_nested = padding.response_field_to_rmpad_label_positions(td_nested["sc_topk_ids"], td_nested, total_nnz=total_nnz)
ids_padded = padding.response_field_to_rmpad_label_positions(td_padded["sc_topk_ids"], td_padded, total_nnz=total_nnz)
check("rmpad: nested field scatters identically to padded field", torch.equal(ids_nested, ids_padded))
check("rmpad: nested scatter is non-trivial", bool((ids_nested >= 0).any()))

print()
print("ALL CHECKS PASSED" if PASS else "SOME CHECKS FAILED")
