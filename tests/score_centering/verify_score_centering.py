"""Gradient / correctness verification for Score Centering (arXiv 2609.20807v1).

Standalone: imports the SC module by file path so it runs on the local Python 3.11
without the cluster's ray/py3.12 environment.
"""

import importlib.util
import sys

import torch
import torch.nn.functional as F

spec = importlib.util.spec_from_file_location(
    "score_centering",
    "/jizhicfs/pkuhetu/siwei/verl/verl/workers/utils/score_centering.py",
)
sc = importlib.util.module_from_spec(spec)
sys.modules["score_centering"] = sc
spec.loader.exec_module(sc)

PASS = True


def check(name, cond, extra=""):
    global PASS
    tag = "OK  " if cond else "FAIL"
    if not cond:
        PASS = False
    print(f"{tag}  {name}{('  ' + extra) if extra else ''}")


# --------------------------------------------------------------------------------------
# Reference: FULL-vocab SC and plain PG over a single position.
# p = softmax(logits), q = fixed sampler distribution (detached), a = sampled token.
# --------------------------------------------------------------------------------------
def full_sc_loss(logits, q, a):
    logp = F.log_softmax(logits, dim=-1)
    return -(logp[a] - (q.detach() * logp).sum())


def plain_pg_loss(logits, q, a):
    logp = F.log_softmax(logits, dim=-1)
    return -logp[a]


def grad_of(fn, logits, q, a):
    x = logits.detach().clone().requires_grad_(True)
    fn(x, q, a).backward()
    return x.grad.detach()


# ======================================================================================
# Check 2: expected SC gradient is zero under the sampling distribution; plain PG is not.
# ======================================================================================
torch.manual_seed(0)
V = 6
logits0 = torch.randn(V, dtype=torch.float64)
p0 = F.softmax(logits0, dim=-1)
# q deliberately different from p (off-policy gap)
q = torch.softmax(torch.randn(V, dtype=torch.float64) * 1.5, dim=-1)

exp_grad_sc = torch.zeros(V, dtype=torch.float64)
exp_grad_pg = torch.zeros(V, dtype=torch.float64)
for a in range(V):
    exp_grad_sc += q[a] * grad_of(full_sc_loss, logits0, q, a)
    exp_grad_pg += q[a] * grad_of(plain_pg_loss, logits0, q, a)

check(
    "check2: E_q[grad SC] == 0",
    exp_grad_sc.abs().max().item() < 1e-10,
    f"max|g|={exp_grad_sc.abs().max().item():.2e}",
)
check(
    "check2: E_q[grad plain PG] != 0 (centering actually matters)",
    exp_grad_pg.abs().max().item() > 1e-3,
    f"max|g|={exp_grad_pg.abs().max().item():.2e}",
)

# ======================================================================================
# Helper: build a small module-level SC call for checks 1, 3, 4.
# ======================================================================================
def make_topk_inputs(logits_p, logits_q, k, sampled_ids):
    """logits_p: [S, V] training logits (grad). logits_q: [S, V] sampler logits (fixed).
    Returns the kwargs the SC token loss needs, for batch=1."""
    S, V = logits_p.shape
    logp = F.log_softmax(logits_p, dim=-1)  # [S,V]
    logq = F.log_softmax(logits_q, dim=-1)  # [S,V]
    topk_ids = logq.topk(k, dim=-1).indices  # [S,k] sampler top-k
    sampler_topk = logq.gather(-1, topk_ids)  # [S,k]  log q(v)
    train_topk = logp.gather(-1, topk_ids)  # [S,k]  log p(v)
    sampled = logp.gather(-1, sampled_ids.unsqueeze(-1)).squeeze(-1)  # [S] log p(a)
    return (
        sampled.unsqueeze(0),
        train_topk.unsqueeze(0),
        sampler_topk.unsqueeze(0),
        torch.ones(1, S, k),
        topk_ids,
    )


# ======================================================================================
# Check 1: p == q  =>  SC correction gradient is zero (SC degenerates to vanilla PG).
# ======================================================================================
torch.manual_seed(1)
S, V, k = 4, 7, 3
logits = torch.randn(S, V)
sampled_ids = torch.randint(0, V, (S,))
logp = F.log_softmax(logits, dim=-1)
topk_ids = logp.topk(k, dim=-1).indices
sampler_topk = logp.gather(-1, topk_ids).detach()
train_topk = logp.gather(-1, topk_ids)
sampled = logp.gather(-1, sampled_ids.unsqueeze(-1)).squeeze(-1).detach()

# correction alone with p==q
x = logits.detach().clone().requires_grad_(True)
lp = F.log_softmax(x, dim=-1)
tt = lp.gather(-1, topk_ids)
tk = torch.exp(lp.gather(-1, topk_ids))  # p_head
q_head = torch.exp(sampler_topk)  # q_head == p_head
q_tail = (1 - q_head.sum(-1)).clamp(min=sc.TAIL_EPS)
p_tail = (1 - tk.sum(-1)).clamp(min=sc.TAIL_EPS)
rho = q_tail / p_tail
coeff = (q_head - rho.unsqueeze(-1) * tk).detach()
corr = (coeff * tt).sum(-1)
corr.sum().backward()
check(
    "check1: p==q => d(correction)/d(logits) == 0",
    x.grad.abs().max().item() < 1e-9,
    f"max|g|={x.grad.abs().max().item():.2e}",
)

# ======================================================================================
# Check 3: autograd matches the hand-derived gradients; detach positions correct.
#   d(token_loss)/d(train_sampled_logprob) = -A
#   d(token_loss)/d(train_topk_logprob_k)  = +A * coeff_k    (coeff detached)
# ======================================================================================
torch.manual_seed(2)
S, V, k = 3, 6, 4
logits_p = torch.randn(S, V)
logits_q = torch.randn(S, V) * 1.3
sampled_ids = torch.randint(0, V, (S,))
adv = torch.randn(S)
rmask = torch.ones(S, dtype=torch.bool)

ts, tt, st, mk, topk_ids = make_topk_inputs(logits_p, logits_q, k, sampled_ids)
ts = ts.squeeze(0).detach().clone().requires_grad_(True)
tt = tt.squeeze(0).detach().clone().requires_grad_(True)

token_loss = sc.score_centering_token_loss(
    train_sampled_logprob=ts.unsqueeze(0),
    train_topk_logprob=tt.unsqueeze(0),
    sampler_topk_logprob=st,
    topk_mask=mk,
    advantages=adv.unsqueeze(0),
    response_mask=rmask.unsqueeze(0),
).squeeze(0)

(token_loss.sum()).backward()

# hand-derived coeff (recompute detached)
q_head = torch.exp(st.squeeze(0))
p_head = torch.exp(tt.detach())
q_tail = (1 - q_head.sum(-1)).clamp(min=sc.TAIL_EPS)
p_tail = (1 - p_head.sum(-1)).clamp(min=sc.TAIL_EPS)
rho = (q_tail / p_tail).clamp(max=sc.RHO_MAX)
coeff = (q_head - rho.unsqueeze(-1) * p_head).detach()

g_sampled_expected = -adv  # sum over sequence, mask all ones
g_topk_expected = (adv.unsqueeze(-1) * coeff)

check(
    "check3: d/d(sampled logp) == -A",
    torch.allclose(ts.grad, g_sampled_expected, atol=1e-5),
    f"max err={(ts.grad - g_sampled_expected).abs().max().item():.2e}",
)
check(
    "check3: d/d(topk logp_k) == A*coeff_k  (coeff detached)",
    torch.allclose(tt.grad, g_topk_expected, atol=1e-5),
    f"max err={(tt.grad - g_topk_expected).abs().max().item():.2e}",
)

# detach discipline: the gradient wrt the train logits must NOT flow through the p_head
# or rho that appear inside the (detached) coefficient. Equivalent statement: replacing
# the coefficient by its constant value reproduces the gradient exactly.
d_loss = tt.grad  # from the backward() above (== A * coeff)
g_const = adv.unsqueeze(-1) * coeff  # coeff treated as a detached constant
check(
    "check3: no gradient flows through p_head/rho inside the coefficient",
    torch.allclose(d_loss, g_const, atol=1e-5),
    f"max diff={(d_loss - g_const).abs().max().item():.2e}",
)

# ======================================================================================
# Check 4: vectorized top-k == naive scalar loop of the same formula.
#   Also report the k=V relation to the *full-sum* SC (entropy residual).
# ======================================================================================
torch.manual_seed(3)
S, V, k = 5, 9, 4
logits_p = torch.randn(S, V)
logits_q = torch.randn(S, V) * 1.2
sampled_ids = torch.randint(0, V, (S,))
adv = torch.randn(S)

ts, tt, st, mk, topk_ids = make_topk_inputs(logits_p, logits_q, k, sampled_ids)

vec = sc.score_centering_token_loss(
    train_sampled_logprob=ts,
    train_topk_logprob=tt,
    sampler_topk_logprob=st,
    topk_mask=mk,
    advantages=adv.unsqueeze(0),
    response_mask=torch.ones(1, S, dtype=torch.bool),
).squeeze(0)


def naive_loop():
    out = []
    for s in range(S):
        qh = torch.exp(st[0, s])
        ph = torch.exp(tt[0, s].detach())
        qt = max(1.0 - qh.sum().item(), sc.TAIL_EPS)
        pt = max(1.0 - ph.sum().item(), sc.TAIL_EPS)
        r = min(qt / pt, sc.RHO_MAX)
        c = (qh - r * ph).detach()
        corr = (c * tt[0, s]).sum()
        out.append(-adv[s] * (ts[0, s] - corr))
    return torch.stack(out)


check(
    "check4: vectorized == naive scalar loop",
    torch.allclose(vec, naive_loop(), atol=1e-5),
    f"max diff={(vec - naive_loop()).abs().max().item():.2e}",
)

# k = V: report relation to full-sum SC. Full SC correction = sum_v q_v log p_v.
k_full = V
ts2, tt2, st2, mk2, ids2 = make_topk_inputs(logits_p, logits_q, k_full, sampled_ids)
vec_full = sc.score_centering_token_loss(
    train_sampled_logprob=ts2,
    train_topk_logprob=tt2,
    sampler_topk_logprob=st2,
    topk_mask=mk2,
    advantages=adv.unsqueeze(0),
    response_mask=torch.ones(1, S, dtype=torch.bool),
).squeeze(0)

# explicit full-vocab SC
logp = F.log_softmax(logits_p, dim=-1)
logq = F.log_softmax(logits_q, dim=-1)
q_probs = torch.exp(logq)
full_sc = torch.stack(
    [
        -adv[s] * (logp[s, sampled_ids[s]] - (q_probs[s] * logp[s]).sum())
        for s in range(S)
    ]
)
resid = (vec_full - full_sc)
# predicted residual = +A * H(q_hat) where H is entropy of the p-head over H (here H=V so q_hat=p)
ent = -(q_probs * torch.log(q_probs.clamp(min=1e-30))).sum(-1)  # H(p) == H(q) only if p==q
# At k=V the tail is 0 -> rho=1 -> coeff = q - p; residual should equal A * sum_v p_v log p_v
# = A * (-H(p)).
H_p = -(torch.exp(logp) * logp).sum(-1)
pred_resid = adv * H_p  # resid = -A * sum(p log p) = +A * H(p)
check(
    "check4: k=V residual vs predicted A*sum(p log p)",
    torch.allclose(resid, pred_resid, atol=1e-4),
    f"max|resid - pred|={(resid - pred_resid).abs().max().item():.2e}",
)
print(
    f"      [info] k=V residual = A*sum_v p_v log p_v = -A*H(p); this is the tail-reconstruction "
    f"entropy term (see report). mean|resid|={resid.abs().mean().item():.3e}"
)

# ======================================================================================
# Check 5: combined SC + IS mechanics.
#   5a) _is_weight modes: none = identity, tis = upper clip, mis = band mask.
#   5b) combined-mode coefficient matches its hand derivation
#       coeff = q_head * f(p_head/q_head) - alpha * p_head,  alpha = rho * f(1/rho).
#   5c) invalid slots are excluded from both q_head and p_head.
# ======================================================================================
torch.manual_seed(4)
S, V, k = 4, 8, 3

w = torch.tensor([0.25, 1.0, 4.0, 20.0])
check("check5a: none = identity", torch.allclose(sc._is_weight(w, "none", 0.5, 5.0), w))
check("check5a: tis clips upper only",
      torch.allclose(sc._is_weight(w, "tis", 0.5, 5.0), torch.tensor([0.25, 1.0, 4.0, 5.0])))
check("check5a: mis zeroes out-of-band",
      torch.allclose(sc._is_weight(w, "mis", 0.5, 5.0), torch.tensor([0.0, 1.0, 4.0, 0.0])))

# hand-built combined coefficient for one position
qh = torch.tensor([0.30, 0.15, 0.10])          # q_head (3 of 4 slots)
ph = torch.tensor([0.25, 0.20, 0.12])          # p_head (detached)
is_low, is_high = 0.5, 5.0
q_tail = 1.0 - qh.sum()
p_tail = 1.0 - ph.sum()
rho = (q_tail / p_tail).clamp(max=sc.RHO_MAX)
head_w = sc._is_weight(ph / qh, "tis", is_low, is_high)
alpha = rho * sc._is_weight((1.0 / rho).unsqueeze(0), "tis", is_low, is_high).squeeze(0)
coeff_manual = qh * head_w - alpha * ph

# produce it through the module
S1, V1 = 1, 4
# sampler top-k logprobs for these 3 slots
st_1 = torch.log(qh).reshape(1, 1, 3)
tt_1 = torch.log(ph).reshape(1, 1, 3).clone().requires_grad_(True)
mask_1 = torch.ones(1, 1, 3)
ts_1 = torch.tensor([[[-1.0]]])                # sampled logprob (not in top-k)
adv_1 = torch.tensor([[1.0]])
rmask_1 = torch.ones(1, 1, dtype=torch.bool)

# torch.log on a detached clone: build coeff via the module's own path by calling the
# token loss with is_mode="tis" and reading its gradient (== coeff * A).
tl = sc.score_centering_token_loss(
    train_sampled_logprob=ts_1,
    train_topk_logprob=tt_1,
    sampler_topk_logprob=st_1,
    topk_mask=mask_1,
    advantages=adv_1,
    response_mask=rmask_1,
    sampled_weight=torch.tensor([[1.0]]),
    is_mode="tis",
    is_low=is_low,
    is_high=is_high,
)
tl.sum().backward()
check("check5b: combined coeff matches hand derivation",
      torch.allclose(tt_1.grad.reshape(-1), coeff_manual, atol=1e-5),
      f"max err={(tt_1.grad.reshape(-1) - coeff_manual).abs().max().item():.2e}")

# 5c: an invalid slot (mask 0) must be excluded from the head mass. Use p != q so the
# correction is non-zero, then confirm toggling the mask changes the loss.
st_2 = torch.tensor([[[-0.5, -1.2, -2.0]]])
tt_2 = torch.tensor([[[-0.3, -1.5, -1.0]]])

def loss_with_mask(m):
    return sc.score_centering_token_loss(
        train_sampled_logprob=torch.zeros(1, 1),
        train_topk_logprob=tt_2,
        sampler_topk_logprob=st_2,
        topk_mask=torch.tensor([[list(m)]]),
        advantages=torch.ones(1, 1),
        response_mask=torch.ones(1, 1, dtype=torch.bool),
    )[0, 0].item()

full = loss_with_mask([1.0, 1.0, 1.0])
masked = loss_with_mask([1.0, 1.0, 0.0])
nomask = loss_with_mask([0.0, 0.0, 0.0])
check("check5c: mask 0 excludes the slot from head mass", full != masked)
check("check5c: all-invalid rows fall back to the centered-zero correction",
      abs(nomask - 0.0) < 1e-6, f"nomask={nomask:.2e}")

print()
print("ALL CHECKS PASSED" if PASS else "SOME CHECKS FAILED")
