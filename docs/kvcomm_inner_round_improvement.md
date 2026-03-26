# KV Reuse with Local Reference vs Global Base
## KVCOMM Analysis & Improvement Proposal

---

## 1. Background

KVCOMM formulates KV reuse as an **offset reconstruction problem**.

Define a global reference (base KV): $KV(x \mid \text{base})$

For each agent context $P_i$, store offset:

$$\Delta_{\text{base} \to P_i}(x) = KV(x \mid P_i) - KV(x \mid \text{base})$$

At inference (reuse stage):

$$KV(x \mid P_j) \approx KV(x \mid \text{base}) + \hat{\Delta}_{\text{base} \to P_j}(x)$$

where the offset is estimated via anchor matching.

---

## 2. What KVCOMM Actually Does

### Anchor Storage

Each anchor stores:

```
ph_key_delta = KV(x | P_agent) - KV(x | base)
pf_key_delta = KV(prefix | P_agent) - KV(prefix | base)
```

### KV Reconstruction

```
new_ph = base_ph_cache + weighted_sum(anchor.ph_key_delta)
new_pf = base_pf_cache + weighted_sum(anchor.pf_key_delta)
```

where $\text{base\_ph\_cache} = KV(x \mid \text{base})$.

---

## 3. Key Observation (Current-Round Setting)

In a multi-agent pipeline **within the same request**, we already have $KV(x \mid P_1)$ because:

- Agent 1 has performed dense prefill
- Its KV cache is stored in shared memory
- Downstream agents (e.g., Agent 2) can fetch it directly

---

## 4. Problem: Suboptimal Reference Choice

KVCOMM ignores this available information and reconstructs:

$$KV(x \mid P_2) \approx KV(x \mid \text{base}) + \Delta_{\text{base} \to P_2}$$

However, we already have a **closer reference**: $KV(x \mid P_1)$

---

## 5. Proposed Reformulation: Local Reference (Cross-Agent Offset)

Instead of using base KV, we propose:

$$KV(x \mid P_2) = KV(x \mid P_1) + \Delta_{P_1 \to P_2}$$

where:

$$\Delta_{P_1 \to P_2} = KV(x \mid P_2) - KV(x \mid P_1)$$

---

## 6. Key Insight: No Extra Storage Needed

Observe:

$$\Delta_{P_1 \to P_2} = \Delta_{\text{base} \to P_2} - \Delta_{\text{base} \to P_1}$$

Thus:

```python
delta_cross = agent2_ph_key_delta - agent1_ph_key_delta
new_ph = KV(x | P1) + weighted_sum(delta_cross)
```

> 👉 This **reuses existing anchor structure** directly — no new storage required.

---

## 7. Geometric Interpretation

```
KVCOMM (global):
    base ──────────────────────────────→ P2

Proposed (local):
    P1 ──────→ P2
```

> 👉 Local reference is closer → **lower bias**.

---

## 8. Hypothesis

Define:

$$\Delta_{\text{abs}}(x) = KV(x \mid P_2) - KV(x \mid \text{base})$$

$$\Delta_{\text{cross}}(x) = KV(x \mid P_2) - KV(x \mid P_1)$$

**Hypothesis:**

$$\text{Var}[\Delta_{\text{cross}}] < \text{Var}[\Delta_{\text{abs}}]$$

> 👉 Cross-agent offsets are more stable across requests  
> 👉 Easier to approximate via anchor matching

---

## 9. Caveat: Error Accumulation

Cross-offset estimation requires:

$$\hat{\Delta}_{\text{cross}} = \hat{\Delta}_{\text{base} \to P_2} - \hat{\Delta}_{\text{base} \to P_1}$$

Thus:

$$\text{error}_{\text{cross}} \approx \text{error}_{P_2} + \text{error}_{P_1}$$

while KVCOMM:

$$\text{error}_{\text{abs}} \approx \text{error}_{P_2}$$

### Trade-off Summary

| Method | Bias | Variance | Error Source |
|---|---|---|---|
| KVCOMM | High | Lower | single offset estimation |
| Cross-offset (Proposed) | Low | Lower | double estimation error |

---

## 10. When Each Method Works Best

**Cross-offset (Proposed)** — best when:
- Same request
- Upstream KV available
- Prefix difference small / moderate

**KVCOMM (Base-offset)** — best when:
- Cross-request reuse
- No upstream KV available
- Arbitrary agent graph

---

## 11. Hybrid Strategy (Recommended)

Simple version:

```python
if shared_KV_available:
    use cross_offset
else:
    use base_offset
```

More robust version:

```python
if similarity(P1, P2) > threshold:
    use cross_offset
else:
    use base_offset
```

---

## 12. Experimental Design

### 12.1 Offset Variance

Measure:

$$\text{Var}[\Delta_{\text{abs}}] \quad \text{vs} \quad \text{Var}[\Delta_{\text{cross}}]$$

### 12.2 Reconstruction Error

Compare $\|\widehat{KV}(x \mid P_2) - KV(x \mid P_2)\|$ for:
- KVCOMM (base reference)
- Proposed (cross reference)

### 12.3 Scenario Breakdown

- **Similar prefix** (expected: cross > base)
- **Divergent prefix** (unclear / stress test)

---

## 13. Final Takeaway

KVCOMM adopts a **global base-conditioned reference** for KV reconstruction, enabling cross-request generalization but ignoring available context-conditioned KV within the same request.

The proposed **local reference formulation** leverages upstream KV and cross-agent offsets, reducing approximation bias and potentially improving reconstruction accuracy.

---

> 🔥 **One-line Insight**
>
> KV reuse is not just about offset estimation — it is about **choosing the right reference frame**.
