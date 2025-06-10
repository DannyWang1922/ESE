# ESE Training Debug Report

## Issue Summary
The NLI fine-tuning was producing results that were 10% worse than the original embedding models (both BERT and BGE). The root cause was incorrect label handling during training.

## Root Cause Analysis

### 1. Label Format Mismatch
**Problem**: NLI datasets (SNLI/MultiNLI) use categorical labels, but the AnglE framework expects continuous similarity scores.

- **NLI Labels**:
  - `0` = Entailment (premise implies hypothesis)
  - `1` = Neutral (no clear relationship)
  - `2` = Contradiction (premise contradicts hypothesis)

- **Expected by AnglE**:
  - Continuous values between `0.0` and `1.0`
  - `1.0` = high similarity
  - `0.0` = low similarity

### 2. Incorrect Label Interpretation
The training code was passing NLI labels directly to the loss functions:
```python
# Original problematic code
ds_split = ds_split.map(lambda obj: {
    "text1": str(obj["premise"]), 
    "text2": str(obj["hypothesis"]), 
    "label": obj["label"]  # Direct use of categorical labels!
})
```

This caused several issues:
- Label `0` (entailment) was treated as `0.0` similarity (completely wrong - should be high similarity)
- Label `1` (neutral) was treated as `1.0` similarity (somewhat reasonable)
- Label `2` (contradiction) was treated as `2.0` similarity (out of expected range [0,1])

### 3. Loss Function Incompatibility
The AnglE loss functions (`cosine_loss`, `angle_loss`, `ibn_loss`) are designed for similarity/ranking tasks, not classification. They expect:
- Continuous similarity scores as ground truth
- Values properly scaled between 0 and 1
- Meaningful distance relationships between labels

## Changes Made

### File: `/mnt/c/Users/WIN11/Desktop/esedebug/ESE/train_moe.py`

#### Change 1: Fixed NLI Label Conversion (Line 362-376)
```python
# NEW CODE - Convert NLI labels to similarity scores
def convert_nli_to_similarity(obj):
    label_map = {0: 1.0, 1: 0.5, 2: 0.0}
    return {
        "text1": str(obj["premise"]), 
        "text2": str(obj["hypothesis"]), 
        "label": label_map.get(obj["label"], 0.5)  # default to 0.5 if label is missing
    }

ds_split = ds_split.map(convert_nli_to_similarity)
ds_split = ds_split.select_columns(["text1", "text2", "label"])
```

**Mapping Logic**:
- Entailment (0) → 1.0 (high similarity)
- Neutral (1) → 0.5 (medium similarity)
- Contradiction (2) → 0.0 (low similarity)

#### Change 2: Fixed Validation Data Normalization (Line 428-433)
```python
# Normalize STS scores from 0-5 range to 0-1 range
valid_ds_split = valid_ds_split.map(lambda obj: {
    "text1": str(obj["sentence1"]), 
    "text2": str(obj['sentence2']), 
    "label": float(obj.get("score", 0.0)) / 5.0  # normalize to [0, 1]
})
```

This ensures consistency between training and validation data ranges.

## Expected Impact

With these fixes:
1. **Training will now use correct similarity relationships** between sentence pairs
2. **Loss functions will receive properly scaled inputs** in the expected [0,1] range
3. **The model should learn meaningful embeddings** that capture semantic similarity
4. **Performance should improve significantly** and surpass the baseline models

## Recommendations

1. **Re-run training** with the fixed code
2. **Monitor the training loss** - it should decrease more smoothly now
3. **Validate on STS benchmarks** - scores should improve significantly
4. **Consider adjusting the label mapping** if needed:
   - Current mapping (1.0, 0.5, 0.0) is reasonable but can be tuned
   - Alternative: (0.9, 0.5, 0.1) to avoid extreme values

## Additional Notes

- The same issue likely exists in `train.py`, but as requested, only `train_moe.py` was modified
- This fix applies to all NLI-based training (BERT, BGE, and MoE variants)
- The validation set (STS benchmark) was already using similarity scores but needed normalization