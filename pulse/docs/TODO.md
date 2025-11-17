# TODO / Known Issues

## Data Quality Issues

### 1. Missing Recordings (Filename Mismatch)
**Status:** Documented, Low Priority

**Issue:** 
- Lost 1 patient (ID: 50321) and 45 recordings total
- Files in GCS bucket have `_1`, `_2`, `_3` suffixes for multiple recordings at same location
- Code in `data/circor.py` line 91 looks for `{patient_id}_{location}.wav` without handling numbered variants
- Example: CSV lists "AV+AV" but bucket has `50321_AV_1.wav`, `50321_AV_2.wav`

**Impact:**
- Training data reduced from 3163 -> 3118 recordings (1.4% loss)
- 942 -> 941 patients
- No label corruption or misalignment detected

**Fix:** Update `circor.py` to handle `{patient_id}_{location}_N.wav` patterns


### 2. Remove dead continued pretraining code from chirp/train/train_utils.py and chirp/config_utils.py (from last iteration on heartperch).


## Model Provenance Issues

### 3. Embedding files don't store model name
**Status:** Cosmetic issue, Low Priority

**Issue:**
- Multiple models can produce same embedding dimensions (e.g., perch_8 and surfperch both output 1280-dim)
- Embedding .npz files don't save which model generated them
- Scripts like `baseline_comparison.py` can't determine exact model from embeddings alone
- Results in ambiguous labels like "Perch (perch_8/surfperch, 1280-dim)" in CSV outputs

**Current Workaround:** Files are saved in model-specific folders (e.g., `./embeddings_binary/perch_8/`) so provenance is implicit from path

**Fix:** Save `model_name` field in embedding .npz files in `pulse/inference/embed_heart_dataset.py` (add to `np.savez_compressed()` call)
