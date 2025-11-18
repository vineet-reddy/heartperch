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




## Upstream Library Issues

### 4. surfperch model path bug in perch_hoplite
**Status:** Workaround implemented in `pulse/inference/perch_embedder.py`

**Issue:**
- `perch_hoplite.zoo.model_configs` has incorrect SURFPERCH_SLUG: `'google/surfperch/tensorFlow2/TensorFlow2/1'`
- Path has duplicate directory names (lowercase then uppercase)
- When combined with `tfhub_version=1`, creates invalid path: `.../TensorFlow2/1/1`
- Correct Kaggle path is: `'google/surfperch/TensorFlow2/1'`

**Workaround:** 
- In `perch_embedder.py`, we patch the config before loading:
  ```python
  preset_info.model_config.tfhub_path = 'google/surfperch/TensorFlow2/1'
  preset_info.model_config.tfhub_version = None  # Already in path
  ```

**Upstream Fix Needed:** File PR to google-research/perch to fix `perch_hoplite/zoo/hub.py` line defining SURFPERCH_SLUG
