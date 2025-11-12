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


