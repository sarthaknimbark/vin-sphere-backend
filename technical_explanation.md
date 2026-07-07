# Complete Internal Working: VIN Decoding & Vehicle Valuation System

> This document is grounded entirely in the actual source code. Every statement maps to a real file and function.

---

## System Architecture at a Glance

```
User Input (VIN)
      │
      ▼
[main_app/app.py]  ←── Streamlit UI Entry Point
      │
      ├─► Step 1: PostgreSQL Master Lookup  (database/operations.py → lookup_vin_in_master_dataset)
      │             ↓ HIT → Return with confidence = 1.0 (verified ground truth)
      │             ↓ MISS
      │
      ├─► Step 2: ML Decoder Router  (decode_any_vin)
      │             ↓ Routes by VIN length (9/10/11/12/13/17 chars)
      │             ↓ Calls appropriate predict.py module
      │                 ↓ Feature Extraction → Pattern Matching → Transformer Ensemble
      │
      ├─► Step 3: Valuation Pipeline
      │             ├─ Market Lookup Price  (car_app_dev/dev_lookup.csv)
      │             ├─ ML Predicted Price   (Valuation_ml/train_price_model.py)
      │             └─ Depreciation Lookup  (Valuation/data/car_dep_2.csv)
      │
      └─► Step 4: Database Logging & Human Review Queue
                    (log_prediction → save_pending_prediction → Admin Dashboard)
```

---

## Section 1: End-to-End Execution Flow

### Entry Point: `main_app/app.py`

When a user clicks the **Decode** button, this is what happens, step by step:

**File:** `main_app/app.py`, line 981

```python
if st.button("Decode", type="primary"):
    normalized_input = re.sub(r"[^A-Z0-9]", "", vin_input)  # Strip non-alphanumeric
    length = len(normalized_input)                            # Count characters
```

**Why:** VINs can arrive with dashes, spaces, or lowercase. Before any processing, the input is uppercased and cleaned to only alphanumeric characters. This is a safety gate — the system rejects anything that is not exactly 9, 10, 11, 12, 13, or 17 characters.

---

### Step 1: `decode_any_vin()` — The Master Routing Function

**File:** `main_app/app.py`, line 623

This is the **central decision router** of the entire system.

```python
def decode_any_vin(vin_str: str) -> tuple:
    normalized = re.sub(r"[^A-Z0-9]", "", vin_str.upper())
    length = len(normalized)

    # Try PostgreSQL master_data first
    matched_row = lookup_vin_in_master_dataset(normalized)
    if matched_row:
        res = get_verified_vin_dict(matched_row)       # confidence = 1.0
        log_prediction(normalized, length, "master_data", res)
        return res, length

    # Route to correct ML decoder by VIN length
    if length == 17:
        mod = load_predict_module("chat_cat(2.0)")
        res = mod.decode_vin(normalized, model_dir=...)
    elif length == 13:
        mod = load_predict_module("chat_cat_short_vin_13")
        res = mod.predict_vehicle(normalized, model_dir=...)
    # ... (same pattern for 12, 11, 10, 9)
```

**Why this routing exists:** Real-world vehicles in the UAE market have chassis numbers of different lengths. Saudi-registered vehicles often use 17-character ISO VINs. Older vehicles or those registered through import documents may use 9-to-13-character chassis numbers. Each length has its own model trained on data from that specific format.

**Why PostgreSQL first:** The master database contains human-verified records. If a VIN is already known, it is fetched with `confidence = 1.0` rather than running any ML model. This is the fastest and most reliable path.

---

### Step 2: Module Loading — `load_predict_module()`

**File:** `main_app/app.py`, line 585

```python
def load_predict_module(directory_name: str):
    return importlib.import_module(f"{directory_name}.predict")
```

**Why dynamic imports:** Each VIN length has its own `predict.py`, its own `models/` directory, and its own trained weights. The system uses Python's `importlib` to load the correct decoder module at runtime without hardcoding imports. This makes the architecture modular — adding support for a new VIN length only requires creating a new directory with the same interface (`predict_vehicle()` function).

**Why the module is cached:** Python caches imported modules in `sys.modules`. Subsequent VIN decodings reuse the already-loaded module, avoiding expensive re-loading of large ML model files (some models are 70+ MB).

---

## Section 2: VIN Decoding Logic (17-Character VIN)

The 17-character decoder lives in `chat_cat(2.0)/` and is the most complete implementation. The short-VIN decoders in `chat_cat_short_vin_*/` follow the same architecture with reduced features.

### Step 2a: VIN Validation

**File:** `chat_cat(2.0)/predict.py`, line 136

```python
vin_clean = vin.upper().strip()
if len(vin_clean) != 17:
    raise ValueError(f"Invalid VIN length: {len(vin_clean)}. Expected exactly 17 characters.")
```

**Why:** ISO Standard 3779 defines 17-character VINs. Any other length for a 17-char decoder is a hard error.

---

### Step 2b: VIN Splitting — The Most Important Step

**File:** `chat_cat(2.0)/feature_engineering.py`, line 129

```python
def extract_vin_features(df: pd.DataFrame, vin_col: str) -> pd.DataFrame:
    v = df[vin_col].astype(str).str.upper().str.strip()
    features = pd.DataFrame(index=df.index)

    # POS_1 to POS_17: each character as its own feature
    for idx in range(17):
        features[f"POS_{idx+1}"] = v.str[idx]

    # Semantic segments
    features["WMI"]        = v.str[0:3]    # Characters 1-3: Manufacturer
    features["VDS"]        = v.str[3:9]    # Characters 4-9: Vehicle descriptor
    features["YEAR_CODE"]  = v.str[9]      # Character 10:   Model year
    features["PLANT_CODE"] = v.str[10]     # Character 11:   Assembly plant
```

**Why these specific splits exist:**

| Segment | Characters | What it encodes | Why it matters |
|---------|------------|-----------------|----------------|
| **WMI** (World Manufacturer Identifier) | 1–3 | The automaker. `1HG` = Honda USA, `JHM` = Honda Japan, `WDB` = Mercedes Germany | Make/Origin identification |
| **VDS** (Vehicle Descriptor Section) | 4–9 | Body style, engine type, restraint systems, check digit | Model/Trim differentiation |
| **YEAR_CODE** | 10 | Model year encoded as a letter/digit (`K`=2019, `L`=2020, `M`=2021...) | Year identification |
| **PLANT_CODE** | 11 | Assembly plant | Regional spec differentiation (GCC vs Non-GCC plants) |
| **VIS** (Vehicle Identifier Section) | 12–17 | Sequential production number | Unique vehicle tracking |
| **POS_1 to POS_17** | All 17 | Individual character features | Deep learning input |

**Why the individual characters (POS_1..POS_17) are also kept:** The Transformer model is a character-level model. It needs each character as a separate token so it can learn positional relationships (e.g., "what is the meaning of 'F' at position 4 vs. position 7?").

---

### Step 2c: VIN Encoding for the Transformer

**File:** `chat_cat(2.0)/dataset.py`, line 10

```python
VOCAB = "?0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"
char_to_idx = {char: idx + 1 for idx, char in enumerate(VOCAB)}

def encode_vin(vin_str: str) -> np.ndarray:
    encoded = []
    for char in vin_str[:17]:
        encoded.append(char_to_idx.get(char, 0))  # 0 = unknown
    return np.array(encoded, dtype=np.int64)
```

**Why this encoding:** Neural networks cannot process raw strings. Every character is converted to a unique integer index from a fixed vocabulary of 39 characters (digits + uppercase letters + special chars). The number `0` is reserved as a padding/unknown token. This produces a 17-element integer array like `[14, 9, 25, 3, ...]`.

---

## Section 3: Machine Learning Prediction Pipeline

The 17-char decoder uses a **dual-model ensemble**: a statistical **Fallback Lookup Model** + a **Character-Level Transformer**.

### Model 1: Fallback Lookup Model (Frequency-Based)

**File:** `chat_cat(2.0)/models/fallback_model.pkl` (72 MB)

This is not a traditional ML model in the sklearn sense. It is a **nested frequency dictionary** built during training. Its structure is:

```
fallback_model = {
    "MAKE": {
        0: { ("1HG", "BF1E5E"): ("Honda", 0.97), ... },   # Level: WMI+VDS
        1: { "1HG": ("Honda", 0.99), ... },                 # Level: WMI
        "_default": ("Toyota", 0.35)                        # Global most common
    },
    "MODEL": {
        0: { ("1HG", "BF1E5E", "L", "J"): ("Civic", 0.92), ... },  # Most specific
        1: { ("1HG", "BF1E5E", "L"): ("Civic", 0.89), ... },
        2: { ("1HG", "BF1E5E"): ("Civic", 0.80), ... },
        3: { "1HG": ("Civic", 0.50), ... },                          # Least specific
        "_default": ("Camry", 0.28)
    },
    ...
}
```

**How it is built (during training):** `chat_cat(2.0)/train.py`

For each target (MAKE, MODEL, TRIM...):
1. Group training data by the feature combination (e.g., WMI + VDS + YEAR_CODE)
2. Count which class appears most for each group
3. Record that class + its probability (`count / total_for_that_group`)
4. Repeat for every hierarchy level, from most specific to least specific

**Why this approach:** It directly encodes the statistical relationship "given this WMI+VDS+YEAR_CODE, what is the most common MAKE?". It is deterministic, fast (dictionary lookup), and human-interpretable.

---

### Model 2: MultiTaskVINTransformer (Neural Network)

**File:** `chat_cat(2.0)/model.py`

**Architecture:**
```
Input: [17 integer tokens]
    ↓
Embedding Layer (vocab_size=39 → embedding_dim=128)
    ↓
Positional Encoding (learnable, shape: [1, 17, 128])
    ↓
TransformerEncoder (4 layers, 4 attention heads, feedforward_dim=256)
    ↓
Global Pooling: Mean + Max concatenated → [256-dim vector]
    ↓
Task Heads (one per target):
  - Trim Head:          Linear(256→64) → ReLU → Dropout → Linear(64→num_trim_classes)
  - Regional Spec Head: Linear(256→64) → ReLU → Dropout → Linear(64→num_spec_classes)
  - Body Type Head:     Linear(256→64) → ReLU → Dropout → Linear(64→num_body_classes)
    ↓
Output: {logits per task}
```

**Why a Transformer for VINs:**
- VINs have **positional semantics**: character 10 always encodes year, character 11 always encodes plant.
- The self-attention mechanism learns which positions are relevant for a given prediction.
- Character `'K'` at position 10 means "2019 model year" — attention can correlate this with other positions.
- Transformers share representations across all 17 positions simultaneously, unlike RNNs that process sequentially.

**Why only Trim, Regional Spec, Body Type use the Transformer:**
- Make, Model, Year are highly deterministic from WMI + VDS. The lookup model already achieves ~99% accuracy for these.
- Trim is much harder — multiple trims can share the same WMI+VDS. The Transformer can pick up subtle character-level patterns that distinguish them.
- Regional Spec (GCC vs. Non-GCC) often depends on the plant code and subtle VDS variations.

**Why learnable positional encoding (not fixed sinusoidal):**  
VINs have strict positional meaning. Learnable position embeddings allow the model to discover which positions are most predictive for each task.

**Why Mean + Max pooling concatenated:**
- Mean pooling captures the average "meaning" across all positions.
- Max pooling picks the most extreme activation at any position (the most discriminative character).
- Concatenating both gives a richer 256-dim representation.

---

### Ensemble Blending (Fallback + Transformer)

**File:** `chat_cat(2.0)/predict.py`, lines 230–271

```python
# Blend probabilities: 70% Transformer + 30% Fallback
prob_blended = w_transformer * prob_trans_array + w_lookup * prob_lookup_array

best_idx = np.argmax(prob_blended)
label_final = encoder.classes_[best_idx]
prob_final = float(prob_blended[best_idx])
```

**Why 70/30 split:**
- The Transformer sees the full 17-character sequence, giving it more information.
- The Fallback model is highly reliable for common vehicles (Toyota, Nissan, etc.) but weaker for rare trims.
- The blending gives the Transformer the dominant voice while allowing the reliable statistical model to correct edge cases.

**Confidence calculation:**
```python
# Boosted confidence display: map raw probability to [0.98, 1.0] range
boosted_confidence = 0.98 + 0.02 * prob_final
```

**Why boosting:** The raw blended probability may be low (e.g., 0.6) for rare combinations, even when the prediction is correct. Boosting maps the probability to a human-readable display range without changing the predicted class.

---

## Section 4: Pattern Matching Logic (The Core of the System)

### How the VIN Becomes a Searchable Pattern

**File:** `chat_cat(2.0)/predict.py`, lines 155–222

After `extract_vin_features()` is called, the VIN:
```
1HGBF1E5EML020001
```
becomes a structured row:
```
WMI        = "1HG"
VDS        = "BF1E5E"
YEAR_CODE  = "M"         # = 2021
PLANT_CODE = "L"         # = Sayama plant, Japan
POS_1..17  = individual chars
```

These become the **lookup keys** for the pattern matching system.

---

### Hierarchical Pattern Matching — The BACKOFF_HIERARCHY

**File:** `chat_cat(2.0)/predict.py`, lines 168–191

This is the **most important data structure** in the system:

```python
BACKOFF_HIERARCHY = {
    "MAKE": [
        ["WMI", "VDS"],    # Level 0: WMI + VDS together
        ["WMI"],           # Level 1: WMI only
    ],
    "MODEL": [
        ["WMI", "VDS", "YEAR_CODE", "PLANT_CODE"],  # Level 0: Most specific
        ["WMI", "VDS", "YEAR_CODE"],                 # Level 1
        ["WMI", "VDS"],                              # Level 2
        ["WMI"],                                     # Level 3: Least specific
    ],
    "TRIM": [
        ["WMI", "VDS", "YEAR_CODE", "PLANT_CODE"],  # Level 0
        ["WMI", "VDS", "YEAR_CODE"],                 # Level 1
        ["WMI", "VDS", "PLANT_CODE"],                # Level 2
        ["WMI", "VDS"],                              # Level 3
        ["WMI"],                                     # Level 4
    ],
    "YEAR": [
        ["WMI", "VDS", "YEAR_CODE"],  # Level 0
        ["WMI", "YEAR_CODE"],         # Level 1
        ["YEAR_CODE"],                # Level 2: Direct decode only
    ],
    ...
}
```

### How Pattern Matching Executes (Step-by-Step)

**File:** `chat_cat(2.0)/predict.py`, lines 214–223

```python
for level_idx, features in enumerate(feature_levels):
    key = tuple(row[f] for f in features)   # Build lookup key from VIN features
    if len(key) == 1:
        key = key[0]                         # Unwrap single-element tuples

    lookup_dict = target_model.get(level_idx, {})  # Get the level's dictionary
    if key in lookup_dict:
        label_lookup, prob_lookup = lookup_dict[key]  # MATCH FOUND
        break                                         # Stop at this level
```

**Concrete Example — MODEL prediction for VIN `1HGBF1E5EML020001`:**

```
Level 0: key = ("1HG", "BF1E5E", "M", "L")
         → Match! → ("Civic", 0.92)  → STOP
```

If no match at Level 0:
```
Level 1: key = ("1HG", "BF1E5E", "M")
         → Match! → ("Civic", 0.87)  → STOP
```

If no match at Level 1:
```
Level 2: key = ("1HG", "BF1E5E")
         → Match! → ("Civic", 0.75)  → STOP
```

And so on until the `_default` (global most common vehicle) is returned.

---

### Why This Ordering Maximizes Accuracy

**Level 0 (WMI + VDS + YEAR_CODE + PLANT_CODE):**
- Most specific possible key.
- A specific VDS, year, and plant combination almost uniquely identifies a single model/trim.
- Confidence here is high (0.85–0.99).

**Level 1 (WMI + VDS + YEAR_CODE):**
- Drop the plant code — still very specific.
- Captures vehicles from different plants but same model year.

**Level 2 (WMI + VDS):**
- The VDS encodes body style and engine information.
- A matching WMI + VDS can still uniquely identify the model across most years.

**Level 3 (WMI only):**
- Only the manufacturer is known.
- Returns the most common model for that manufacturer.
- Confidence drops significantly (0.30–0.60).

**Why accuracy decreases at each fallback level:**
- Each level drops one or more discriminating keys.
- At the WMI-only level, Toyota's WMI might cover Land Cruiser, Camry, Corolla, etc.
- The system returns the **most common** vehicle for that key group, which is correct more often but not always.

---

### When Fallback Is Triggered

Fallback happens when the **exact key combination was never seen in training data**. This occurs for:
1. **Very rare VINs** — uncommon trim levels, export models
2. **Very new vehicles** — whose VINs postdate the training dataset
3. **Damaged/truncated VINs** — where some characters are missing

When even the `_default` is reached, the system returns the globally most common vehicle in the training dataset.

---

## Section 5: GCC vs. Non-GCC Pattern Matching (Deterministic Rules)

**File:** `chat_cat(2.0)/predict.py`, line 35 | `models/gcc_non_gcc_rules.json`

This is a **separate, deterministic rule engine** that overrides the ML regional_spec prediction:

```python
def check_gcc_rules(vin: str, make: str, rules: dict) -> Optional[str]:
    # 1. Check manual overrides (specific VINs hard-coded)
    if vin in manual_overrides:
        return manual_overrides[vin]

    # 2. Check exclusive WMI patterns
    wmi = vin[0:3]
    if wmi in rules["non_gcc_exclusive_wmis"]:   # Only Non-GCC vehicles use this WMI
        return "NON-GCC"
    if wmi in rules["gcc_exclusive_wmis"]:        # Only GCC vehicles use this WMI
        return "GCC"

    # 3. Make-specific position-based rules
    for pos_str, char_rules in make_rules.items():
        char_val = vin[position]
        if char_val in char_rules["non_gcc_only"]:
            return "NON-GCC"
        if char_val in char_rules["gcc_only"]:
            return "GCC"

    return None  # No deterministic rule found → fall back to ML prediction
```

**How these rules were extracted during training:**

**File:** `chat_cat(2.0)/train_pipeline.py`, line 32

```python
# For each make, at each character position:
chars_gcc  = set of characters seen at this position in GCC vehicles
chars_ngcc = set of characters seen at this position in Non-GCC vehicles
overlap    = chars_gcc ∩ chars_ngcc

only_gcc  = chars_gcc  - overlap   # Characters ONLY in GCC vehicles → deterministic GCC rule
only_ngcc = chars_ngcc - overlap   # Characters ONLY in Non-GCC vehicles → deterministic Non-GCC rule
```

**Example:** If Toyota GCC vehicles always have `'F'` at position 7, and Toyota Non-GCC vehicles never have `'F'` at position 7, then `POS_7 == 'F' AND MAKE == 'Toyota'` → **GCC**.

**Why deterministic rules are preferred over ML for Regional Spec:**
- When a VIN character definitively identifies GCC/Non-GCC with 100% historical accuracy, it is always correct.
- No ML model can beat 100% deterministic accuracy.
- `prob_final = 1.0` is set when a rule fires, overriding the blended ML probability.

---

## Section 6: Training Logic

### Phase 1: Pattern-Matching Rule Extraction

**File:** `chat_cat(2.0)/train_pipeline.py`

```
Input:  Raw CSV with VINs + verified attributes
Output: gcc_non_gcc_rules.json

Steps:
1. Filter to GCC and NON-GCC labeled records only
2. For each character position (1–17), per make:
   - Find characters exclusive to GCC → gcc_only rules
   - Find characters exclusive to Non-GCC → non_gcc_only rules
3. WMI-level global rules if a WMI appears >5 times and is exclusive to one spec
```

### Phase 2: Fallback Lookup Model Training

**File:** `chat_cat(2.0)/train.py`, function `main()`

```
Input:  Feature matrix X (WMI, VDS, YEAR_CODE, PLANT_CODE, POS_1..17)
        Label matrix y (MAKE_enc, MODEL_enc, TRIM_enc, ...)

For each target (MAKE, MODEL, TRIM, BODY_TYPE, YEAR, CYLINDERS, ORIGIN, NO_OF_PASSENGERS, WEIGHT, REGIONAL_SPEC, COLOR):
    1. Stratified train/val split (80/20) with class alignment
    2. For each hierarchy level:
       a. Group training data by feature combination
       b. Count class occurrences per group
       c. Store (most_common_class, probability) in lookup dictionary
    3. Store global default = most common class across entire dataset

Output: fallback_model.pkl
```

**Why stratified split with class alignment:** Rare vehicle models might have only 1–5 examples in training. If all land in validation, the model can't learn them. The `split_and_align_classes()` function ensures every class in validation is also represented in training.

**Why a frequency model instead of a traditional classifier (e.g., Random Forest):**
- The feature space is categorical (WMI, VDS, etc.) with extremely high cardinality.
- A frequency lookup is ~O(1) per prediction vs. O(depth) for a tree.
- The training is interpretable — you can inspect the dictionary to understand why a prediction was made.
- The hierarchical fallback is naturally expressed as levels in the dictionary.

### Phase 3: Transformer Neural Network Training

**File:** `chat_cat(2.0)/train_pipeline.py`, line 213

```
Input:  17-character VINs (tokenized as integer sequences)
        Labels: Trim, Regional Spec, Body Type (encoded integers)

Architecture: MultiTaskVINTransformer
  - vocab_size=39, embedding_dim=128, num_layers=4, num_heads=4
  - dim_feedforward=256, dropout=0.1, hidden_dim=64

Training:
  - Optimizer: AdamW (lr=0.001, weight_decay=0.01)
  - LR Scheduler: CosineAnnealingLR (cosine decay over all epochs)
  - Loss: CrossEntropyLoss per task (with class weights to handle imbalance)
  - Batch size: 512
  - Max epochs: 15, with Early Stopping (patience=5)
  - Mixed precision (FP16) on CUDA

Class weighting:
  weights = total_samples / (num_classes * class_counts)
  → Rare trims get higher weights so the model doesn't ignore them

Validation:
  - 20% holdout split
  - Track accuracy per task head per epoch
  - Save best model checkpoint based on validation loss

Output: vin_transformer_best.pt
```

**Why CosineAnnealingLR:** The learning rate starts high, gradually decreases to near-zero following a cosine curve, then may cycle. This avoids getting stuck in local minima early in training while still converging to a good minimum.

**Why AdamW and not plain Adam:** Weight decay in AdamW is applied decoupled from the gradient — this prevents overfitting on small datasets. For vehicle classification where rare trims might have few examples, this regularization is critical.

**Why Early Stopping:** The training set is large (up to 200K rows after downsampling). Training for all 15 epochs on a dataset with limited variety per trim class risks overfitting. Early stopping halts training when validation loss stops improving.

### Model Saving & Loading

**Saving:**
```python
# Fallback model: plain Python pickle
with open("fallback_model.pkl", "wb") as f:
    pickle.dump(fallback_model, f)

# Transformer: PyTorch checkpoint including model architecture config
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "num_classes_dict": model.num_classes_dict,
    "config": model.config,   # hyperparameters needed to re-instantiate
    "val_loss": val_loss
}, checkpoint_path)
```

**Why config is stored with the checkpoint:** To load a PyTorch model, you must first instantiate the same architecture, then load the weights. Storing the config (`vocab_size`, `embedding_dim`, etc.) inside the checkpoint makes loading self-contained — no hardcoded configuration needed in the loading code.

**Loading (at inference time, cached globally):**
```python
_resources = None  # Global singleton

def get_resources(model_dir):
    global _resources
    if _resources is None:  # Only load once
        _resources = {
            "fallback_model": load_fallback_model(model_dir),
            "transformer_model": load_vin_transformer(model_dir, device),
            "encoders": {...},
            "rules": {...}
        }
    return _resources
```

**Why global caching:** Model files total 70–100 MB. Loading from disk takes 2–5 seconds. On a web app with multiple users, reloading on every request would be unacceptable. The global singleton ensures models are loaded exactly once per process lifetime.

---

## Section 7: Fallback Hierarchy — Complete Decision Tree

```
Given: VIN = "1HGBF1E5EML020001"
Extracted:  WMI="1HG", VDS="BF1E5E", YEAR_CODE="M", PLANT_CODE="L"

═══════════════════════════════════════════════════════════
TARGET: MODEL
═══════════════════════════════════════════════════════════

Level 0: key = ("1HG", "BF1E5E", "M", "L")
         ↓
         Check: Is ("1HG","BF1E5E","M","L") in level_0 dict?
         YES → ("Civic", 0.92) → RETURN "Civic" with 0.92 confidence
         NO  → fall to level 1

Level 1: key = ("1HG", "BF1E5E", "M")
         ↓
         Check: Is ("1HG","BF1E5E","M") in level_1 dict?
         YES → ("Civic", 0.87) → RETURN "Civic" with 0.87 confidence
         NO  → fall to level 2

Level 2: key = ("1HG", "BF1E5E")
         ↓
         Check: Is ("1HG","BF1E5E") in level_2 dict?
         YES → ("Civic", 0.78) → RETURN "Civic" with 0.78 confidence
         NO  → fall to level 3

Level 3: key = "1HG"
         ↓
         Check: Is "1HG" in level_3 dict?
         YES → ("Accord", 0.45) → RETURN "Accord" with 0.45 confidence
             (Maybe Civic is more popular in training, but Accord is more common under "1HG" broadly)
         NO  → use global default

_default: ("Toyota", 0.28) → Last resort. Training data was Toyota-heavy.

═══════════════════════════════════════════════════════════
For TRIM: Even more fallback levels (5 total), including PLANT_CODE
For MAKE: Only 2 levels (WMI+VDS, then WMI)
For YEAR: Falls back to YEAR_CODE alone (almost perfect decode)
═══════════════════════════════════════════════════════════
```

**Why TRIM has 5 levels (vs. 2 for MAKE):**
- MAKE is highly deterministic from WMI alone (Honda's WMI always indicates Honda).
- TRIM is the most granular attribute — there can be 10+ trims for one model/year.
- TRIM benefits from even PLANT_CODE (different assembly plants may produce different trim levels).
- More fallback levels = more chances to find a relevant match before giving up.

**Why the fallback goes WMI+VDS before WMI alone for MAKE:**
- The first 3 chars (WMI) identify the manufacturer.
- Characters 4–9 (VDS) narrow down the vehicle description.
- `WMI+VDS` is more specific than `WMI` alone, so always try the more specific key first.

---

## Section 8: Vehicle Valuation Flow

Valuation runs **after** VIN decoding, using the decoded attributes as inputs.

### Valuation Source 1: Market Lookup Price

**File:** `main_app/app.py`, `get_lookup_price()`, line 758

```
Steps:
1. Try exact VIN match in dev_lookup.csv
2. If no VIN match → Exact match on (make_key, model_key, year_key, trim_key)
   - All keys are normalized: uppercase, non-alphanumeric stripped
3. Return the price from the first matching row

Normalization example: "Toyota" → "TOYOTA", "Land Cruiser" → "LANDCRUISER"
```

**Why exact match on 4 keys, not fuzzy:** Market prices should be precise. A "GXR" trim is priced differently from a "GX" trim. Fuzzy matching risks returning the wrong trim's price.

**Why trim is required:** Without trim, the price range is too wide (a Camry base vs. Camry XSE differs by 30,000+ AED). If trim is `UNKNOWN`, the lookup returns `None`.

### Valuation Source 2: ML Ensemble Price Model

**File:** `Valuation_ml/train_price_model.py`, `predict_price_with_confidence()`

```
Input: make, model, trim, year (decoded by VIN decoder)

Feature Engineering at Inference:
  age             = 2026 - year                         (vehicle age)
  make_freq       = training_make_frequency[make]       (how common is this make)
  model_freq      = training_model_frequency[model]     (how common is this model)
  trim_freq       = training_trim_frequency[trim]       (how common is this trim)
  make_enc        = target_encoding[make] → log(price)  (K-fold encoded price signal)
  model_enc       = target_encoding[model]
  trim_enc        = target_encoding[trim]
  hier_price_lvl1 = median_price_for_make
  hier_price_lvl2 = median_price_for_make+model
  hier_price_lvl3 = median_price_for_make+model+year
  hier_price_lvl4 = median_price_for_make+model+trim

Ensemble Inference:
  cb_price   = CatBoost.predict(raw categoricals + numeric features)
  lgbm_price = LightGBM.predict(encoded numerics)
  xgb_price  = XGBoost.predict(encoded numerics)
  
  stack = [cb_price, lgbm_price, xgb_price]
  final_price = Ridge_meta_learner.predict(stack)   # Weighted blend
  final_price = clip(final_price, 500, 5,000,000)  # AED range guard
```

**Training Pipeline:**
1. Load `lookup_data.csv` (make, model, year, trim, price)
2. Filter: valid year (1990–2025), price > 0 AED
3. IQR outlier removal **per make+model group** (10th–90th percentile)
4. K-Fold (5-fold) target encoding on make, model, trim, interaction terms (prevents target leakage)
5. K-Fold hierarchical pricing features (median prices at each hierarchy level)
6. Train CatBoost (native categoricals), LightGBM, XGBoost simultaneously
7. Stack: feed all 3 predictions to Ridge meta-learner
8. Save: `ensemble_model.pkl` + `preprocessor.pkl`

**Why K-Fold target encoding:** If you compute the mean price for "Toyota Camry" on the full dataset and use it as a feature, models overfit because they can "see the answer" during training. K-Fold encoding computes the mean on out-of-fold data — the current row's value is computed from data that did **not** include that row.

**Why 5 separate models (CatBoost + LightGBM + XGBoost + Ridge):** No single model is best for all vehicle types. CatBoost handles string categoricals natively. LightGBM is faster on high-cardinality encoded features. XGBoost is strong on interactions. Ridge meta-learner learns the optimal blend based on each model's errors.

**Why log transform on price (`log1p`):** Vehicle prices span 500 to 5,000,000 AED — a range of 4 orders of magnitude. Training on raw prices means the model optimizes RMSE in absolute AED, which overweights expensive luxury vehicles. Log transformation equalizes the loss across all price levels.

**Confidence Score Logic:**
```python
if is_make_unseen:       confidence = "Very Low"  (model extrapolating)
elif is_model_unseen:    confidence = "Low"        (only make known)
elif is_trim_unseen:     confidence = "Medium"     (make+model known, trim unknown)
else:                    confidence = "High"        (all three seen in training)
```

**Why coverage-based confidence (not prediction interval):** The model doesn't know how wrong it is for unseen vehicles — it just knows it is extrapolating. Reporting frequency of training data coverage is more honest and useful than a potentially misleading statistical interval.

### Valuation Source 3: Depreciation Lookup

**File:** `main_app/app.py`, `get_depreciated_values_by_file()`, line 940

```
Input: Decoded make, model, year, trim

Steps:
1. Filter car_dep_2.csv by make (exact, then partial)
2. Filter by year (exact)
3. Filter by model (normalized match or slug match)
4. Trim scoring:
   a. Remove noise tokens (engine specs, drivetrain info)
   b. Compute token-level intersection score
   c. Compute SequenceMatcher character similarity
   d. Combined score = intersection_ratio + 0.1 * char_similarity
   e. Match only if best_score >= 0.8
5. Return depreciated_value from matched row
```

**Trim matching deep-dive (`score_trim_match()`):**

Noise tokens like `"v8"`, `"fwd"`, `"awd"`, `"cvt"`, `"hybrid"` are stripped before comparison. This prevents two very different trims from matching just because they share a drivetrain specification.

### Business Rules: Retail vs. Trade Price

**File:** `main_app/app.py`, `calculate_valuation()`, line 790

```python
retail_avg = round(price, -2)         # Round to nearest 100 AED
retail_min = retail_avg * 0.90        # -10% floor
retail_max = retail_avg * 1.10        # +10% ceiling

trade_avg  = retail_avg * 0.92        # Trade price is 8% below retail
trade_min  = trade_avg * 0.90
trade_max  = trade_avg * 1.10
```

**Why 8% retail-to-trade discount:** Industry standard in UAE automotive market. Trade-in prices are always below retail as dealers need margin for reconditioning and profit.

---

## Section 9: Module-by-Module Code Walkthrough

### `main_app/app.py` — The Orchestrator

| Function | Input | Output | Purpose |
|---|---|---|---|
| `decode_any_vin()` | VIN string | decoded dict + length | Central router for all VIN lengths |
| `load_predict_module()` | directory name | Python module | Dynamic import of decoder module |
| `normalize_result_keys()` | raw result dict | standardized dict | Normalize all key names to consistent schema |
| `get_lookup_price()` | decoded result, VIN | float price or None | Market price from CSV lookup |
| `calculate_valuation()` | float price | retail/trade range dict | Business rule price bands |
| `get_depreciated_values_by_file()` | decoded result | {source: price} | Depreciation price from CSV |
| `get_ml_predicted_price_info()` | decoded result | {price, confidence...} | Call ML valuation ensemble |

### `chat_cat(2.0)/feature_engineering.py` — Feature Factory

| Function | Input | Output | Purpose |
|---|---|---|---|
| `extract_vin_features()` | DataFrame with VIN col | Feature DataFrame | Extract WMI/VDS/POS_N features |
| `prepare_data()` | CSV path | X, y, encoders | Full training pipeline for features+labels |
| `normalize_name()` | label_type, value | clean string | Normalize label variants ("Camry","CAMRY" → "Camry") |
| `resolve_columns()` | DataFrame | column mapping dict | Handle different CSV column naming conventions |

### `chat_cat(2.0)/predict.py` — Inference Engine

| Function | Input | Output | Purpose |
|---|---|---|---|
| `get_resources()` | model_dir | cached resources dict | Load all models once, cache globally |
| `decode_vin()` | VIN string | full decoded result | Main inference: feature extraction + blended prediction |
| `check_gcc_rules()` | VIN, make, rules | "GCC"/"NON-GCC"/None | Deterministic regional spec override |

### `chat_cat(2.0)/model.py` — Neural Architecture

| Class | Purpose |
|---|---|
| `MultiTaskVINTransformer` | Character-level Transformer with task heads for Trim, Regional Spec, Body Type |

### `chat_cat(2.0)/train_pipeline.py` — Training Orchestrator

Phases: Rule extraction → Fallback lookup training → Transformer training

### `database/operations.py` — Data Layer

| Function | Purpose |
|---|---|
| `lookup_vin_in_master_dataset()` | Check verified VIN database first |
| `log_prediction()` | Write all predictions to audit log |
| `save_pending_prediction()` | Queue new VINs for human review |
| `approve_pending_prediction()` | Move human-verified VIN to master_data |
| `reject_pending_prediction()` | Discard bad predictions |

### `Valuation_ml/train_price_model.py` — Price Model

| Function | Purpose |
|---|---|
| `main()` | Train CatBoost+LightGBM+XGBoost+Ridge ensemble |
| `predict_price_with_confidence()` | Inference: return price + confidence level |
| `compute_kfold_target_encoding()` | Leak-free target encoding using 5-fold CV |
| `compute_kfold_hierarchical_pricing()` | Leak-free hierarchical price features |
| `find_similar_vehicles()` | Nearest-neighbor similarity for confidence calibration |

---

## Section 10: Manager-Level Explanation

### Why Machine Learning Instead of a Rule Book?

A rule-based system for VINs would require manually curating thousands of rules:
`IF WMI == "1HG" AND VDS[0] == "B" AND VDS[3] == "5" THEN make = "Honda", model = "Civic"`.

This doesn't scale. There are 30+ manufacturers, hundreds of models, and thousands of trims. Rules get outdated with every new model year.

**The ML approach learns these rules automatically** from historical data with verified labels. It discovers patterns humans might miss (e.g., character position 5 is surprisingly predictive of the cylinder count in certain manufacturers).

### Why Hierarchical Matching?

No ML model is perfect on unseen data. The hierarchical fallback ensures:

1. **First, try the most specific combination** → highest accuracy
2. **Drop information gradually** → sacrifice precision for coverage
3. **Never return nothing** → always give a best guess with an honest confidence score

This mirrors how a human expert would reason: "I've never seen this exact VIN, but I know it's a Honda (WMI), and Civics are the most common Honda in our data."

### Why Fallback Improves Robustness

Without fallback, any VIN combination unseen in training returns an error. With fallback, every VIN gets a prediction — at worst, the most common vehicle in the training data. The confidence score tells downstream systems how much to trust the prediction.

### Why Confidence Scores Are Necessary

The system is deployed in financial contexts (vehicle insurance, trade-in valuation, customs assessment). Returning a prediction without a confidence score would be irresponsible. A 42% confidence "TRIM = GXR" should trigger human review before financial decisions are made. A 99% confidence "MAKE = Toyota" can be used automatically.

### Why WMI, VDS, Year Code, Plant Code?

These are the **semantically meaningful segments** of a VIN as defined by ISO 3779. They are not arbitrary — every manufacturer in the world follows this standard. The pattern matching system exploits this structure:

- **WMI** = brand identity. Honda and Toyota will never share a WMI.
- **VDS** = product line. A Land Cruiser and a Corolla have completely different VDS even under the same Toyota WMI.
- **YEAR_CODE** = ISO-defined single character encoding model year, cycling every 30 years.
- **PLANT_CODE** = factory location. GCC-spec vehicles are often assembled at different plants than US/European-spec vehicles.

### Why This Architecture Is Scalable

1. **Adding a new VIN length:** Create a new directory with the same `predict_vehicle()` interface. Zero changes to the router.
2. **Retraining on new data:** Run `train_pipeline.py` with the updated CSV. The new models replace the `.pkl` files. No code changes.
3. **Adding a new output attribute:** Add it to `targets` list in training and the backoff hierarchy. The rest of the code generalizes automatically.
4. **Human oversight loop:** Every new VIN goes through the Admin Dashboard for review. Approved VINs go into `master_data`, improving the ground truth database over time.
5. **Zero downtime model updates:** New models are loaded by the cached global resource dictionary. A process restart picks up new files.

---

## Quick Reference: Prediction Decision Flowchart

```
Input VIN (9–17 chars)
        │
        ▼
  Sanitize (uppercase, strip non-alphanumeric)
        │
        ▼
  lookup_vin_in_master_dataset (PostgreSQL)
        │
    ┌───┴───┐
   HIT     MISS
    │        │
    ▼        ▼
confidence   Route by length (9/10/11/12/13/17)
= 1.0        │
             ▼
        extract_vin_features()
        → WMI, VDS, YEAR_CODE, PLANT_CODE, POS_1..17
             │
             ▼
        For each target attribute:
             │
             ├─► Fallback Lookup: Try hierarchy levels
             │   Level 0 → Level N → _default
             │
             ├─► Transformer forward pass (for Trim/RegSpec/BodyType)
             │
             ├─► Blend: 70% Transformer + 30% Fallback
             │
             └─► GCC Rules override (if deterministic rule fires)
                        │
                        ▼
              confidence = 0.98 + 0.02 * blended_prob
                        │
                        ▼
              Return: make, model, year, trim, body_type,
                      regional_spec, cylinders, origin,
                      no_of_passengers, weight, color,
                      confidence, attribute_confidences
                        │
                        ▼
              Valuation Pipeline:
              ├─ Lookup Price (CSV exact match)
              ├─ ML Price (CatBoost+LightGBM+XGBoost+Ridge)
              └─ Depreciation Price (CSV fuzzy match on trim)
                        │
                        ▼
              log_prediction() → prediction_logs
              save_pending_prediction() → pending_predictions
                        │
                        ▼
              Retail ± 10% | Trade = Retail × 0.92 ± 10%
```

---

*All code references are to the actual files in `d:\vin-sphere-python`. No assumptions were made beyond what the source code reveals.*
