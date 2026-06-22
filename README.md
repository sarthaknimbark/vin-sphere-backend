# VIN Sphere

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vin-decoder.streamlit.app/)

**Live Deployment**: [vin-decoder.streamlit.app](https://vin-decoder.streamlit.app/)

A unified vehicle chassis decoder, historical market lookup, and ML-based pricing dashboard.

## The Business Problem

In the GCC (Gulf Cooperation Council) automotive ecosystem, used car dealerships, insurance underwriters, fleet managers, and financial institutions face significant operational and financial bottlenecks when evaluating vehicles. This stems from three core challenges:

### 1. The Gray Market and Non-Standard "Short Chassis" Numbers
A substantial portion of the GCC vehicle fleet consists of gray-market imports—primarily from Japan (JDM), Europe, and North America. Unlike standard vehicles that feature a 17-character ISO-compliant Vehicle Identification Number (VIN), these imports often utilize non-standard, regional "short chassis numbers" (ranging from 9 to 13 characters). 
* **Decoding Failure**: Commercial VIN decoding services are hardcoded for 17-character formats. When presented with a short chassis number like `RZN1470016178`, they fail completely.
* **Manual Bottleneck**: Staff must manually research Japanese auction sheets, regional catalog PDFs, or historic spec sheets to identify basic vehicle parameters (e.g., engine displacement, body type, origin, or regional spec compatibility).

### 2. High Cost and Latency of External API Dependencies
Automotive platforms often rely on third-party SaaS decoders and valuation engines.
* **Prohibitive Cost**: Querying these external databases costs between $0.50 to $2.00+ per lookup. For operations processing thousands of potential leads, trade-in requests, or auction listings daily, these fees erode profit margins.
* **Reliability Risks**: External API requests introduce latency, depend on internet connectivity, and present security risks by transmitting proprietary inventory or customer data to third-party endpoints.

### 3. Sparse Historical Transactions & Multi-Modal Pricing Ambiguity
Valuing a used car in the GCC is not a straightforward regression problem. It requires reconciling three distinct, often conflicting pricing signals:
* **The "GCC Spec" Discount**: Vehicles imported from non-GCC markets (e.g., US, Japan) trade at a 15% to 30% discount compared to official "GCC Spec" models because they lack desert-ready cooling systems, heavy-duty air conditioning, and local distributor warranty support.
* **Noisy/Dirty Trim Classifications**: Trim data in regional sales logs is messy. Variations like `DLX`, `Deluxe`, `Super GL`, and `S-GL` refer to identical specifications but break exact-string matches, causing automated systems to misprice vehicles by thousands of dirhams.
* **Dynamic Depreciation vs. Static Transactions**: Historical sales databases are sparse. If a specific 2018 model trim hasn't sold in the last 6 months, a database lookup yields nothing. Conversely, relying purely on a machine learning model might miss the sharp depreciation curves unique to luxury German brands compared to Japanese brands.

---

### The VIN Sphere Solution

**VIN Sphere** is a self-contained, local intelligence engine designed to solve these challenges directly on-premise:
* **Hierarchical Router**: Automatically detects the length of the input chassis key (from 9-character JDM imports to 17-character ISO VINs) and routes it to the corresponding custom spec extraction model.
* **Ensemble Spec Extraction**: Uses prefix statistics, suffix pattern matching, and Jaro-Winkler string similarity to clean and map dirty trim labels to standard canonical trims.
* **Hybrid Pricing Engine**: Integrates a direct historical transaction database lookup, a robust stacked machine learning regressor (CatBoost, LightGBM, XGBoost) to fill data gaps, and scraped DriveArabia depreciation indices for a reliable, multi-modal market valuation dashboard.

## Features

- **Multi-Length Chassis Decoding**: Automatically routes and decodes 9, 10, 11, 12, 13, and 17-character chassis numbers/VINs using custom hierarchical models.
- **Hybrid Pricing Engine**:
  - **Market Lookup**: Retrieves historical retail/trade pricing statistics from a transaction database.
  - **Ensemble Machine Learning**: Predicts prices using stacked regressors (CatBoost, LightGBM, XGBoost) and estimates data coverage confidence.
  - **Scraped Depreciation**: Computes current depreciated values utilizing DriveArabia scraped trends.
- **Interactive UI**: Dashboard built with Streamlit for decoding queries and side-by-side market valuation.

## Directory Structure

- `main_app/`: Core Streamlit dashboard and styling.
- `chat_cat_short_vin*/`: Dedicated feature extraction, prefix analysis, and modeling modules for shorter chassis lengths (9, 10, 11, 12, 13 chars).
- `chat_cat(3.0)/`: 17-character standard VIN decoder.
- `Valuation_ml/`: Production price prediction ensemble models.
- `Data/`: Datasets for chassis specs, price lookups, and depreciation indexes.

## Getting Started

1. Set up a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the Streamlit application:
   ```bash
   python -m streamlit run main_app/app.py
   ```