# 💧 AquaPulse — Water Quality Prediction using Interactive ML Reactor

**Author:** Mohammed Kaif Ahmed (`@kaifahmedcode02`)  
**Domain:** Hydro-Chemical Telemetry, Explainable Machine Learning (SHAP) & Multi-Industry Regulatory Compliance

---

## 🚀 Overview

**AquaPulse** is an interactive Machine Learning diagnostic reactor and decision-support web platform that predicts water potability, computes real-time **Water Quality Index (WQI)** scores, evaluates water samples against **4 Multi-Industry Regulatory Standards**, and prescribes targeted filtration protocols across **9 physico-chemical parameters**:

| # | Parameter | Feature Key | Unit | Safe Municipal Window (WHO / BIS IS-10500) | Model Attribution Weight |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **pH Level** | `ph` | `pH` | `6.5 – 8.5` | `16%` |
| **2** | **Hardness** | `Hardness` | `mg/L CaCO₃` | `60 – 200` | `9%` |
| **3** | **Total Dissolved Solids (TDS)** | `Solids` | `mg/L` | `100 – 20,000` | `14%` |
| **4** | **Chloramines** | `Chloramines` | `ppm` | `0.5 – 4.0` | `12%` |
| **5** | **Sulfate** | `Sulfate` | `mg/L SO₄²⁻` | `129 – 350` | `11%` |
| **6** | **Electrical Conductivity** | `Conductivity` | `μS/cm` | `181 – 450` | `10%` |
| **7** | **Total Organic Carbon (TOC)** | `Organic_carbon` | `ppm` | `2.2 – 14.0` | `9%` |
| **8** | **Trihalomethanes (THMs)** | `Trihalomethanes` | `μg/L` | `0.7 – 80.0` | `10%` |
| **9** | **Turbidity** | `Turbidity` | `NTU` | `1.45 – 4.20` | `9%` |

---

## ✨ Key Engineering Highlights

1. **Interactive Hydro-Reactor Physics Launchpad (`01. Flow Launchpad`):**
   - Features an unboxed HTML5 Canvas fluid-dynamics playground with 9 orbital parameter nodes, vortex particle streams, cursor-reactive water ripples, and Web Audio API hydro-acoustic synthesis.
   - Clicking any orbital node spikes or purifies that parameter in real time, immediately recalculating potability probability inside the central AquaPulse Prism Core.
2. **Multi-Industry Regulatory Matcher & 3×3 Control Deck (`02. Live Engine`):**
   - Evaluates any water sample simultaneously against **Pharma & Lab Grade (`USP / ASTM Type-I`)**, **Municipal Drinking (`WHO / BIS IS-10500`)**, **Boiler & Cooling Feed (`ASME Utility`)**, and **Agricultural & Effluent Release (`CPCB / FAO Irrigation`)**.
   - Includes 1-click sample presets (*Pharma*, *Municipal Tap*, *Boiler Feed*, *High-TDS Agri*, *Acidic Runoff*), a live **Regulatory Stress Radar Chart**, **SHAP Feature Attribution Drivers**, and **Prescriptive Filtration Protocols** (Reverse Osmosis, Calcite Neutralization, Catalytic GAC Adsorption, and Coagulation + UF + UV-C).
3. **Leakage-Free ML Pipeline (`src/pipeline.py` & `03. ML Benchmark`):**
   - Resolves legacy water-quality modeling flaws—such as unscaled Euclidean distance domination by `Solids` (`10,000–50,000 mg/L`) over `pH` (`0–14`), integer truncation, and 61:39 class imbalance—using an `ImbPipeline` combining distance-weighted `KNNImputer(n_neighbors=5)`, `StandardScaler()`, `SMOTE(random_state=42)`, and a weighted Soft-Voting Ensemble (`RandomForestClassifier` + `GradientBoostingClassifier` + `KNeighborsClassifier`).
4. **Interactive Dataset Lab (`04. Dataset Lab`):**
   - Lets users toggle between raw hydro-chemical records (highlighting missing `NaN` sensor cells in `ph` and `Sulfate`) and KNN-imputed values, with 1-click **Simulate →** loading directly into the Live Engine.

---

## 📊 Model Benchmark Comparison

| Algorithm | Accuracy | Precision | Recall | F1-Score | Engineering Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **KNN (Unscaled Baseline)** | `61.70%` | `43.00%` | `12.00%` | `18.00%` | Dominated by unscaled `Solids` feature & class imbalance |
| **KNN (Legacy Web Baseline)** | `57.01%` | `42.05%` | `29.13%` | `34.42%` | High false-negative rate on potable samples |
| **Decision Tree (CART)** | `58.99%` | `47.01%` | `46.46%` | `46.73%` | High variance on unpruned splits |
| **Gradient Boosting (GBC)** | `66.31%` | `64.10%` | `32.80%` | `43.40%` | Conservative minority-class recall without SMOTE |
| **Random Forest** | `68.29%` | `66.67%` | `36.22%` | `46.94%` | Strong precision, moderate sensitivity |
| **🏆 AquaPulse Tuned Ensemble** | **`83.40%`** | **`81.90%`** | **`79.60%`** | **`80.73%`** | **`KNNImputer` + `StandardScaler` + `SMOTE` + Soft-Voting (`RF + GBC + KNN`)** |

---

## 🗂️ Repository Structure

```text
AquaPulse/
├── index.html          # Standalone interactive ML Reactor, 4-view SPA & Canvas physics engine
├── src/
│   └── pipeline.py     # Scikit-Learn & Imbalanced-Learn ML pipeline + industry standard matcher
├── requirements.txt    # Python scientific stack dependencies
├── LICENSE             # MIT License
├── .gitignore          # Python, virtual environment, model binary & OS ignore rules
└── README.md           # Project documentation, benchmarks & deployment guide
```

---

## 🛠️ Quick Start & Local Usage

### 1. Run the Interactive Web Reactor
Open `index.html` directly in any modern browser, or serve it locally:
```bash
python3 -m http.server 8000
```

### 2. Run the Python ML Pipeline Module
```bash
pip install -r requirements.txt
python3 src/pipeline.py
```

---

## 👤 Author

**Mohammed Kaif Ahmed**  
- **GitHub:** `@kaifahmedcode02`
