# 🌫️ Beijing Air Quality Analysis & PM2.5 Prediction

> **CMP7005 – Programming for Data Analysis**  
> Cardiff Metropolitan University | 2025–2026  
> Practical Assessment 1

---

## 📌 Project Overview

Air pollution remains one of the most pressing public health challenges in rapidly urbanising cities. This project focuses on **Beijing, China** — a city notorious for its severe particulate matter pollution — and applies a full data science pipeline to four years of hourly air quality monitoring data.

The central objective is to **understand, visualise, and predict PM2.5 concentrations** (fine particulate matter with a diameter of less than 2.5 micrometres) using meteorological and pollutant variables collected between **March 2013 and February 2017**.

The project is structured across four tasks:

| Task | Description |
|------|-------------|
| **Task 1** | Data loading, merging, cleaning, and preprocessing |
| **Task 2** | Statistical analysis and visualisation |
| **Task 3** | Machine learning model building and evaluation |
| **Task 4** | Interactive Streamlit web application |

---

## 📡 Dataset

The dataset is sourced from the **UCI Machine Learning Repository** and comprises hourly air quality readings from **four monitoring stations** in Beijing:

| Station | Location Type | Characteristics |
|---------|--------------|-----------------|
| **Dongsi** | Inner (Urban) | Central Beijing; high traffic density |
| **Tiantan** | Inner (Urban) | Near Temple of Heaven; dense residential area |
| **Shunyi** | Outer (Suburban) | North-east suburb; mixed industrial/residential |
| **Huairou** | Outer (Suburban) | North-west rural fringe; low emission baseline |

**Dataset Statistics:**
- 📊 **140,256** total hourly records
- 🏭 **4** monitoring stations (35,064 rows each)
- 📅 **Period:** March 2013 – February 2017
- 📋 **18** original features per record

**Variables Recorded:**

| Type | Variables |
|------|-----------|
| Pollutants | PM2.5, PM10, SO2, NO2, CO, O3 |
| Meteorological | TEMP, PRES, DEWP, RAIN, WSPM, WD |
| Temporal | year, month, day, hour |

---

## 🔬 Methodology

### Data Preprocessing
- Merged four station CSV files into a single unified dataset
- Built a `datetime` index from year/month/day/hour columns
- Removed duplicate records (none found — data integrity confirmed)
- Applied **per-station forward-fill then back-fill** imputation to resolve missing values across all numeric columns
- Engineered new features: `season`, `AQI_Category`, `station_type`

### Feature Engineering
- **Lag features:** PM2.5 values at 1h, 3h, 6h, 12h, and 24h prior — capturing temporal autocorrelation
- **Rolling mean features:** 3h, 6h, 12h, and 24h rolling averages — smoothing short-term noise and capturing pollution trends

### Machine Learning
- **Baseline model:** Linear Regression
- **Primary model:** Random Forest Regressor
- **Optimisation:** GridSearchCV with 3-fold cross-validation
- **Split strategy:** Chronological 80/20 train-test split to prevent data leakage

---

## 📈 Model Results

| Model | MAE (µg/m³) | RMSE (µg/m³) | R² |
|-------|------------|-------------|-----|
| Linear Regression (Baseline) | 8.45 | 15.02 | 0.9669 |
| Random Forest (Default) | 7.32 | 13.66 | 0.9726 |
| **Random Forest (Tuned)** | **6.87** | **13.29** | **0.9741** |

**Best Hyperparameters (GridSearchCV):**
```
max_depth        : 10
min_samples_split: 5
n_estimators     : 200
```

**5-Fold Cross-Validation R²:** 0.9565 (± 0.0175)

The tuned Random Forest explains approximately **97.4%** of the variance in PM2.5 concentrations, with an average prediction error of just **6.87 µg/m³** — a strong result for hourly pollution forecasting.

---

## 🖥️ Streamlit Web Application

An interactive dashboard was built using Streamlit to present all findings in an accessible, non-technical format.

🔗 **Live App:** [https://assessmentrepo-82zbqj6e4fxc66eyjbgrkz.streamlit.app/](https://assessmentrepo-82zbqj6e4fxc66eyjbgrkz.streamlit.app/)

**App Pages:**

| Page | Features |
|------|----------|
| 🏠 **Home** | Project overview and dataset summary metrics |
| 📊 **Dataset Explorer** | Filter by station/year/month, view statistics and missing values |
| 📈 **Visualisation** | Distributions, scatter plots, temporal trends, correlation heatmap |
| 🤖 **Model Outputs** | Performance metrics, actual vs predicted chart, feature importances |

---

## 🗂️ Repository Structure

```
Assessment_Repo/
│
├── app.py                                        # Streamlit web application
├── requirements.txt                              # Python dependencies
│
├── PRSA_Data_Dongsi_20130301-20170228.csv        # Raw station data
├── PRSA_Data_Tiantan_20130301-20170228.csv       # Raw station data
├── PRSA_Data_Shunyi_20130301-20170228.csv        # Raw station data
├── PRSA_Data_Huairou_20130301-20170228.csv       # Raw station data
│
├── model_predictions.csv                         # Tuned RF model predictions
├── feature_importances.csv                       # Feature importance scores
│
├── programmng_for_Data_Analysis_st20350980.ipynb # Main analysis notebook
└── README.md                                     # Project documentation
```

---

## ⚙️ Running Locally

**1. Clone the repository**
```bash
git clone https://github.com/AJasa1983/Assessment_Repo.git
cd Assessment_Repo
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Launch the app**
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## 📦 Dependencies

```
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
plotly
```

---

## 🔑 Key Findings

- **Winter is consistently the worst season** for PM2.5 across all stations, driven by coal-burning heating systems and temperature inversions that trap pollutants near the surface
- **Inner urban stations** (Dongsi, Tiantan) record significantly higher PM2.5 than outer suburban stations (Huairou, Shunyi) at all times of year
- **Low wind speed** is one of the strongest meteorological contributors to pollution accumulation — the majority of hourly readings fall below 3 m/s
- **PM2.5 lag features** (particularly the 1-hour and 3-hour lags) are the most important predictors in the Random Forest model, confirming strong temporal autocorrelation in pollution data
- Over **48% of hourly readings** fall into the *Unhealthy* or worse AQI category, underlining the scale of Beijing's air quality challenge during this period

---

## 📚 References

- Breiman, L. (2001) 'Random forests', *Machine Learning*, 45(1), pp. 5–32.
- Brauer, M. et al. (2021) 'Ambient particulate matter air pollution and its effects on health', *New England Journal of Medicine*, 384(5), pp. 425–434.
- Hyndman, R.J. and Athanasopoulos, G. (2021) *Forecasting: Principles and Practice*. 3rd edn. Melbourne: OTexts.
- US EPA (2024) *Air Quality Index (AQI) Basics*. Available at: https://www.airnow.gov/aqi/aqi-basics
- Zamani Joharestani, M. et al. (2019) 'PM2.5 prediction based on random forest, XGBoost, and deep learning', *Atmosphere*, 10(7), p. 373.
- Zhang, S. et al. (2017) *Beijing Multi-Site Air Quality Dataset*. UCI Machine Learning Repository.

---

## 👤 Author

**Student ID:** st20350980 
**Name:** Iyke O. Ofem 
**Institution:** Cardiff Metropolitan University  
**Module:** CMP7005 – Programming for Data Analysis  
**Academic Year:** 2025–2026
**Assessment ID:** PRAC 1

---

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/fa/Flag_of_the_People%27s_Republic_of_China.svg/320px-Flag_of_the_People%27s_Republic_of_China.svg.png" width="60"/>
  <br>
  <em>Beijing Air Quality Research | Cardiff Metropolitan University</em>
</p>
