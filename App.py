# ============================================================
# CMP7005 – Programming for Data Analysis
# TASK 4: APPLICATION DEVELOPMENT
# Framework : Streamlit  (pip install streamlit)
# Run with  : streamlit run app.py
#
# Place this file in the SAME folder as the 4 raw PRSA CSVs.
# No cleaned_combined_data.csv needed — data is merged and
# cleaned here exactly as done in the Task 1/2 notebook.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Beijing Air Quality Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── EXACT column names as they appear in the raw PRSA CSVs ──
# Pollutants : PM2.5, PM10, SO2, NO2, CO, O3
# Met vars   : TEMP, PRES, DEWP, RAIN, WSPM
# Other      : No, year, month, day, hour, wd, station (added by us)

POLLUTANTS = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
MET_VARS   = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']

# ── RAW CSV FILENAMES ────────────────────────────────────────
RAW_FILES = {
    'Dongsi':  'PRSA_Data_Dongsi_20130301-20170228.csv',
    'Tiantan': 'PRSA_Data_Tiantan_20130301-20170228.csv',
    'Shunyi':  'PRSA_Data_Shunyi_20130301-20170228.csv',
    'Huairou': 'PRSA_Data_Huairou_20130301-20170228.csv',
}

# ── AQI helper ───────────────────────────────────────────────
def aqi_category(pm25):
    if pd.isna(pm25):     return 'Unknown'
    elif pm25 <= 12.0:    return 'Good'
    elif pm25 <= 35.4:    return 'Moderate'
    elif pm25 <= 55.4:    return 'Unhealthy for Sensitive Groups'
    elif pm25 <= 150.4:   return 'Unhealthy'
    elif pm25 <= 250.4:   return 'Very Unhealthy'
    else:                 return 'Hazardous'

# ── LOAD & CLEAN (mirrors the notebook exactly) ──────────────
@st.cache_data
def load_data():
    frames = []
    missing = []

    for station, fname in RAW_FILES.items():
        if os.path.exists(fname):
            tmp = pd.read_csv(fname)
            tmp['station'] = station
            frames.append(tmp)
        else:
            missing.append(fname)

    # ── fallback if no CSVs found ────────────────────────────
    if not frames:
        st.error(
            "❌ **No PRSA CSV files found next to app.py.**\n\n"
            "Make sure these files are in the **same folder** as `app.py`:\n\n"
            + "\n".join(f"- `{f}`" for f in RAW_FILES.values())
        )
        st.stop()

    if missing:
        st.warning(f"⚠️ Missing files (skipped): {', '.join(missing)}")

    df = pd.concat(frames, ignore_index=True)

    # Drop the raw row-number index column
    if 'No' in df.columns:
        df.drop(columns=['No'], inplace=True)

    # Build datetime (same as notebook)
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df = df.sort_values('datetime').reset_index(drop=True)

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Fill missing numeric values per station (forward-fill then back-fill)
    numeric_cols = [c for c in POLLUTANTS + MET_VARS if c in df.columns]
    df[numeric_cols] = (
        df.groupby('station')[numeric_cols]
          .transform(lambda x: x.ffill().bfill())
    )

    # Feature engineering — Season
    season_map = {
        12:'Winter', 1:'Winter',  2:'Winter',
         3:'Spring', 4:'Spring',  5:'Spring',
         6:'Summer', 7:'Summer',  8:'Summer',
         9:'Autumn', 10:'Autumn', 11:'Autumn'
    }
    df['season'] = df['month'].map(season_map)

    # Feature engineering — AQI Category
    if 'PM2.5' in df.columns:
        df['AQI_Category'] = df['PM2.5'].apply(aqi_category)

    # Feature engineering — Station type
    inner = ['Dongsi', 'Tiantan']
    df['station_type'] = df['station'].apply(
        lambda x: 'Inner (Urban)' if x in inner else 'Outer (Suburban)'
    )

    return df


@st.cache_data
def load_predictions():
    if os.path.exists('model_predictions.csv'):
        return pd.read_csv('model_predictions.csv')
    return None


@st.cache_data
def load_feature_importance():
    if os.path.exists('feature_importances.csv'):
        return pd.read_csv('feature_importances.csv')
    return None


# ── LOAD ─────────────────────────────────────────────────────
df          = load_data()
predictions = load_predictions()
feat_imp_df = load_feature_importance()

STATIONS    = sorted(df['station'].unique().tolist())
ALL_NUMERIC = [c for c in df.select_dtypes(include=np.number).columns
               if c not in ['year', 'month', 'day', 'hour', 'No']]

# ── SIDEBAR ──────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fa/"
    "Flag_of_the_People%27s_Republic_of_China.svg/320px-Flag_of_the_People%27s_Republic_of_China.svg.png",
    width=60
)
st.sidebar.title("🌫️ Beijing Air Quality")
st.sidebar.caption("CMP7005 | PRAC1 | Cardiff Met")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to",
    ["🏠 Home", "📊 Dataset Explorer", "📈 Visualisation", "🤖 Model Outputs"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Stations Analysed**")
for s in STATIONS:
    st.sidebar.markdown(f"• {s}")
st.sidebar.markdown("**Period:** Mar 2013 – Feb 2017")


# ════════════════════════════════════════════════════════════
# PAGE 1 – HOME
# ════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🌫️ Beijing Air Quality Analysis Dashboard")
    st.markdown("""
    > **CMP7005 – Programming for Data Analysis | Practical Assessment 1**
    > Cardiff Metropolitan University | 2025–2026
    """)
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📌 Project Overview")
        st.markdown("""
        This interactive application presents the outputs of a data analysis pipeline
        applied to hourly air quality data from **4 monitoring stations in Beijing**,
        spanning **March 2013 to February 2017**.

        The primary pollutant of interest is **PM2.5** — fine particulate matter with
        a diameter of less than 2.5 micrometres — which poses significant risks to
        cardiovascular and respiratory health (Brauer et al., 2021).

        #### Stations Selected
        | Station | Type | Rationale |
        |---------|------|-----------|
        | **Dongsi** | Inner (Urban) | Central Beijing; high traffic density |
        | **Tiantan** | Inner (Urban) | Near Temple of Heaven; urban residential |
        | **Shunyi** | Outer (Suburban) | North-east suburb; mixed industrial/residential |
        | **Huairou** | Outer (Suburban) | North-west rural fringe; low emission baseline |
        """)

    with col2:
        st.subheader("📐 Dataset Summary")
        st.metric("Total Records", f"{len(df):,}")
        st.metric("Features",      f"{df.shape[1]}")
        st.metric("Stations",      f"{df['station'].nunique()}")
        if 'PM2.5' in df.columns:
            st.metric("Avg PM2.5", f"{df['PM2.5'].mean():.1f} µg/m³")
            st.metric("Max PM2.5", f"{df['PM2.5'].max():.1f} µg/m³")

    st.markdown("---")
    st.subheader("🗺️ How to Use This App")
    st.markdown("""
    Use the **sidebar** on the left to navigate between sections:

    | Page | Description |
    |------|-------------|
    | 📊 **Dataset Explorer** | Browse raw data, filter by station/date, view statistics |
    | 📈 **Visualisation** | Explore univariate, bivariate, and temporal charts |
    | 🤖 **Model Outputs** | Review ML model performance and PM2.5 predictions |
    """)


# ════════════════════════════════════════════════════════════
# PAGE 2 – DATASET EXPLORER
# ════════════════════════════════════════════════════════════
elif page == "📊 Dataset Explorer":
    st.title("📊 Dataset Explorer")
    st.markdown("Browse, filter, and summarise the cleaned air quality dataset.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        sel_stations = st.multiselect("Filter by Station", STATIONS, default=STATIONS)
    with col2:
        years = sorted(df['year'].unique())
        sel_years = st.multiselect("Filter by Year", years, default=years)
    with col3:
        months = sorted(df['month'].unique())
        sel_months = st.multiselect("Filter by Month", months, default=months)

    dff = df[df['station'].isin(sel_stations)]
    dff = dff[dff['year'].isin(sel_years)]
    dff = dff[dff['month'].isin(sel_months)]

    if dff.empty:
        st.warning("No data matches the current filters.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",           f"{len(dff):,}")
    c2.metric("Columns",        f"{dff.shape[1]}")
    c3.metric("Missing Values", f"{dff.isnull().sum().sum():,}")
    if 'PM2.5' in dff.columns:
        c4.metric("Mean PM2.5", f"{dff['PM2.5'].mean():.1f} µg/m³")

    st.subheader("🗃️ Data Sample")
    n_rows = st.slider("Rows to display", 5, 100, 20)
    st.dataframe(dff.head(n_rows), use_container_width=True)

    st.subheader("📐 Statistical Summary")
    default_cols = [c for c in ['PM2.5', 'PM10', 'NO2', 'TEMP', 'WSPM'] if c in ALL_NUMERIC]
    cols_desc = st.multiselect("Select columns", ALL_NUMERIC, default=default_cols)
    if cols_desc:
        st.dataframe(dff[cols_desc].describe().round(2), use_container_width=True)

    st.subheader("🔍 Missing Value Analysis")
    missing = dff.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        fig, ax = plt.subplots(figsize=(10, 3))
        missing.plot(kind='bar', ax=ax, color='tomato', edgecolor='white')
        ax.set_title('Missing Values per Column')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.success("✅ No missing values in the filtered dataset.")

    with st.expander("📋 Column Data Types"):
        st.dataframe(pd.DataFrame({
            'Column':         dff.dtypes.index,
            'Data Type':      dff.dtypes.values.astype(str),
            'Non-Null Count': dff.count().values
        }), use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 3 – VISUALISATION
# ════════════════════════════════════════════════════════════
elif page == "📈 Visualisation":
    st.title("📈 Visualisation")
    st.markdown("Explore pollutant distributions, relationships, and temporal trends.")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📦 Distributions", "🔗 Relationships", "🕐 Temporal Trends", "🔥 Correlation"
    ])

    # ── TAB 1: Distributions ────────────────────────────────
    with tab1:
        st.subheader("Univariate Distributions")
        c1, c2 = st.columns(2)
        with c1:
            idx = ALL_NUMERIC.index('PM2.5') if 'PM2.5' in ALL_NUMERIC else 0
            var = st.selectbox("Variable", ALL_NUMERIC, index=idx)
        with c2:
            sf = st.multiselect("Station(s)", STATIONS, default=STATIONS, key='dist_s')

        sub = df[df['station'].isin(sf)] if sf else df

        if sub.empty:
            st.warning("No data for selected filters.")
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].hist(sub[var].dropna(), bins=50, color='steelblue',
                         edgecolor='white', alpha=0.85)
            axes[0].set_title(f'Distribution of {var}')
            axes[0].set_xlabel(var)
            axes[0].set_ylabel('Frequency')

            pairs = [(s, sub[sub['station'] == s][var].dropna().values)
                     for s in sf if len(sub[sub['station'] == s][var].dropna()) > 0]
            if pairs:
                lbls, arrs = zip(*pairs)
                axes[1].boxplot(arrs, labels=lbls, patch_artist=True,
                                boxprops=dict(facecolor='lightblue'))
            axes[1].set_title(f'{var} by Station')
            axes[1].set_ylabel(var)
            plt.xticks(rotation=15)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.caption("Histogram = frequency distribution. Boxplot = median, IQR and outliers per station.")

    # ── TAB 2: Relationships ─────────────────────────────────
    with tab2:
        st.subheader("Bivariate Relationship")
        c1, c2, c3 = st.columns(3)
        with c1:
            xi = ALL_NUMERIC.index('TEMP') if 'TEMP' in ALL_NUMERIC else 0
            xv = st.selectbox("X axis", ALL_NUMERIC, index=xi)
        with c2:
            yi = ALL_NUMERIC.index('PM2.5') if 'PM2.5' in ALL_NUMERIC else min(1, len(ALL_NUMERIC)-1)
            yv = st.selectbox("Y axis", ALL_NUMERIC, index=yi)
        with c3:
            col_by = st.selectbox("Colour by", ['None'] + STATIONS)

        samp = df[[xv, yv, 'station']].dropna().sample(min(5000, len(df)), random_state=42)

        fig, ax = plt.subplots(figsize=(9, 5))
        if col_by == 'None':
            ax.scatter(samp[xv], samp[yv], alpha=0.3, s=8, color='steelblue')
        else:
            for s in STATIONS:
                sub = samp[samp['station'] == s]
                ax.scatter(sub[xv], sub[yv], alpha=0.4, s=8, label=s)
            ax.legend(markerscale=3)

        corr = samp[[xv, yv]].corr().iloc[0, 1]
        strength  = 'strong' if abs(corr) > 0.6 else 'moderate' if abs(corr) > 0.3 else 'weak'
        direction = 'positive' if corr > 0 else 'negative'
        ax.set_xlabel(xv); ax.set_ylabel(yv)
        ax.set_title(f'{yv} vs {xv}   (Pearson r = {corr:.3f})')
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.info(f"**Pearson r = {corr:.3f}** — {strength} {direction} linear relationship.")

    # ── TAB 3: Temporal Trends ───────────────────────────────
    with tab3:
        st.subheader("Temporal Trend Analysis")
        c1, c2 = st.columns(2)
        with c1:
            poll_opts = [c for c in POLLUTANTS if c in df.columns]
            tv = st.selectbox("Variable", poll_opts, index=0)
        with c2:
            agg = st.selectbox("Aggregation", ['Monthly', 'Daily', 'Hourly (avg by hour)'])

        tst = st.multiselect("Stations", STATIONS, default=STATIONS, key='trend_s')
        dft = df[df['station'].isin(tst)] if tst else df

        fig, ax = plt.subplots(figsize=(13, 4))
        plotted = False
        for s in tst:
            sub = dft[dft['station'] == s].copy()
            if sub.empty or tv not in sub.columns:
                continue
            try:
                if agg == 'Monthly':
                    grp = sub.groupby(['year','month'])[tv].mean().reset_index()
                    grp['period'] = pd.to_datetime(grp[['year','month']].assign(day=1))
                    ax.plot(grp['period'], grp[tv], label=s, linewidth=1.5)
                    plotted = True
                elif agg == 'Daily':
                    grp = sub.resample('D', on='datetime')[tv].mean()
                    ax.plot(grp.index, grp.values, label=s, linewidth=0.8, alpha=0.8)
                    plotted = True
                elif agg == 'Hourly (avg by hour)':
                    grp = sub.groupby('hour')[tv].mean()
                    ax.plot(grp.index, grp.values, marker='o', label=s, linewidth=2)
                    plotted = True
            except Exception as e:
                st.warning(f"Could not plot {s}: {e}")

        if plotted:
            ax.set_xlabel('Time' if agg != 'Hourly (avg by hour)' else 'Hour of Day')
            ax.set_ylabel(f'Mean {tv}')
            ax.set_title(f'{agg} Average {tv} by Station')
            ax.legend(); plt.tight_layout(); st.pyplot(fig)
        else:
            st.warning("No data for selected configuration.")
        plt.close()

    # ── TAB 4: Correlation Heatmap ───────────────────────────
    with tab4:
        st.subheader("Multivariate Correlation Heatmap")
        default_hm = [c for c in POLLUTANTS + ['TEMP', 'WSPM', 'PRES'] if c in ALL_NUMERIC]
        hm_cols = st.multiselect("Variables", ALL_NUMERIC, default=default_hm)
        if len(hm_cols) >= 2:
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(df[hm_cols].corr(), annot=True, fmt='.2f', cmap='RdYlGn',
                        center=0, ax=ax, linewidths=0.5, square=True,
                        annot_kws={"size": 9})
            ax.set_title('Pearson Correlation Matrix')
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.caption("+1 = strong positive correlation, −1 = strong negative correlation.")
        else:
            st.warning("Select at least 2 variables.")


# ════════════════════════════════════════════════════════════
# PAGE 4 – MODEL OUTPUTS
# ════════════════════════════════════════════════════════════
elif page == "🤖 Model Outputs":
    st.title("🤖 Model Outputs — PM2.5 Prediction")
    st.markdown("Results from the **Random Forest Regression** model trained in Task 3.")
    st.markdown("---")

    st.subheader("📐 Model Performance Metrics")
    st.markdown("""
    Evaluated on a **chronological 20% hold-out test set** — a time-based split
    prevents future data leaking into the training period (Hyndman & Athanasopoulos, 2021).
    """)

    if predictions is not None:
        ya = predictions['Actual_PM25']
        yp = predictions['Predicted_PM25']

        mae  = np.mean(np.abs(ya - yp))
        rmse = np.sqrt(np.mean((ya - yp) ** 2))
        r2   = 1 - (np.sum((ya - yp)**2) / np.sum((ya - ya.mean())**2))
        mape = np.mean(np.abs((ya - yp) / ya.replace(0, np.nan))) * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE (µg/m³)",  f"{mae:.2f}")
        c2.metric("RMSE (µg/m³)", f"{rmse:.2f}")
        c3.metric("R² Score",     f"{r2:.4f}")
        c4.metric("MAPE (%)",     f"{mape:.1f}")

        st.markdown("---")
        st.subheader("📊 Actual vs Predicted PM2.5")
        sp = predictions.sample(min(3000, len(predictions)), random_state=42)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].scatter(sp['Actual_PM25'], sp['Predicted_PM25'], alpha=0.3, s=8, color='steelblue')
        mv = max(sp['Actual_PM25'].max(), sp['Predicted_PM25'].max())
        axes[0].plot([0, mv], [0, mv], 'r--', lw=2, label='y = x (perfect)')
        axes[0].set_xlabel('Actual PM2.5'); axes[0].set_ylabel('Predicted PM2.5')
        axes[0].set_title(f'Actual vs Predicted  (R² = {r2:.4f})'); axes[0].legend()

        res = sp['Actual_PM25'] - sp['Predicted_PM25']
        axes[1].hist(res, bins=60, color='coral', edgecolor='white', alpha=0.85)
        axes[1].axvline(0, color='black', linestyle='--', lw=1.5)
        axes[1].set_xlabel('Residual'); axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residual Distribution')
        plt.tight_layout(); st.pyplot(fig); plt.close()

        if feat_imp_df is not None:
            st.subheader("🏆 Feature Importance (Top 20)")
            top20 = feat_imp_df.head(20).sort_values('Importance')
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.barh(top20['Feature'], top20['Importance'], color='teal', edgecolor='white')
            ax.set_xlabel('Importance Score')
            ax.set_title('Random Forest – Top 20 Feature Importances')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.subheader("📋 Sample Predictions")
        n = st.slider("Rows", 10, 200, 30)
        dcols = [c for c in ['Actual_PM25','Predicted_PM25','Residual'] if c in predictions.columns]
        st.dataframe(predictions[dcols].head(n).round(2), use_container_width=True)
        st.download_button("⬇️ Download Predictions CSV",
                           predictions[dcols].to_csv(index=False).encode(),
                           'pm25_predictions.csv', 'text/csv')
    else:
        st.warning("⚠️ `model_predictions.csv` not found. Run Task 3 notebook to generate it.")

    with st.expander("📖 Modelling Decisions & Justification"):
        st.markdown("""
        ### Why Random Forest?
        - Handles **non-linear relationships** between meteorological variables and PM2.5.
        - **Robust to outliers** — important given extreme pollution events in Beijing.
        - Provides **feature importance scores** to aid interpretability.
        - Strong empirical performance in air quality prediction literature (Zamani Joharestani et al., 2019).

        ### Why a Chronological Train/Test Split?
        Preserves temporal order and prevents data leakage — the model never sees future data during training.

        ### References
        - Breiman, L. (2001) *Random forests*. Machine Learning, 45(1), pp.5–32.
        - Hyndman, R.J. and Athanasopoulos, G. (2021) *Forecasting: Principles and Practice*. 3rd edn.
        - Zamani Joharestani, M. et al. (2019) *PM2.5 prediction based on Random Forest*, Atmosphere, 10(7).
        """)


# ── FOOTER ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:grey; font-size:0.8em;'>"
    "CMP7005 Programming for Data Analysis | Cardiff Metropolitan University | 2025–2026"
    "</p>",
    unsafe_allow_html=True
)

