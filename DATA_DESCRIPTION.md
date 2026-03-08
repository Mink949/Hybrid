# Weather-Mortality Data Description
## Environmental Epidemiology Research Dataset

**Generated:** March 5, 2026  
**Processing Script:** `preprocess_weather_mortality_data.py`  
**Final Dataset:** `weather_mortality_processed.csv`

---

## 📋 Dataset Overview

### Summary Statistics
- **Total Records:** 1,566 weekly observations
- **Time Period:** January 1, 2015 - December 23, 2024 (10 years)
- **Geographic Coverage:** 3 Australian states (NSW, VIC, QLD)
- **Total Features:** 60 variables (original + engineered + scaled)
- **Records per State:** 522 weeks each

### Key Metrics by State
| State | Population | Mean Deaths/Week | Mean Temp (°C) | P95 Heat Threshold (°C) |
|-------|-----------|------------------|----------------|------------------------|
| NSW   | 8,000,000 | 1,064.5         | 18.8          | 30.50                 |
| VIC   | 6,500,000 | 810.6           | 18.8          | 33.00                 |
| QLD   | 5,200,000 | 644.9           | 18.8          | 32.90                 |

---

## 🔄 Data Merging Process

### Step 1: Source Data Collection

#### Input Files Structure
```
merge/
├── Weather Data (Daily, 3 files)
│   ├── new_south_wales_weather.csv   (3,653 days)
│   ├── victoria_weather.csv          (3,653 days)
│   └── queensland_weather.csv        (3,653 days)
│
└── Death Data (Weekly, 3 files)
    ├── deaths_new_south_wales_weekly.csv  (522 weeks)
    ├── deaths_victoria_weekly.csv         (522 weeks)
    └── deaths_queensland_weekly.csv       (522 weeks)
```

### Step 2: Data Cleaning & Transformation

#### 2.1 Weather Data Processing
**Original Columns:**
- `Date` (YYYYMMDD format) → converted to datetime
- `Max_Temperature_C` (daily maximum temperature)
- `Min_Temperature_C` (daily minimum temperature)
- `Relative_Humidity_Max_Temp_Percent`
- `Relative_Humidity_Min_Temp_Percent`
- `Rainfall_mm` (daily rainfall)
- `Solar_Radiation_MJ_m2`

**Processing Steps:**
1. **Date Conversion:** YYYYMMDD string → datetime object
2. **State Identification:** Added `State` column (NSW/VIC/QLD)
3. **Missing Value Imputation:**
   - Method: Linear interpolation (time-series appropriate)
   - Fallback: Forward/backward fill for edge cases
   - Result: 0 missing values after imputation

#### 2.2 Death Data Processing
**Original Columns:**
- `State` (state identifier)
- `Year` (calendar year)
- `Week` (ISO week number)
- `Deaths` (weekly death count)

**Processing Steps:**
1. **Date Construction:** Year + Week → `Date` (first day of ISO week)
2. **State Normalization:** Standardized state codes
3. **No imputation needed:** Complete dataset

### Step 3: Merge Strategy - Long Format

#### 3.1 Vertical Concatenation (Long Format)
All three states merged into single DataFrame:

```python
# Weather: 3,653 days × 3 states = 10,959 daily records
weather_combined = pd.concat([nsw_weather, vic_weather, qld_weather])

# Deaths: 522 weeks × 3 states = 1,566 weekly records  
deaths_combined = pd.concat([nsw_deaths, vic_deaths, qld_deaths])
```

**Why Long Format?**
- Enables state-specific analysis with `groupby()`
- Supports mixed-effects models (state as random effect)
- Efficient for time-series operations per state
- Scales well for adding more states

#### 3.2 Weekly Aggregation of Weather Data

**Temporal Alignment Challenge:**
- Weather data: **Daily** granularity
- Death data: **Weekly** granularity
- Solution: Aggregate weather from daily → weekly

**Aggregation Functions (Preserving Extreme Characteristics):**

| Original Variable | Aggregation | New Variable(s) | Rationale |
|-------------------|-------------|-----------------|-----------|
| Max_Temperature_C | mean, max, min, std | Mean_Max_Temp, Max_Temp, Min_Max_Temp, SD_Temp | Capture both average exposure and extreme events |
| Min_Temperature_C | mean, min | Mean_Min_Temp, Min_Temp | Identify cold extremes |
| Max_Temperature_C | count > P95 | Heat_Days_Count | Extreme heat event frequency |
| Humidity | mean | Mean_Humidity_Max, Mean_Humidity_Min | Average weekly humidity |
| Rainfall_mm | sum | Total_Rainfall | Cumulative weekly precipitation |
| Solar_Radiation | mean | Mean_Solar_Radiation | Average weekly solar exposure |

**Derived Variable:**
```python
Mean_Temp = (Mean_Max_Temp + Mean_Min_Temp) / 2
```

#### 3.3 Final Merge - Weather + Deaths

**Join Key:** `[State, Year, Week]`  
**Join Type:** Inner join (keep only matching records)

```python
merged = pd.merge(
    weekly_weather,      # 1,569 records
    deaths,              # 1,566 records  
    on=['State', 'Year', 'Week'],
    how='inner'          # Result: 1,566 records
)
```

**Added Variables:**
- `Population`: State population estimates
- `Death_Rate_Per_100k`: (Deaths / Population) × 100,000
- `Log_Population`: ln(Population) - GAM offset variable

---

## 🔧 Feature Engineering

### 1. Lag Features (Temporal Effects)

**Purpose:** Capture delayed health impacts of weather events

**Implementation:**
- **Lag 1 Week:** Acute/immediate effects
- **Lag 2 Weeks:** Sub-acute/delayed effects

**Variables with Lags:**
- `Mean_Temp_Lag1`, `Mean_Temp_Lag2`
- `Max_Temp_Lag1`, `Max_Temp_Lag2`
- `Min_Temp_Lag1`, `Min_Temp_Lag2`
- `SD_Temp_Lag1`, `SD_Temp_Lag2`
- `Heat_Days_Count_Lag1`, `Heat_Days_Count_Lag2`
- `Mean_Humidity_Max_Lag1`, `Mean_Humidity_Max_Lag2`
- `Total_Rainfall_Lag1`, `Total_Rainfall_Lag2`

**Total Lag Features:** 14 (7 variables × 2 lag periods)

**Note:** First 2 weeks per state have NA in lag features (expected behavior)

### 2. Extreme Heat Definition

**State-Specific Thresholds (95th Percentile):**
- Accounts for climate acclimatization differences
- NSW: 30.50°C (cooler baseline)
- VIC: 33.00°C (moderate baseline)
- QLD: 32.90°C (warmer baseline)

**Heat_Days_Count Calculation:**
```python
# For each day in week:
if Max_Temperature > P95_threshold[state]:
    heat_day = 1
    
# Sum heat days in week (0-7 range)
Heat_Days_Count = sum(heat_days)
```

### 3. GAM Offset Variable

**Purpose:** Adjust for population size in Poisson/Negative Binomial models

```python
Log_Population = ln(Population)
```

**Model Usage:**
```R
# GAM formula with offset
Deaths ~ s(Mean_Temp) + s(Heat_Days_Count) + offset(Log_Population)
```

This models **mortality rate** rather than raw counts, accounting for different state populations.

### 4. Standardization (Z-Score Scaling)

**Variables Scaled:** 26 continuous variables (original + lags)

**Formula:**
```python
X_scaled = (X - mean(X)) / std(X)
```

**Standardized Variables** (suffix `_Scaled`):
- All temperature metrics
- Heat_Days_Count
- Humidity measures
- Rainfall totals
- Solar radiation
- Death_Rate_Per_100k
- All lag features

**Purpose:**
- Model convergence in GAM/ML algorithms
- Comparable effect sizes across predictors
- Improved numerical stability

**Scaler Storage:** `scaler_dict.pkl` (for inverse transformation)

---

## 📊 Final Dataset Structure

### Column Categories

#### **Identifiers (5 columns)**
- `State` - NSW, VIC, QLD
- `Year` - 2015-2024
- `Week` - ISO week number (1-53)
- `Week_Start_Date` - First day of week (datetime)
- `Population` - State population estimate

#### **Outcome Variables (2 columns)**
- `Deaths` - Weekly death count
- `Death_Rate_Per_100k` - Deaths per 100,000 population

#### **Temperature Features (11 columns)**
- `Mean_Temp` - Average temperature for the week
- `Max_Temp` - Highest temperature in the week
- `Min_Temp` - Lowest temperature in the week
- `SD_Temp` - Temperature variability (standard deviation)
- `Mean_Max_Temp` - Average of daily maximums
- `Mean_Min_Temp` - Average of daily minimums
- `Min_Max_Temp` - Minimum of daily maximums
- `Heat_Days_Count` - Days exceeding P95 threshold
- + 3 scaled versions (Mean_Temp_Scaled, Max_Temp_Scaled, Min_Temp_Scaled)

#### **Other Weather Features (6 columns)**
- `Mean_Humidity_Max` - Average humidity at max temp
- `Mean_Humidity_Min` - Average humidity at min temp
- `Total_Rainfall` - Cumulative weekly rainfall (mm)
- `Mean_Solar_Radiation` - Average solar radiation (MJ/m²)
- + 2 scaled versions

#### **Lag Features (14 columns)**
- 7 weather variables × 2 lag periods (Lag1, Lag2)

#### **Scaled Features (26 columns)**
- All continuous variables with `_Scaled` suffix

#### **Model Variables (2 columns)**
- `Log_Population` - ln(Population) for GAM offset
- `Death_Rate_Per_100k_Scaled` - Scaled mortality rate

**Total:** 60 columns

---

## 🔍 Data Quality Checks

### Automated Validation (All Passed ✓)

| Check | Status | Details |
|-------|--------|---------|
| Duplicate Records | ✓ Pass | No duplicate (State, Year, Week) combinations |
| Date Range | ✓ Pass | 3,644 days (10.0 years) of coverage |
| Death Counts | ✓ Pass | All values > 0 |
| Temperature Ranges | ✓ Pass | Max: -10 to 50°C, Min: -20 to 40°C |
| Max ≥ Min Temp | ✓ Pass | Logical consistency maintained |
| Heat Days Count | ✓ Pass | All values ≤ 7 (within week) |
| Missing Values | ✓ Pass | No NaN in critical columns |

### Known Data Characteristics

**Missing Values by Design:**
- Lag features: First 2 weeks per state (6 records) = 63 total NA values
- These are **expected** and handled appropriately in modeling

**Data Distribution:**
- Deaths: Right-skewed (use Poisson/Negative Binomial distribution)
- Temperature: Approximately normal (seasonal patterns)
- Rainfall: Highly right-skewed (many weeks with little rain)

---

## 📖 Usage Guide

### Loading the Dataset

```python
import pandas as pd
import pickle

# Load processed data
df = pd.read_csv('weather_mortality_processed.csv')

# Load scalers for inverse transformation
with open('scaler_dict.pkl', 'rb') as f:
    scalers = pickle.load(f)

# Convert date column to datetime
df['Week_Start_Date'] = pd.to_datetime(df['Week_Start_Date'])
```

### Inverse Scaling (For Interpretation)

```python
# Example: Convert scaled temperature back to original scale
original_temp = scalers['Mean_Temp'].inverse_transform(
    df[['Mean_Temp_Scaled']]
)

# Access scaler parameters
print(f"Mean: {scalers['Mean_Temp'].mean_[0]:.2f}")
print(f"Std Dev: {scalers['Mean_Temp'].scale_[0]:.2f}")
```

### Filtering by State

```python
# Analyze single state
nsw_data = df[df['State'] == 'NSW']

# Compare states
state_comparison = df.groupby('State').agg({
    'Deaths': 'mean',
    'Mean_Temp': 'mean',
    'Heat_Days_Count': 'sum'
})
```

### Time Series Subsetting

```python
# Filter by year
recent_data = df[df['Year'] >= 2020]

# Filter by date range
summer_2020 = df[
    (df['Week_Start_Date'] >= '2020-12-01') &
    (df['Week_Start_Date'] <= '2021-03-01')
]
```

### Handling Lag Features

```python
# Remove rows with NA lags (if needed)
complete_cases = df.dropna(subset=[col for col in df.columns if 'Lag' in col])

# Or keep all rows (model will handle NAs)
# Most GAM/GLM implementations handle missing predictors
```

---

## 📈 Recommended Analyses

### 1. Generalized Additive Model (GAM)

```python
from pygam import PoissonGAM, s

# Use non-scaled features + offset
gam = PoissonGAM(
    s(0) +           # Mean_Temp (smooth term)
    s(1) +           # Heat_Days_Count  
    s(2)             # Mean_Humidity_Max
)

X = df[['Mean_Temp', 'Heat_Days_Count', 'Mean_Humidity_Max']]
y = df['Deaths']
offset = df['Log_Population']

gam.fit(X, y, offset=offset)
```

### 2. State-Specific Analysis

```python
# Separate models per state
for state in ['NSW', 'VIC', 'QLD']:
    state_df = df[df['State'] == state]
    # Fit state-specific model...
```

### 3. Distributed Lag Models

```python
# Include multiple lags to capture cumulative effects
X_lags = df[[
    'Mean_Temp',
    'Mean_Temp_Lag1', 
    'Mean_Temp_Lag2'
]]
```

### 4. Seasonal Decomposition

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# For each state
nsw = df[df['State'] == 'NSW'].set_index('Week_Start_Date')
decomp = seasonal_decompose(nsw['Deaths'], model='additive', period=52)
```

---

## ⚠️ Important Notes

### Temporal Dependencies
- Data is **time-ordered** within each state
- For causal inference, consider lagged predictors
- Account for autocorrelation in standard errors

### State Heterogeneity  
- Different climates require state-specific thresholds
- Consider state as fixed/random effect in models
- Population differences handled via offset

### Extreme Events
- `Heat_Days_Count` captures frequency, not intensity
- Consider interaction terms (e.g., Heat_Days × Humidity)
- Plateau effects may exist above certain thresholds

### Confounding Variables (Not Included)
- **Air pollution** (PM2.5, ozone) - major confounder
- **Influenza/COVID-19 seasons** - epidemic confounding
- **Public holidays** - reporting artifacts
- **Socioeconomic factors** - vulnerability modifiers
- **Age structure changes** - demographic shifts

Consider adding these variables for more robust causal inference.

---

## 📚 References

### Data Sources
- **Weather Data:** Australian Bureau of Meteorology stations
  - NSW: Centennial Park (66160)
  - VIC: Melbourne (86071)
  - QLD: Brisbane (40913)

- **Mortality Data:** State health registries (2015-2024)

### Methodological Approach
- **Interpolation:** Preserves temporal continuity
- **Weekly Aggregation:** Balances granularity vs. noise
- **P95 Threshold:** Standard in heat-health literature
- **GAM Framework:** Flexible non-linear relationships

### Processing Artifacts
- **3 records lost** in final merge (1,569 → 1,566)
  - Likely incomplete weeks at dataset boundaries
  - Minimal impact on 10-year analysis

---

## 🔄 Data Versioning

**Version:** 1.0  
**Processing Date:** March 5, 2026  
**Script Version:** `preprocess_weather_mortality_data.py` (initial release)

### Reproducibility
- All processing parameters hard-coded in script
- Scalers saved for exact inverse transformation
- Metadata preserved in `processing_metadata.json`

### Updates Needed If:
- New states added → Update `STATE_POPULATION` dict
- New years added → Re-run full script (P95 thresholds may shift)
- Weather variables added → Update `WEATHER_COLUMNS` list

---

## 📧 Contact & Support

For questions about:
- **Data Processing:** Review `preprocess_weather_mortality_data.py` (extensively commented)
- **Variable Definitions:** See sections above
- **Modeling Recommendations:** See "Recommended Analyses" section

**Last Updated:** March 5, 2026
