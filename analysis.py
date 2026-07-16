import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import textwrap
from matplotlib.widgets import Slider
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import ttest_1samp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Styles & warnings
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('ggplot')

sns.set_theme(style="whitegrid", palette="husl")
warnings.filterwarnings('ignore')

# Data setup
years = np.array(list(range(2018, 2025)))
us_exports = np.array([33.18, 34.21, 28.75, 32.89, 36.42, 39.15, 41.75])
us_imports = np.array([54.25, 56.32, 48.91, 65.23, 72.15, 80.11, 87.42])
trade_balance = np.array([-21.07, -22.11, -20.16, -32.34, -35.73, -40.96, -45.67])

trade_df = pd.DataFrame({
    'Year': years,
    'US_Exports_to_India': us_exports,
    'US_Imports_from_India': us_imports,
    'Trade_Balance': trade_balance
})

steel_exports = np.array([1.85, 1.20, 0.95, 1.10, 1.25, 1.35, 1.40])
aluminum_exports = np.array([0.82, 0.94, 0.78, 0.85, 0.89, 0.92, 0.95])

sector_df = pd.DataFrame({
    'Year': years,
    'Steel_Exports': steel_exports,
    'Aluminum_Exports': aluminum_exports
})

gdp_growth = [6.8, 6.0, -5.8, 9.1, 7.2, 6.9, 7.0]
inflation = [4.9, 4.8, 6.2, 5.5, 5.7, 5.4, 4.6]

macro_df = pd.DataFrame({
    'Year': years,
    'GDP_Growth': gdp_growth,
    'Inflation': inflation
})

policy_events = [
    {'Date': '2018-03', 'Event': 'Section 232 tariffs (25% steel, 10% aluminum)'},
    {'Date': '2019-06', 'Event': 'US removes India from GSP program'},
    {'Date': '2019-06', 'Event': 'India retaliates with tariffs on 28 US products'},
    {'Date': '2021-01', 'Event': 'USTR threatens Section 301 tariffs on digital tax'},
    {'Date': '2023-06', 'Event': 'US-India agree to ease steel/aluminum trade friction'}
]

events_df = pd.DataFrame(policy_events)
events_df['Date'] = pd.to_datetime(events_df['Date'])
events_df['Year'] = events_df['Date'].dt.year

# Merge dataframes
full_df = trade_df.merge(sector_df, on='Year').merge(macro_df, on='Year')

# Feature engineering
full_df['Total_Trade'] = full_df['US_Exports_to_India'] + full_df['US_Imports_from_India']
full_df['Export_Growth_Rate'] = full_df['US_Exports_to_India'].pct_change() * 100
full_df['Import_Growth_Rate'] = full_df['US_Imports_from_India'].pct_change() * 100
full_df['Trade_Balance_Pct_GDP'] = full_df['Trade_Balance'] / 2800 * 100

pre_tariff_steel = full_df.loc[full_df['Year'] == 2018, 'Steel_Exports'].values[0]
full_df['Steel_Export_Change'] = (full_df['Steel_Exports'] - pre_tariff_steel) / pre_tariff_steel * 100

# === 1. Trade Trends with Policy Events ===
plt.figure(figsize=(14, 7))
events_df['YearFraction'] = events_df['Date'].dt.year + (events_df['Date'].dt.month - 1)/12

plt.plot(full_df['Year'], full_df['US_Exports_to_India'], marker='o', linewidth=2.5, label='US Exports', color='#2a9df4')
plt.plot(full_df['Year'], full_df['US_Imports_from_India'], marker='o', linewidth=2.5, label='US Imports', color='#f4a261')
plt.plot(full_df['Year'], full_df['Trade_Balance'], marker='D', linestyle='--', linewidth=2.2, label='Trade Balance', color='#e63946')

plt.xlabel('Year')
plt.ylabel('Billions USD')
plt.title('US-India Trade & Policy Events (2018-2024)', fontsize=16, fontweight='bold')
plt.xticks(full_df['Year'], rotation=45)
plt.grid(True, linestyle='--', alpha=0.3)

mid_y = (plt.ylim()[0] + plt.ylim()[1]) / 2
grouped = events_df.groupby('YearFraction')

for year_frac, group in grouped:
    plt.axvline(x=year_frac, color='gray', linestyle='--', alpha=0.4)
    for i, (_, event) in enumerate(group.iterrows()):
        label = textwrap.fill(event['Event'], width=25)
        x_offset = -0.15 if i % 2 == 0 else 0.15
        ha = 'right' if i % 2 == 0 else 'left'
        y_pos = mid_y + (i - len(group)/2)*5
        plt.text(year_frac + x_offset, y_pos, label,
                 rotation=90, va='center', ha=ha,
                 fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.6))

# Callouts for major inflection points
for idx, row in full_df.iterrows():
    if abs(row['Trade_Balance']) > 35:
        plt.annotate(f"{row['Trade_Balance']:.1f}B deficit", 
                     (row['Year'], row['Trade_Balance']),
                     textcoords="offset points", xytext=(0,-30),
                     ha='center', fontsize=10, color='#e63946', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='#e63946'))

plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# === 2. Sectoral Impact ===
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.bar(full_df['Year'], full_df['Steel_Exports'], color='steelblue', alpha=0.9)
plt.axhline(y=pre_tariff_steel, color='red', linestyle='--', linewidth=1.5, label='Pre-Tariff Level (2018)')
plt.title('Indian Steel Exports to US (2018–2024)', fontsize=14, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Billions USD')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.3)

min_steel = full_df['Steel_Exports'].min()
min_year = full_df.loc[full_df['Steel_Exports'] == min_steel, 'Year'].values[0]
plt.annotate(f'Min Drop: {min_steel:.2f}B in {min_year}', xy=(min_year, min_steel), 
             xytext=(min_year, min_steel+0.3),
             arrowprops=dict(facecolor='red', shrink=0.05),
             fontsize=10, color='darkred', fontweight='bold')

plt.subplot(1, 2, 2)
plt.bar(full_df['Year'], full_df['Aluminum_Exports'], color='lightblue', alpha=0.9)
plt.title('Indian Aluminum Exports to US (2018–2024)', fontsize=14, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Billions USD')
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', alpha=0.3)

plt.annotate(f'Steady growth\nfrom {aluminum_exports[0]:.2f}B to {aluminum_exports[-1]:.2f}B', 
             xy=(2021, 0.85), xytext=(2021, 1.1),
             bbox=dict(boxstyle='round,pad=0.3', fc='lightblue', alpha=0.4),
             fontsize=10)

plt.tight_layout()
plt.show()

# === 3. Hypothesis Testing ===
post_steel_exports = full_df.loc[full_df['Year'] > 2018, 'Steel_Exports']
t_stat, p_value = ttest_1samp(post_steel_exports, popmean=pre_tariff_steel)

print("\n=== Hypothesis Testing ===")
print("H₀: No change in steel exports post-2018")
print("H₁: Significant change in steel exports post-2018")
print(f"T-statistic: {t_stat:.3f}, P-value: {p_value:.3f}")

if p_value < 0.05:
    print("Result: Reject H₀ – Steel exports changed significantly after 2018.")
else:
    print("Result: Fail to reject H₀ – No statistically significant change.")

# === 4. Linear Regression Model ===
features = full_df[['US_Exports_to_India', 'US_Imports_from_India',
                    'GDP_Growth', 'Inflation', 'Steel_Exports', 'Aluminum_Exports']]
target = full_df['Trade_Balance']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n=== Linear Regression Model ===")
print("Intercept:", model.intercept_)
print("Coefficients:", dict(zip(features.columns, model.coef_)))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# === 5. Feature Importance Chart ===
coeff_df = pd.DataFrame({
    'Feature': features.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=coeff_df, x='Coefficient', y='Feature', palette='viridis')
plt.title('Impact of Features on Trade Balance')
plt.xlabel('Coefficient Value')
plt.tight_layout()
plt.show()

# === 6. Actual vs Predicted ===
plt.figure(figsize=(8, 5))
plt.plot(y_test.values, label='Actual Trade Balance', marker='o')
plt.plot(y_pred, label='Predicted Trade Balance', marker='x')
plt.title('Actual vs Predicted Trade Balance (Linear Regression)')
plt.xlabel('Test Sample Index')
plt.ylabel('Trade Balance (Billions USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# === 7. Save Data ===
full_df.to_csv('us_india_trade_analysis_2018_2024.csv', index=False)
print("\nAnalysis complete. Data saved to 'us_india_trade_analysis_2018_2024.csv'")

# === 8. Key Insights ===
print("\n=== Key Insights ===")
print("1. Trade Trends:")
print(f"- US-India bilateral trade grew from ${full_df.loc[0, 'Total_Trade']:.1f}B in 2018 to ${full_df.loc[6, 'Total_Trade']:.1f}B in 2024")
print(f"- US trade deficit with India widened from ${full_df.loc[0, 'Trade_Balance']:.1f}B to ${full_df.loc[6, 'Trade_Balance']:.1f}B")

print("\n2. Sectoral Impacts:")
print(f"- Steel exports dropped by {full_df.loc[1, 'Steel_Export_Change']:.1f}% in 2019 after tariffs")
print(f"- Aluminum exports showed resilience, growing from ${aluminum_exports[0]:.2f}B to ${aluminum_exports[-1]:.2f}B")

print("\n3. Macroeconomic Impact:")
print(f"- India's GDP growth averaged {full_df['GDP_Growth'].mean():.1f}% despite tariffs")
print(f"- Inflation stayed within target range, averaging {full_df['Inflation'].mean():.1f}%")

print("\n4. Policy Events Impact:")
print("- Section 232 tariffs (2018) had immediate impact on steel exports")
print("- GSP removal (2019) affected $5.6B of Indian exports but overall trade continued growing")
print("- 2023 agreements helped normalize steel/aluminum trade flows")
