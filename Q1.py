import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import PartialDependenceDisplay
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Load the datasets
data_2001 = pd.read_csv("/Users/anv/Desktop/Stats Project/pc01_vd_clean_pc01dist.csv")
data_2011 = pd.read_csv("/Users/anv/Desktop/Stats Project/pc11_vd_clean_pc11dist.csv")

# Merge datasets on a common key (modify based on actual key names)
merged_data = pd.merge(data_2001, data_2011, left_on="pc01_state_id", right_on="pc11_state_id", suffixes=('_2001', '_2011'))

# Calculate percentage change for population
merged_data["population_change_pct"] = (
    (merged_data["pc11_vd_t_p"] - merged_data["pc01_vd_t_p"]) / merged_data["pc01_vd_t_p"]
) * 100

# Feature engineering: Calculate changes for key variables
merged_data["schools_change"] = merged_data["pc11_vd_p_sch_gov"] - merged_data["pc01_vd_p_sch"]
merged_data["hospitals_change"] = merged_data["pc11_vd_all_hosp"] - merged_data["pc01_vd_hosp"]
merged_data["electricity_change"] = merged_data["pc11_vd_power_all"] - merged_data["pc01_vd_power_supl"]

# Add interaction term for schools and hospitals
merged_data["schools_hospitals_interaction"] = merged_data["schools_change"] * merged_data["hospitals_change"]

# Keep only necessary columns
features = [
    "schools_change",
    "hospitals_change",
    "electricity_change",
    "schools_hospitals_interaction",
    "population_change_pct"
]
data = merged_data[features].dropna()

# Outlier handling: Remove extreme outliers based on Z-scores
data = data[(np.abs(zscore(data)) < 3).all(axis=1)]

# Shift `population_change_pct` to make all values positive for log transformation
min_population_change = data["population_change_pct"].min()
shift_value = abs(min_population_change) + 1 if min_population_change <= 0 else 0
data["log_population_change"] = np.log1p(data["population_change_pct"] + shift_value)

# Split into predictors and target variable
X = data.drop(columns=["population_change_pct", "log_population_change"])
y_log = data["log_population_change"]

# Standardize predictors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA to reduce dimensionality
pca = PCA(n_components=2)  # Use 2 components for simplicity
X_pca = pca.fit_transform(X_scaled)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_log, test_size=0.2, random_state=42)

# Fit a robust regression model
model = sm.RLM(y_train, sm.add_constant(X_train), M=sm.robust.norms.HuberT()).fit()
print("\nRobust Regression Summary:\n", model.summary())

# Predict using the robust regression model
y_pred_robust = model.predict(sm.add_constant(X_test))
mse_robust = mean_squared_error(y_test, y_pred_robust)
print(f"Mean Squared Error (Robust Regression): {mse_robust:.2f}")

# Breusch-Pagan Test for heteroscedasticity
_, bp_pval, _, _ = het_breuschpagan(y_train - model.predict(sm.add_constant(X_train)), sm.add_constant(X_train))
print(f"Breusch-Pagan p-value (after log-transform): {bp_pval:.4f}")

# Variance Inflation Factor (VIF) to check multicollinearity
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
print("\nVariance Inflation Factors (VIF):\n", vif_data)

# Quantile Regression
quant_reg = QuantReg(y_train, sm.add_constant(X_train))
quant_reg_model = quant_reg.fit(q=0.5)  # Median regression
print("\nQuantile Regression Summary:\n", quant_reg_model.summary())

# Fit a Random Forest model
rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# Predict using the Random Forest model
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Mean Squared Error (Random Forest): {mse_rf:.2f}")
print(f"R-squared (Random Forest): {r2_rf:.2f}")

# Partial Dependence Plots (PDP) for Random Forest on original features
fig, ax = plt.subplots(figsize=(12, 6))
PartialDependenceDisplay.from_estimator(rf_model, X_scaled, [0, 1, 2], feature_names=X.columns, ax=ax)
plt.title("Partial Dependence Plots (Random Forest)")
plt.show()

# Plot residuals for both models
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_test - y_pred_robust, color="blue", label="Robust Regression Residuals")
plt.scatter(y_test, y_test - y_pred_rf, color="green", label="Random Forest Residuals")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Observed Log Population Growth")
plt.ylabel("Residuals")
plt.legend()
plt.title("Residual Plot (After Log-Transformation)")
plt.show()

# Correlation matrix of original predictors
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Pairplot to visualize distributions and relationships
sns.pairplot(data)
plt.show()