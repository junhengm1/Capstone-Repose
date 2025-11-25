import numpy as np
import pandas as pd
import geopandas as gpd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

# 1. Import data
x_vector = pd.read_csv('/dataset/x_vector.csv', index_col='property_gid')
y_labels = pd.read_csv('/dataset/y_labels.csv', index_col='property_gid')

# Unified index (ensure feature and label matching)
common_index = x_vector.index.intersection(y_labels.index)
x_matched = x_vector.loc[common_index]
y_matched = y_labels.loc[common_index]

# Generate final labels (excluding contradictory samples)
y = y_matched['is_high_risk'].copy()
y[(y_matched['is_high_risk'] == 1) & (y_matched['is_low_risk'] == 1)] = np.nan
y = y.dropna().astype(int)

# Separate labeled and unlabeled samples
labeled_idx = y.index
unlabeled_idx = x_vector.index.difference(labeled_idx)
x_labeled = x_vector.loc[labeled_idx]
x_unlabeled = x_vector.loc[unlabeled_idx]

fire_history_cols = ['fire_count', 'yrs_since_last_burn']
x_labeled.loc[:, fire_history_cols] = np.nan

#--------------------------------------------------------------
# Step 1: Supervised learning training initial risk model
#--------------------------------------------------------------

# Split the dataset into a training set (80%) and a validation set (20%).
x_train, x_val, y_train, y_val = train_test_split(
    x_labeled, y, test_size=0.2, random_state=42, stratify=y
)

# Define LightGBM parameters
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'n_estimators': 1000,
    'max_depth': 6,
    'num_leaves': 31,
    'class_weight':{0: 1.0, 1: 1.1}, # Balancing high/low risk samples 'balanced'
    'verbose': -1
}

n_runs = 5
best_auc = 0.0
best_model = None

for run in range(n_runs):
    model = lgb.LGBMClassifier(**params)
    model.fit(
        x_train, y_train,
        eval_set=[(x_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=0)],
        eval_metric='auc'
    )
    current_auc = model.best_score_['valid_0']['auc']

    if current_auc > best_auc:
        best_auc = current_auc
        best_model = model

print(f"\n=== Best Model Selected (AUC: {best_auc:.4f}) ===")
y_val_pred = best_model.predict_proba(x_val)[:, 1]
print("Classification report (threshold 0.5):")
print(classification_report(
    y_val,
    (y_val_pred >= 0.5).astype(int),
    labels=[0, 1],
    target_names=['Low Risk (Renewable Candidate)', 'High Risk'],
    digits=4,
    zero_division=0
))

feature_importance = pd.DataFrame({
    'feature': x_labeled.columns,
    'importance':best_model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

print("="*60)
print("Top 10 Feature Importance ：")
print("="*60)
for idx, row in feature_importance.iterrows():
    print(f"{row['feature']:<30} Importance: {row['importance']:>5d}")

#--------------------------------------------------------------
# Step 2: Semi-supervised learning optimizes unlabeled samples
#--------------------------------------------------------------
x_extended = x_train.copy()
y_extended = y_train.copy()
current_model = best_model

max_iter = 3
new_samples_ratio_per_iter = 0.2

for i in range(max_iter):

    target_new_samples = int(len(x_extended) * new_samples_ratio_per_iter)
    unlabeled_pred = current_model.predict_proba(x_unlabeled)[:, 1]

    k1 = int(target_new_samples * 0.5)
    k2 = target_new_samples - k1
    high_risk_idx = np.argsort(unlabeled_pred)[-k1:] if k1 > 0 else []
    low_risk_idx = np.argsort(unlabeled_pred)[:k2] if k2 > 0 else []
    selected_idx = np.unique(np.concatenate([high_risk_idx, low_risk_idx]))
    new_y = np.zeros(len(selected_idx))
    new_y[:len(high_risk_idx)] = 1
    new_x = x_unlabeled.iloc[selected_idx]
    new_y = pd.Series(new_y, name='label')

    x_extended = pd.concat([x_extended, new_x], ignore_index=True)
    y_extended = pd.concat([y_extended, new_y], ignore_index=True)

    current_model = lgb.LGBMClassifier(**params)
    current_model.fit(
        x_extended, y_extended,
        eval_set=[(x_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=0)]
    )

    iter_auc = current_model.best_score_['valid_0']['auc']
    print(f"\n{'-'*60}")
    print(f"Iteration {i+1}/{max_iter} AUC: {iter_auc:.4f}")

    x_unlabeled = x_unlabeled[~x_unlabeled.index.isin(selected_idx)].reset_index(drop=True)

print(f"\n{'='*100}")
print("FINAL SEMI-SUPERVISED MODEL EVALUATION REPORT")

y_val_pred_prob = current_model.predict_proba(x_val)[:, 1]
y_val_pred = (y_val_pred_prob >= 0.5).astype(int)

print(f"Final Model Validation AUC: {current_model.best_score_['valid_0']['auc']:.4f}")
print("\nClassification Report (Threshold = 0.5):")
print(classification_report(
    y_val,
    y_val_pred,
    labels=[0, 1],
    target_names=['Low Risk (Renewable Candidate)', 'High Risk'],
    digits=4,
    zero_division=0
))

risk_prob = current_model.predict_proba(x_labeled)[:, 1]
risk_score = (risk_prob * 100).round(1)

pred_result = pd.DataFrame({
    'property_gid': x_labeled.index,
    'risk_probability': risk_prob.round(4),
    'risk_score': risk_score,
    'risk_level': pd.cut(
        risk_score,
        bins=[0, 30, 80, 100],
        labels=['Low', 'Medium', 'High']
    ),
    'original_level': y_labels.loc[x_labeled.index, 'is_high_risk'].map({0: 'Low', 1: 'High'})
}).reset_index(drop=True).reset_index(names='plot_index')

level_map = {'Low': 0, 'Medium': 1, 'High': 2}
pred_result['risk_level_num'] = pred_result['risk_level'].map(level_map)
pred_result['original_level_num'] = pred_result['original_level'].map(level_map)

plt.figure(figsize=(14, 6))
plt.plot(pred_result['plot_index'], pred_result['risk_level_num'], 'o-',
         color='#1f77b4', label='Predicted Risk Level', markersize=6, linewidth=1)
plt.plot(pred_result['plot_index'], pred_result['original_level_num'], 's-',
         color='#ff4444', label='Original Risk Level', markersize=6, linewidth=1)

mismatch_idx = pred_result[pred_result['risk_level_num'] != pred_result['original_level_num']].index
plt.scatter(mismatch_idx, pred_result.loc[mismatch_idx, 'risk_level_num'],
            color='orange', marker='x', s=100, linewidth=2, label='Mismatched Samples')

plt.yticks([0, 1, 2], ['Low (0)', 'Medium (1)', 'High (2)'], fontsize=10)
plt.xlabel('Sample Index (0, 1, 2...)', fontsize=11)
plt.ylabel('Risk Level', fontsize=11)
plt.title('Predicted vs Original Risk Level (Only Labeled Samples)', fontsize=12, pad=15)
plt.legend(loc='best', fontsize=10)
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()

mismatch_count = len(mismatch_idx)
match_rate = (1 - mismatch_count / len(pred_result)) * 100
print(f"Num of inconsistent samples: {mismatch_count}")
print(f"Match rate: {match_rate:.2f}%")

# Predict the risk probability of all samples to be evaluated (including labeled and unlabeled samples).
all_property = pd.concat([x_labeled, x_unlabeled])
risk_prob = current_model.predict_proba(all_property)[:, 1]

# Mapped to 0-100 points
risk_score = (risk_prob * 100).round(1)

# Adjust score distribution
high_risk_threshold = np.percentile(risk_score, 80) # Top 20% are high-risk

# Generate the final result table
risk_result = pd.DataFrame({
    'property_gid': all_property.index,
    'risk_probability': risk_prob.round(4),
    'risk_score': risk_score,
    'risk_level': pd.cut(
        risk_score,
        bins=[0, 30, 80, 100],
        labels=['Low', 'Medium', 'High']
    )
}).set_index('property_gid')

risk_result = risk_result.sort_index()
risk_result.to_csv('/content/drive/My Drive/risk_results_relabel.csv')

labeled_result = pd.DataFrame({
    'property_gid': x_labeled.index,
    'risk_probability': np.nan,
    'risk_score': np.nan,
    'risk_level': y_labels.loc[x_labeled.index, 'is_high_risk'].map({0: 'Low', 1: 'High'})
}).set_index('property_gid')

unlabeled_prob = current_model.predict_proba(x_unlabeled)[:, 1]
unlabeled_score = (unlabeled_prob * 100).round(1)
unlabeled_result = pd.DataFrame({
    'property_gid': x_unlabeled.index,
    'risk_probability': unlabeled_prob.round(4),
    'risk_score': unlabeled_score,
    'risk_level': pd.cut(
        unlabeled_score,
        bins=[0, 30, 80, 100],
        labels=['Low', 'Medium', 'High']
    )
}).set_index('property_gid')


risk_result = pd.concat([labeled_result, unlabeled_result]).sort_index()
risk_result.to_csv('/content/drive/My Drive/risk_results.csv')