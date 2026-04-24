"""
2025NS01076 — ML Assignment 2
Early Warning System for Student Performance
Streamlit App — exact equivalent of 2025NS01076_MLAssignment2.ipynb
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import io
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection   import train_test_split, learning_curve
from sklearn.preprocessing     import LabelEncoder, StandardScaler
from sklearn.linear_model      import LogisticRegression
from sklearn.tree              import DecisionTreeClassifier
from sklearn.ensemble          import (RandomForestClassifier,
                                        GradientBoostingClassifier,
                                        VotingClassifier)
from sklearn.svm               import SVC
from sklearn.neural_network    import MLPClassifier
from sklearn.metrics           import (accuracy_score, precision_score,
                                        recall_score, f1_score,
                                        confusion_matrix,
                                        classification_report)

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 110

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Assignment 2 — Student Performance",
    layout="wide"
)

st.title("Early Warning System for Student Performance")
st.markdown(
    "**ML Assignment 2 | Student ID: 2025NS01076**  \n"
    "Upload `StudentPerformance_Dataset.csv` to run the full pipeline."
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# FILE UPLOAD  (replaces: from google.colab import files / files.upload())
# ─────────────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload StudentPerformance_Dataset.csv", type=["csv"]
)
if uploaded_file is None:
    st.info("Please upload the dataset CSV to proceed.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.success(f"Dataset loaded — {len(df)} rows × {len(df.columns)} columns.")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — PROBLEM UNDERSTANDING
# (notebook Cell 4 — problem type, X/y definition, class distribution)
# ─────────────────────────────────────────────────────────────────────────────
st.header("Problem Understanding")

col_t1a, col_t1b = st.columns(2)

with col_t1a:
    st.subheader("Problem Type")
    st.markdown(
        "- **Type:** Supervised Binary Classification  \n"
        "- **Target:** `Pass` column — 0 = Fail, 1 = Pass  \n"
        "- **Features:** 7 input variables  \n"
        "- **Samples:** " + str(len(df))
    )
    st.subheader("Feature Summary")
    feat_info = pd.DataFrame({
        "Feature":  ["Study_Hours", "Attendance", "Previous_Marks",
                     "Assignment_Score", "Sleep_Hours", "Internet_Usage", "Extra_Coaching"],
        "Type":     ["Numerical", "Numerical", "Numerical",
                     "Numerical", "Numerical", "Categorical (encoded)", "Binary"],
        "Role":     ["Input X"] * 7,
    })
    st.dataframe(feat_info, hide_index=True, use_container_width=True)

with col_t1b:
    st.subheader("Class Distribution")
    counts = df['Pass'].value_counts()
    for label, count in counts.items():
        st.write(
            f'{"Pass" if label==1 else "Fail"} ({label}) : '
            f'**{count}** students ({count/len(df)*100:.1f}%)'
        )
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

st.markdown(
    "**Justification for Classification:** The target is discrete (0 or 1). "
    "Classification models output class probabilities, enabling risk ranking. "
    "Metrics like Recall directly measure how well the Early Warning System catches failing students."
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — DATA PREPROCESSING
# (notebook Cells 5–14)
# ─────────────────────────────────────────────────────────────────────────────
st.header("Data Preprocessing")

# 2.1 Missing values (Cell 6)
st.subheader("2.1 Missing Value Check")
missing = df.isnull().sum().rename("Missing Count")
st.dataframe(missing.to_frame(), use_container_width=True)
total_missing = int(missing.sum())
if total_missing == 0:
    st.success("No missing values found — no imputation required.")
else:
    st.warning(f"{total_missing} missing values found — imputation applied.")

# 2.2 Label encoding (Cell 8)
st.subheader("2.2 Label Encoding — Internet_Usage")
df_processed = df.copy()
le = LabelEncoder()
df_processed['Internet_Usage'] = le.fit_transform(df_processed['Internet_Usage'])
encoding_map = dict(zip(le.classes_, le.transform(le.classes_).tolist()))
enc_df = pd.DataFrame(
    list(encoding_map.items()),
    columns=["Original Value", "Encoded Integer"]
)
st.dataframe(enc_df, hide_index=True, use_container_width=False)

# 2.3 Feature / target split (Cell 10)
st.subheader("2.3 Feature (X) and Target (y) Split")
X = df_processed.drop('Pass', axis=1)
y = df_processed['Pass']
st.write(f"Feature matrix **X**: `{X.shape}`   |   Target vector **y**: `{y.shape}`")
st.write(f"Features: `{list(X.columns)}`")

# 2.4 Stratified train-test split (Cell 12)
st.subheader("2.4 Stratified Train-Test Split (80 / 20)")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
col_s1, col_s2, col_s3, col_s4 = st.columns(4)
col_s1.metric("Training Samples", len(X_train))
col_s2.metric("Test Samples",     len(X_test))
col_s3.metric("Train Pass Rate",  f"{y_train.mean()*100:.1f}%")
col_s4.metric("Test Pass Rate",   f"{y_test.mean()*100:.1f}%")
st.caption("stratify=y preserves the original class ratio in both splits.")

# 2.5 Feature scaling (Cell 14)
st.subheader("2.5 Feature Scaling — StandardScaler")
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # fit on train only
X_test_sc  = scaler.transform(X_test)         # transform test with train params
scale_df = pd.DataFrame({
    "Feature":    list(X.columns),
    "Train Mean": scaler.mean_.round(3),
    "Train Std":  scaler.scale_.round(3),
})
st.dataframe(scale_df, hide_index=True, use_container_width=False)
st.caption(
    "Scaler fitted on training data only. "
    "Test set transformed using training parameters to prevent data leakage."
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 — EDA
# (notebook Cells 15–22)
# ─────────────────────────────────────────────────────────────────────────────
st.header("Exploratory Data Analysis (EDA)")

# 3.1 Distribution (Cell 16)
st.subheader("3.1 Pass vs Fail Distribution")
fig, ax = plt.subplots(figsize=(6, 4))
counts_plot = df['Pass'].value_counts()
bars = ax.bar(
    ['Fail (0)', 'Pass (1)'], counts_plot.values,
    color=['#e74c3c', '#2ecc71'], edgecolor='white', width=0.5
)
for bar, val in zip(bars, counts_plot.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
            str(val), ha='center', fontweight='bold', fontsize=12)
ax.set_title('Pass vs Fail — Count', fontweight='bold')
ax.set_ylabel('Number of Students')
ax.set_ylim(0, max(counts_plot.values) + 80)
plt.suptitle('Target Variable Distribution', fontsize=13, fontweight='bold')
plt.tight_layout()
st.pyplot(fig)
plt.close()

# 3.2 Correlation heatmap (Cell 18)
st.subheader("3.2 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(
    df_processed.corr(), annot=True, fmt='.2f',
    cmap='twilight', center=0, ax=ax,
    linewidths=0.5, annot_kws={'size': 9}
)
ax.set_title('Feature Correlation Heatmap', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.write("**Correlation with Pass (target) — ranked:**")
corr_series = (
    df_processed.corr()['Pass'].drop('Pass')
    .sort_values(ascending=False)
    .round(3)
)
st.dataframe(corr_series.rename("Correlation").to_frame(), use_container_width=False)

# 3.3 Box plots (Cell 20)
st.subheader("3.3 Feature Distributions — Pass vs Fail (Box Plots)")
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
num_feats = ['Study_Hours','Attendance','Previous_Marks','Assignment_Score','Sleep_Hours']
for i, feat in enumerate(num_feats):
    bp = axes[i].boxplot(
        [df[df['Pass']==0][feat], df[df['Pass']==1][feat]],
        labels=['Fail', 'Pass'],
        patch_artist=True,
        medianprops=dict(color='black', linewidth=2.5)
    )
    bp['boxes'][0].set_facecolor('#1b0a08')
    bp['boxes'][1].set_facecolor('#db82b9')
    for j, grp in enumerate([df[df['Pass']==0][feat], df[df['Pass']==1][feat]]):
        axes[i].plot(j+1, grp.mean(), 'D', color='white', markersize=7, zorder=5,
                     markeredgecolor='black', markeredgewidth=1)
    axes[i].set_title(feat.replace('_', ' '), fontweight='bold', fontsize=10)
    axes[i].set_ylabel('Value')
axes[5].axis('off')
axes[5].text(0.5, 0.5, 'Diamond (◆) = Mean\nBox = IQR\nLine = Median',
             ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='blue', edgecolor='gray'))
fig.suptitle('Feature Distributions — Pass vs Fail', fontsize=13, fontweight='bold')
plt.tight_layout()
st.pyplot(fig)
plt.close()

# 3.4 Pass rate analysis (Cell 22)
st.subheader("3.4 Pass Rate Analysis — All 7 Features")

def pass_rate_bar(ax, series_x, series_pass, title, x_labels=None):
    """Plots pass rate (%) per category with count labels."""
    df_tmp = pd.DataFrame({'x': series_x, 'pass': series_pass})
    grp = df_tmp.groupby('x')['pass'].agg(['mean','count']).reset_index()
    labels    = x_labels if x_labels else grp['x'].astype(str).tolist()
    pass_rates = grp['mean'].values * 100
    counts_arr = grp['count'].values
    avg        = series_pass.mean() * 100
    colors     = plt.cm.coolwarm(pass_rates / 100)
    bars       = ax.bar(labels, pass_rates, color=colors, edgecolor='white', linewidth=1.2)
    for bar, rate, cnt in zip(bars, pass_rates, counts_arr):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                f'{rate:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()/2,
                f'n={cnt}', ha='center', va='center', fontsize=7,
                color='white', fontweight='bold')
    ax.axhline(y=avg, color='navy', linestyle='--', linewidth=1.3,
               alpha=0.8, label=f'Overall avg: {avg:.1f}%')
    ax.set_title(title, fontweight='bold', fontsize=9)
    ax.set_ylabel('Pass Rate (%)', fontsize=8)
    ax.set_ylim(0, 90)
    ax.legend(fontsize=7)
    ax.tick_params(axis='x', labelsize=7.5)

fig, axes = plt.subplots(3, 3, figsize=(16, 13))
axes = axes.flatten()

att_bins = pd.cut(df['Attendance'], bins=[38,55,65,75,85,100])
pass_rate_bar(axes[0], att_bins.astype(str), df['Pass'], 'Attendance Range vs Pass Rate')

pm_bins = pd.cut(df['Previous_Marks'], bins=[0,40,55,65,80,101])
pass_rate_bar(axes[1], pm_bins.astype(str), df['Pass'], 'Previous Marks Range vs Pass Rate')

as_bins = pd.cut(df['Assignment_Score'], bins=[0,40,55,65,80,101])
pass_rate_bar(axes[2], as_bins.astype(str), df['Pass'], 'Assignment Score Range vs Pass Rate')

pass_rate_bar(axes[3], df['Study_Hours'],    df['Pass'], 'Study Hours vs Pass Rate')
pass_rate_bar(axes[4], df['Sleep_Hours'],    df['Pass'], 'Sleep Hours vs Pass Rate')
pass_rate_bar(axes[5], df['Internet_Usage'], df['Pass'], 'Internet Usage vs Pass Rate')
pass_rate_bar(axes[6], df['Extra_Coaching'].map({0:'No',1:'Yes'}), df['Pass'],
              'Extra Coaching vs Pass Rate')
axes[7].axis('off')
axes[8].axis('off')

fig.suptitle('Pass Rate Analysis — All 7 Features', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
st.pyplot(fig)
plt.close()
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TASK 4 — MODEL DEVELOPMENT
# (notebook Cells 23–40)
# ─────────────────────────────────────────────────────────────────────────────
st.header("Model Development")
st.info("Training all 8 models — please wait...")

# 4.1 Logistic Regression (Cell 24)
lr = LogisticRegression(random_state=42, max_iter=500)
lr.fit(X_train_sc, y_train)
y_pred_lr = lr.predict(X_test_sc)

# 4.2 Decision Tree (Cell 26)
dt = DecisionTreeClassifier(random_state=42, max_depth=5, criterion='gini')
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# 4.3 Random Forest (Cell 28)
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_features='sqrt')
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# 4.4 MLP / ANN (Cell 30)
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
    max_iter=500, random_state=42, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=10
)
mlp.fit(X_train_sc, y_train)
y_pred_mlp = mlp.predict(X_test_sc)

# 4.5 SVM — RBF Kernel (Cell 32)
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
svm_rbf.fit(X_train_sc, y_train)
y_pred_svm_rbf = svm_rbf.predict(X_test_sc)

# 4.6 SVM — Linear Kernel (Cell 34)
svm_lin = SVC(kernel='linear', C=0.1, class_weight='balanced',
              probability=True, random_state=42)
svm_lin.fit(X_train_sc, y_train)
y_pred_svm_lin = svm_lin.predict(X_test_sc)

# 4.7 Gradient Boosting (Cell 36)
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                 max_depth=3, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

# 4.8 Voting Ensemble (Cell 38)
voting = VotingClassifier(
    estimators=[
        ('rf',  RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb',  GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
    ],
    voting='soft'
)
voting.fit(X_train_sc, y_train)
y_pred_voting = voting.predict(X_test_sc)

st.success("All 8 models trained successfully!")

model_summary = pd.DataFrame({
    "Model":       ["Logistic Regression","Decision Tree","Random Forest",
                    "MLP / ANN","SVM (RBF)","SVM (Linear)",
                    "Gradient Boosting","Voting Ensemble"],
    "Type":        ["Linear","Tree","Bagging Ensemble","Neural Network",
                    "Kernel","Kernel","Boosting Ensemble","Meta-Ensemble"],
    "Scaled Data": ["Yes","No","No","Yes","Yes","Yes","No","Yes"],
    "Bonus":       ["No","No","No","No","Yes","Yes","Yes","Yes"],
})
st.dataframe(model_summary, hide_index=True, use_container_width=True)

# 4.9 Training & Learning Curves (Cell 40)
st.subheader("4.9 Training & Learning Curves")
train_sizes = np.linspace(0.1, 1.0, 8)

def plot_lc(ax, estimator, X_data, y_data, title, color):
    tr_sz, tr_sc, cv_sc = learning_curve(
        estimator, X_data, y_data, cv=5,
        train_sizes=train_sizes, scoring='accuracy'
    )
    ax.plot(tr_sz, tr_sc.mean(axis=1), 'o-', color=color, lw=2.2, label='Train Accuracy')
    ax.fill_between(tr_sz, tr_sc.mean(axis=1)-tr_sc.std(axis=1),
                           tr_sc.mean(axis=1)+tr_sc.std(axis=1), alpha=0.12, color=color)
    ax.plot(tr_sz, cv_sc.mean(axis=1), 's--', color='#1f77b4', lw=2, label='Validation Accuracy')
    ax.fill_between(tr_sz, cv_sc.mean(axis=1)-cv_sc.std(axis=1),
                           cv_sc.mean(axis=1)+cv_sc.std(axis=1), alpha=0.12, color='#1f77b4')
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.set_xlabel('Training Set Size'); ax.set_ylabel('Accuracy')
    ax.set_ylim(0.30, 0.85); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

fig, axes = plt.subplots(3, 3, figsize=(16, 13))
axes = axes.flatten()

plot_lc(axes[0], LogisticRegression(random_state=42,max_iter=500),
        X_train_sc, y_train, 'Logistic Regression', '#8e44ad')

plot_lc(axes[1], DecisionTreeClassifier(random_state=42,max_depth=5),
        X_train, y_train, 'Decision Tree', '#16a085')

plot_lc(axes[2], RandomForestClassifier(n_estimators=50,random_state=42),
        X_train, y_train, 'Random Forest', '#d35400')

plot_lc(axes[3], SVC(kernel='rbf',C=1.0,probability=True,random_state=42),
        X_train_sc, y_train, 'SVM (RBF Kernel)', '#c0392b')

plot_lc(axes[4], SVC(kernel='linear',C=0.1,class_weight='balanced',probability=True,random_state=42),
        X_train_sc, y_train, 'SVM (Linear Kernel)', '#2980b9')

plot_lc(axes[5], GradientBoostingClassifier(n_estimators=50,random_state=42),
        X_train, y_train, 'Gradient Boosting', '#27ae60')

epochs = list(range(1, len(mlp.loss_curve_)+1))
axes[6].plot(epochs, mlp.loss_curve_, 'o-', color='#2c3e50', lw=2.5, ms=5, label='Training Loss')
if hasattr(mlp,'validation_scores_') and mlp.validation_scores_:
    axes[6].plot(range(1,len(mlp.validation_scores_)+1),
                 [1-s for s in mlp.validation_scores_],
                 's--', color='#e67e22', lw=2, ms=5, label='Validation Loss')
axes[6].set_title('MLP/ANN — Loss per Epoch', fontweight='bold', fontsize=10)
axes[6].set_xlabel('Epoch'); axes[6].set_ylabel('Loss')
axes[6].legend(fontsize=8); axes[6].grid(True, alpha=0.3)

axes[7].plot(gb.train_score_, 'o-', color='#7f8c8d', lw=2.5, ms=3, label='GB Train Score')
axes[7].set_title('Gradient Boosting — Score per Tree', fontweight='bold', fontsize=10)
axes[7].set_xlabel('Number of Trees'); axes[7].set_ylabel('Training Score')
axes[7].legend(fontsize=8); axes[7].grid(True, alpha=0.3)

axes[8].axis('off')
axes[8].text(0.5,0.5,
    'Interpreting Learning Curves\n\n'
    'Train >> Val  → Overfitting\n'
    'Both low      → Underfitting\n'
    'Train ≈ Val   → Well-fitted\n\n'
    'Shaded band = ±1 std (5-fold CV)',
    ha='center', va='center', fontsize=10,
    bbox=dict(boxstyle='round,pad=0.6', facecolor='lavender', edgecolor='#666'))
    
fig.suptitle('Training & Learning Curves — All 8 Models', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout(); plt.show()

fig.suptitle('Training & Learning Curves — All 8 Models', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
st.pyplot(fig)
plt.close()
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TASK 5 — MODEL EVALUATION
# (notebook Cells 41–46)
# ─────────────────────────────────────────────────────────────────────────────
st.header("Model Evaluation")

# 5.1 Metrics summary (Cell 42)
st.subheader("5.1 Metrics Summary — All 8 Models")

def get_metrics(y_true, y_pred, name):
    return {
        'Model':     name,
        'Accuracy':  accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall':    recall_score(y_true, y_pred),
        'F1-Score':  f1_score(y_true, y_pred),
    }

eval_results = [
    get_metrics(y_test, y_pred_lr,      'Logistic Regression'),
    get_metrics(y_test, y_pred_dt,      'Decision Tree'),
    get_metrics(y_test, y_pred_rf,      'Random Forest'),
    get_metrics(y_test, y_pred_mlp,     'MLP / ANN'),
    get_metrics(y_test, y_pred_svm_rbf, 'SVM (RBF)'),
    get_metrics(y_test, y_pred_svm_lin, 'SVM (Linear)'),
    get_metrics(y_test, y_pred_gb,      'Gradient Boosting'),
    get_metrics(y_test, y_pred_voting,  'Voting Ensemble'),
]

metrics_df = pd.DataFrame(eval_results).set_index('Model')
mdf = metrics_df.copy()

# replaces: display(metrics_df.style.format('{:.2%}'))
st.dataframe(
    metrics_df.style.format('{:.2%}').background_gradient(cmap='YlGn', axis=0),
    use_container_width=True
)

st.write("**Best model per metric:**")
best_cols = st.columns(4)
for col, metric in zip(best_cols, ['Accuracy','Precision','Recall','F1-Score']):
    best_name = metrics_df[metric].idxmax()
    best_val  = metrics_df[metric].max()
    col.metric(metric, f"{best_val*100:.2f}%", best_name)

# 5.2 Performance comparison bar charts (Cell 44)
st.subheader("5.2 Performance Comparison")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()
for i, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1-Score']):
    sns.barplot(x=metrics_df.index, y=metrics_df[metric], ax=axes[i],
                palette='Oranges', hue=metrics_df.index, legend=False)
    axes[i].set_title(f'{metric} Comparison', fontweight='bold')
    axes[i].set_ylabel(metric)
    axes[i].set_xlabel('')
    axes[i].set_ylim(0, 1.05)
    axes[i].tick_params(axis='x', rotation=30)
    for container in axes[i].containers:
        axes[i].bar_label(container, fmt='%.2f', fontsize=8)
fig.suptitle('Model Performance Comparison — All 8 Models', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# 5.3 Confusion matrices (Cell 46)
st.subheader("5.3 Confusion Matrices")

def plot_cm(ax, y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='twilight', cbar=False, ax=ax,
                xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
    ax.set_title(model_name, fontweight='bold', fontsize=10)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
for ax, (y_pred, name) in zip(axes, [
    (y_pred_lr,      'Logistic Regression'),
    (y_pred_dt,      'Decision Tree'),
    (y_pred_rf,      'Random Forest'),
    (y_pred_mlp,     'MLP / ANN'),
    (y_pred_svm_rbf, 'SVM (RBF)'),
    (y_pred_svm_lin, 'SVM (Linear)'),
    (y_pred_gb,      'Gradient Boosting'),
    (y_pred_voting,  'Voting Ensemble'),
]):
    plot_cm(ax, y_test, y_pred, name)
fig.suptitle('Confusion Matrices — All 8 Models', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# 5.4 Best model justification + classification report
st.subheader("5.4 Best Model Justification")
best_model_name = metrics_df['F1-Score'].idxmax()
best_preds_map  = {
    'Logistic Regression': y_pred_lr,
    'Decision Tree':        y_pred_dt,
    'Random Forest':        y_pred_rf,
    'MLP / ANN':            y_pred_mlp,
    'SVM (RBF)':            y_pred_svm_rbf,
    'SVM (Linear)':         y_pred_svm_lin,
    'Gradient Boosting':    y_pred_gb,
    'Voting Ensemble':      y_pred_voting,
}
st.success(
    f"**Best Model: {best_model_name}** — selected on highest F1-Score "
    f"({metrics_df.loc[best_model_name,'F1-Score']*100:.2f}%).  \n"
    "F1-Score balances Precision and Recall. In an Early Warning System, both missing "
    "a failing student (False Negative) and generating false alarms (False Positive) "
    "have real consequences — F1-Score penalises both."
)
st.write(f"**Detailed Classification Report — {best_model_name}:**")
report_str = classification_report(
    y_test, best_preds_map[best_model_name],
    target_names=['Fail (0)', 'Pass (1)']
)
st.code(report_str, language=None)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TASK 6 — FEATURE IMPORTANCE & INSIGHTS
# (notebook Cells 47–50)
# ─────────────────────────────────────────────────────────────────────────────
st.header("Feature Importance and Insights")

# 6.1 RF vs GB importance (Cell 48)
st.subheader("6.1 Feature Importance — Random Forest vs Gradient Boosting vs Decision Tree")
rf_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
gb_imp = pd.Series(gb.feature_importances_, index=X.columns).sort_values(ascending=True)
dt_imp = pd.Series(dt.feature_importances_, index=X.columns).sort_values(ascending=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (imp, title, color) in zip(axes, [
    (rf_imp, 'Random Forest',    '#95a5a6'),
    (gb_imp, 'Gradient Boosting','#fd79a8'),
    (dt_imp, 'Decision Tree',    '#008080')
]):
    bars = ax.barh(imp.index, imp.values, color=color, edgecolor='white')
    for bar, val in zip(bars, imp.values):
        ax.text(val+0.002, bar.get_y()+bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)
    ax.set_title(f'Feature Importance — {title}', fontweight='bold')
    ax.set_xlabel('Importance Score')
    ax.axvline(x=1/7, color='navy', ls='--', alpha=0.5, label='Equal baseline (1/7)')
    ax.legend()
plt.suptitle('RF vs Gradient Boosting vs Decision Tree Feature Importance', fontsize=13, fontweight='bold', y=1.05)
plt.tight_layout()
st.pyplot(fig)
plt.close()

rank_data = sorted(
    zip(X.columns, rf.feature_importances_, gb.feature_importances_, dt.feature_importances_),
    key=lambda x: -x[1]
)
rank_df = pd.DataFrame(rank_data, columns=['Feature','RF Score','GB Score','DT Score'])
rank_df.insert(0, 'Rank', range(1, len(rank_df)+1))
rank_df['RF Score'] = rank_df['RF Score'].round(4)
rank_df['GB Score'] = rank_df['GB Score'].round(4)
rank_df['DT Score'] = rank_df['DT Score'].round(4)
st.dataframe(rank_df, hide_index=True, use_container_width=True)

# 6.2 Sleep Hours & Extra Coaching analysis (Cell 50)
st.subheader("6.2 Specific Analysis — Sleep Hours & Extra Coaching Impact")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sleep_analysis = df.groupby('Sleep_Hours')['Pass'].agg(['mean','count']).reset_index()
sleep_analysis.columns = ['Sleep_Hours','Pass_Rate','Count']
sleep_analysis['Pass_Rate'] *= 100

axes[0].bar(
    sleep_analysis['Sleep_Hours'], sleep_analysis['Pass_Rate'],
    color=plt.cm.coolwarm(sleep_analysis['Pass_Rate'].values / 100),
    edgecolor='white', linewidth=1.2
)
for _, row in sleep_analysis.iterrows():
    axes[0].text(row['Sleep_Hours'], row['Pass_Rate']+1.5,
                 f"{row['Pass_Rate']:.0f}%\nn={row['Count']:.0f}",
                 ha='center', va='bottom', fontsize=8, fontweight='bold')
axes[0].axhline(df['Pass'].mean()*100, color='navy', ls='--', lw=1.3,
                label=f"Overall avg: {df['Pass'].mean()*100:.1f}%")
axes[0].set_title('Impact of Sleep Hours on Pass Rate', fontweight='bold')
axes[0].set_xlabel('Sleep Hours per Day')
axes[0].set_ylabel('Pass Rate (%)')
axes[0].set_ylim(0, 95)
axes[0].legend()

coaching_labels   = {0: 'No Extra\nCoaching', 1: 'Extra\nCoaching'}
coaching_analysis = df.groupby('Extra_Coaching')['Pass'].agg(['mean','count']).reset_index()
bars_c = axes[1].bar(
    [coaching_labels[k] for k in coaching_analysis['Extra_Coaching']],
    coaching_analysis['mean'] * 100,
    color=['#fd79a8','#008080'], edgecolor='white', width=0.4
)
for bar, (_, row) in zip(bars_c, coaching_analysis.iterrows()):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                 f"{row['mean']*100:.1f}%\nn={row['count']:.0f}",
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[1].axhline(df['Pass'].mean()*100, color='navy', ls='--', lw=1.3,
                label=f"Overall avg: {df['Pass'].mean()*100:.1f}%")
axes[1].set_title('Impact of Extra Coaching on Pass Rate', fontweight='bold')
axes[1].set_ylabel('Pass Rate (%)')
axes[1].set_ylim(0, 95)
axes[1].legend()

plt.suptitle('Sleep Hours & Extra Coaching — Detailed Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
st.pyplot(fig)
plt.close()

no_c      = df[df['Extra_Coaching']==0]['Pass'].mean() * 100
yes_c     = df[df['Extra_Coaching']==1]['Pass'].mean() * 100
opt_sleep = sleep_analysis.loc[sleep_analysis['Pass_Rate'].idxmax(), 'Sleep_Hours']

col_sl, col_ec = st.columns(2)
with col_sl:
    st.write("**Sleep Hours — Key Finding:**")
    st.write(f"Optimal sleep for highest pass rate: **{int(opt_sleep)} hours/day**")
    for _, row in sleep_analysis.sort_values('Pass_Rate', ascending=False).head(3).iterrows():
        st.write(f"  • {int(row.Sleep_Hours)} hrs → {row.Pass_Rate:.1f}% pass rate (n={int(row.Count)})")
with col_ec:
    st.write("**Extra Coaching — Key Finding:**")
    st.write(f"Without extra coaching: **{no_c:.1f}%** pass rate")
    st.write(f"With extra coaching: **{yes_c:.1f}%** pass rate")
    st.write(f"Difference: **+{yes_c-no_c:.1f} percentage points**")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TASK 7 — RECOMMENDATIONS
# (notebook Cell — added in updated notebook, replaces: print recommendations)
# ─────────────────────────────────────────────────────────────────────────────
st.header("Recommendations")
st.markdown(
    "Five evidence-based actionable recommendations based on feature importance, "
    "pass rate analysis, and model insights:"
)

recommendations = [
    {
        "title":    "1. Enforce Minimum Attendance Policy (Target ≥ 75%)",
        "evidence": "Attendance is a top-3 most important feature. Students with < 55% attendance "
                    "have near-zero pass rates; those above 85% pass at dramatically higher rates.",
        "action":   "Implement automated alerts when attendance drops below 75%. Assign a faculty "
                    "mentor to any student with < 65% attendance for two consecutive weeks.",
        "kpi":      "Reduce students with < 65% attendance by 40% within one semester.",
    },
    {
        "title":    "2. Early Assignment Score Interventions",
        "evidence": "Assignment_Score is a high-importance, early-available signal. Students scoring "
                    "below 40 on assignments have dramatically lower pass rates.",
        "action":   "Flag students scoring below 50% on the first two assignments for mandatory "
                    "tutoring. Provide weekly feedback loops and supplementary practice.",
        "kpi":      "Raise average assignment score of flagged students by 10 points within 4 weeks.",
    },
    {
        "title":    "3. Promote Structured Study Habits (≥ 4 Hours / Day)",
        "evidence": "Study_Hours shows a near-linear positive relationship with pass rate. "
                    "Students studying ≥ 4 hours/day consistently outperform those who study less.",
        "action":   "Introduce structured study sessions in the curriculum. Provide digital study "
                    "planners and gamification-based study streak rewards on the EdTech platform.",
        "kpi":      "Increase median self-reported study hours to ≥ 4 hours/day within one month.",
    },
    {
        "title":    "4. Expand Extra Coaching Access for At-Risk Students",
        "evidence": f"Students with extra coaching pass at {yes_c:.1f}% vs {no_c:.1f}% without — "
                    f"a +{yes_c-no_c:.1f} pp improvement.",
        "action":   "Use the model's predicted failure probability (> 60% fail risk) to auto-enrol "
                    "students into a free coaching programme within 2 weeks of identification.",
        "kpi":      "Enrol ≥ 80% of model-flagged at-risk students in supplementary coaching each semester.",
    },
    {
        "title":    "5. Deploy the ML Early Warning System with Weekly Re-scoring",
        "evidence": f"Best model ({best_model_name}) achieves strong Recall — reliably identifying "
                    "failing students before the final examination.",
        "action":   "Deploy this Streamlit app for academic advisors. Re-score all students weekly "
                    "using the latest attendance and assignment data. Retrain each semester.",
        "kpi":      "Reduce the institution's Fail rate by 15% in the semester following deployment.",
    },
]

for rec in recommendations:
    with st.expander(f"{rec['title']}", expanded=True):
        col_e, col_a, col_k = st.columns(3)
        col_e.markdown(f"**Evidence**\n\n{rec['evidence']}")
        col_a.markdown(f"**Action**\n\n{rec['action']}")
        col_k.markdown(f"**KPI**\n\n{rec['kpi']}")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SAVE MODEL ARTIFACTS
# (notebook Cell 52 — replaces: files.download() → st.download_button)
# ─────────────────────────────────────────────────────────────────────────────
st.header("Save Model Artifacts")

artifacts = {
    'models': {
        'Logistic Regression': lr,
        'Decision Tree':       dt,
        'Random Forest':       rf,
        'SVM (RBF)':           svm_rbf,
        'SVM (Linear)':        svm_lin,
        'Gradient Boosting':   gb,
        'Voting Ensemble':     voting,
        'MLP / ANN':           mlp,
    },
    'scaler':        scaler,
    'label_encoder': le,
    'feature_names': list(X.columns),
    'metrics_df':    mdf,
    'rf_importance': pd.Series(rf.feature_importances_, index=X.columns),
    'gb_importance': pd.Series(gb.feature_importances_, index=X.columns),
}

buf = io.BytesIO()
pickle.dump(artifacts, buf)
buf.seek(0)

st.download_button(
    label="Download model_artifacts.pkl",
    data=buf,
    file_name="2025NS01076_MLAssignment2_ModelArtifacts.pkl",
    mime="application/octet-stream",
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# LIVE PREDICTION
# (notebook Cell 54 — replaces: for-loop print → interactive Streamlit widgets)
# ─────────────────────────────────────────────────────────────────────────────
st.header("Live Student Prediction")
st.markdown(
    "Adjust the inputs below to get an instant Pass/Fail prediction "
    "from all 8 trained models simultaneously."
)

col1, col2 = st.columns(2)
with col1:
    study_hours    = st.slider("Study Hours",      0,   10,  3)
    attendance     = st.slider("Attendance (%)",   38,  100, 55)
    previous_marks = st.slider("Previous Marks",   0,   100, 40)
    assignment_sc  = st.slider("Assignment Score", 0,   100, 35)
with col2:
    sleep_hours    = st.slider("Sleep Hours",      3,   10,  5)
    internet_usage = st.selectbox("Internet Usage", le.classes_)
    extra_coaching = st.selectbox("Extra Coaching", ["No", "Yes"])

if st.button("Predict", type="primary"):
    new_student = pd.DataFrame({
        'Study_Hours':      [study_hours],
        'Attendance':       [attendance],
        'Previous_Marks':   [previous_marks],
        'Assignment_Score': [assignment_sc],
        'Sleep_Hours':      [sleep_hours],
        'Internet_Usage':   [le.transform([internet_usage])[0]],
        'Extra_Coaching':   [1 if extra_coaching == "Yes" else 0],
    })
    new_sc = scaler.transform(new_student)

    model_list = [
        ('Logistic Regression', lr,      True),
        ('Decision Tree',       dt,      False),
        ('Random Forest',       rf,      False),
        ('MLP / ANN',           mlp,     True),
        ('SVM (RBF)',           svm_rbf, True),
        ('SVM (Linear)',        svm_lin, True),
        ('Gradient Boosting',   gb,      False),
        ('Voting Ensemble',     voting,  True),
    ]

    pred_results = []
    for name, model, use_sc in model_list:
        data  = new_sc if use_sc else new_student
        pred  = model.predict(data)[0]
        prob  = model.predict_proba(data)[0]
        pred_results.append({
            'Model':         name,
            'Prediction':    'PASS' if pred == 1 else 'FAIL',
            'Fail Prob (%)': f'{prob[0]*100:.1f}%',
            'Pass Prob (%)': f'{prob[1]*100:.1f}%',
        })

    pred_df = pd.DataFrame(pred_results).set_index('Model')
    st.dataframe(pred_df, use_container_width=True)

    pass_count = sum(1 for r in pred_results if 'PASS' in r['Prediction'])
    fail_count = len(pred_results) - pass_count
    if pass_count >= 6:
        st.success(f"**Consensus: PASS** — {pass_count}/8 models predict Pass.")
    elif fail_count >= 6:
        st.error(f"**Consensus: FAIL** — {fail_count}/8 models predict Fail. Consider intervention.")
    else:
        st.warning(f"**Mixed prediction** — {pass_count} Pass / {fail_count} Fail. Monitor closely.")
