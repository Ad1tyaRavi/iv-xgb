import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

def chronological_split(df: pd.DataFrame, train_frac: float = 0.7):
    n = len(df)
    cut = int(n * train_frac)
    return df.iloc[:cut], df.iloc[cut:]

def train_baseline_logreg(trainX, trainy, testX, testy):
    # Create a pipeline that first imputes missing values, then scales the data
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])
    pipeline.fit(trainX, trainy)
    proba = pipeline.predict_proba(testX)[:,1]
    auc = roc_auc_score(testy, proba)
    # For compatibility with old return format, we can return the whole pipeline
    return {'model': pipeline, 'scaler': pipeline.named_steps['scaler'], 'auc': auc, 'proba': proba}

def train_xgb(trainX, trainy, testX, testy):
    # Create a pipeline that imputes, scales, and then classifies
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            tree_method='hist',
            scale_pos_weight=max(1.0, (len(trainy)-trainy.sum())/max(1.0,trainy.sum()))
        ))
    ])

    # Adjust params for the pipeline
    params = {
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__n_estimators': [100, 200],
        'classifier__subsample': [0.8, 0.9],
        'classifier__colsample_bytree': [0.8, 0.9],
    }
    
    tscv = TimeSeriesSplit(n_splits=3)
    gs = GridSearchCV(pipeline, params, scoring='roc_auc', cv=tscv, n_jobs=-1, verbose=0)
    gs.fit(trainX, trainy)
    best = gs.best_estimator_
    proba = best.predict_proba(testX)[:,1]
    auc = roc_auc_score(testy, proba)
    return {'model': best, 'auc': auc, 'proba': proba, 'cv_results': gs.cv_results_}

def plot_roc_curves(y_true, p1, p2, labels, outpath):
    fpr1, tpr1, _ = roc_curve(y_true, p1)
    fpr2, tpr2, _ = roc_curve(y_true, p2)
    plt.figure(figsize=(6,5))
    plt.plot(fpr1, tpr1, label=f"{labels[0]}")
    plt.plot(fpr2, tpr2, label=f"{labels[1]}")
    plt.plot([0,1],[0,1],'--',linewidth=1)
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.tight_layout()
    plt.savefig(outpath, dpi=140); plt.close()

def plot_confusions(y_true, p1, p2, outpath):
    thr = 0.5
    y1 = (p1>=thr).astype(int)
    y2 = (p2>=thr).astype(int)
    cms = [confusion_matrix(y_true, y1), confusion_matrix(y_true, y2)]
    titles = ['Baseline(LogReg)','XGBoost']
    fig, axes = plt.subplots(1,2, figsize=(8,4))
    for ax, cm, title in zip(axes, cms, titles):
        im = ax.imshow(cm, interpolation='nearest')
        ax.set_title(title)
        for (i,j), val in np.ndenumerate(cm):
            ax.text(j, i, int(val), ha='center', va='center')
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    fig.tight_layout()
    fig.savefig(outpath, dpi=140); plt.close(fig)

def feature_importance_xgb(model: Pipeline, columns: list[str]) -> pd.DataFrame:
    # Extract the XGBoost model from the pipeline
    xgb_model = model.named_steps['classifier']
    gain = xgb_model.get_booster().get_score(importance_type='gain')
    weight = xgb_model.get_booster().get_score(importance_type='weight')
    keys = set(gain.keys()) | set(weight.keys())
    rows = []
    for k in keys:
        idx = int(k.strip('f'))
        rows.append({
            'feature': columns[idx] if idx < len(columns) else k,
            'gain': gain.get(k, 0.0),
            'weight': weight.get(k, 0.0)
        })
    imp = pd.DataFrame(rows).sort_values('gain', ascending=False)
    return imp
