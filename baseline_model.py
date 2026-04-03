# ============================================================
# FILE: baseline_models.py
# ============================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

def get_baseline_models():
    """
    Returns paper-faithful model configurations wrapped in pipelines.
    Hyperparameters are set to reasonable defaults matching the paper's
    reported performance; tune further via GridSearchCV (see Part 3).
    """

    # ── Random Forest ──────────────────────────────────────────
    # Paper reports ~97% accuracy. RF with 100 trees is standard.
    rf = Pipeline([
        ('scaler', MinMaxScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=100,       # number of trees
            max_depth=None,         # grow full trees (paper likely used default)
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',    # sqrt(31) ≈ 5 features per split
            bootstrap=True,
            class_weight=None,      # balanced classes in UCI dataset
            random_state=42,
            n_jobs=-1
        ))
    ])

    # ── Support Vector Machine ─────────────────────────────────
    # RBF kernel is standard for tabular classification.
    svm = Pipeline([
        ('scaler', MinMaxScaler()),
        ('clf', SVC(
            kernel='rbf',
            C=1.0,                  # regularization — tune if needed
            gamma='scale',          # 1 / (n_features * X.var())
            probability=True,       # needed for stacking later
            random_state=42
        ))
    ])

    # ── Decision Tree ──────────────────────────────────────────
    dt = Pipeline([
        ('scaler', MinMaxScaler()),
        ('clf', DecisionTreeClassifier(
            criterion='gini',
            max_depth=None,         # full tree; prune via ccp_alpha if overfitting
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        ))
    ])

    # ── Artificial Neural Network (MLP) ────────────────────────
    # Paper likely used a shallow network. Two hidden layers is standard.
    ann = Pipeline([
        ('scaler', MinMaxScaler()),
        ('clf', MLPClassifier(
            hidden_layer_sizes=(64, 32),  # two hidden layers
            activation='relu',
            solver='adam',
            alpha=0.0001,           # L2 regularization
            batch_size='auto',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,    # prevent overfitting
            validation_fraction=0.1,
            random_state=42
        ))
    ])

    return {
        'Random Forest': rf,
        'SVM': svm,
        'Decision Tree': dt,
        'ANN': ann
    }