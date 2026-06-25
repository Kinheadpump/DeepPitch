import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier

try:
    from lightgbm import LGBMClassifier
    _LGBM = True
except ImportError:
    _LGBM = False


def _ordinal_match_score(y_true, y_pred):
    """Ordinal betting point system at the 1X2 level.

    Classes are ordered: 0=away_win, 1=draw, 2=home_win.

      +2  correct outcome (winner/draw identified)
       0  adjacent mistake  (e.g. home_win predicted as draw)
      -2  catastrophic mistake (home_win predicted as away_win, or vice versa)

    Catastrophic mistakes are penalised 2× harder than standard accuracy,
    pushing the model to prefer a cautious draw prediction over a confidently
    wrong direction when the evidence is weak.

    Return value is normalised to [-1, +1] so GridSearchCV can compare folds.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    diff = np.abs(y_t - y_p)
    points = np.where(diff == 0, 2, np.where(diff == 1, 0, -2))
    return float(points.sum()) / (2 * len(y_t))


_ordinal_scorer = make_scorer(_ordinal_match_score)

class MetaMachineLearningModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'elo_diff', 'elo_avg', 'poisson_diff', 'form_diff',
            'att_form_diff', 'def_form_diff',
            'att_diff', 'mid_diff', 'def_diff',
            'is_neutral', 'tournament_weight',
        ]
        self.best_params = {}

    def train(self, df_features: pd.DataFrame):
        df_sorted = df_features.sort_values('date').reset_index(drop=True)
        X = df_sorted[self.feature_names]
        y = df_sorted['outcome']
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=self.feature_names)

        # Exponential time-decay: 3-year half-life so recent World Cup games matter
        # more without dramatically discounting older data. Normalised to mean=1 so
        # the effective sample size is unchanged — only relative emphasis shifts.
        days_ago = (pd.Timestamp.today() - pd.to_datetime(df_sorted['date'])).dt.days.clip(lower=0)
        raw_w = np.exp(-np.log(2) * days_ago.values / (3 * 365))
        sample_weight = raw_w / raw_w.mean()

        tscv = TimeSeriesSplit(n_splits=5)

        if _LGBM:
            print("[Model] LightGBM verfügbar — verwende LGBMClassifier + Isotonic Calibration...")
            estimator = LGBMClassifier(
                random_state=42,
                class_weight='balanced',
                verbose=-1,
                n_jobs=-1,
            )
            param_grid = {
                'n_estimators': [200, 400],
                'learning_rate': [0.04, 0.08],
                'num_leaves': [15, 31],
                'min_child_samples': [20, 40],
                'subsample': [0.8],
                'colsample_bytree': [0.8],
                'reg_alpha': [0.0, 0.5],
                'reg_lambda': [0.0, 0.5],
            }
            # 2×2×2×2×1×1×2×2 = 64 candidates × 5 folds = 320 fits
        else:
            print("[Model] LightGBM nicht gefunden — Fallback auf GradientBoostingClassifier...")
            estimator = GradientBoostingClassifier(random_state=42)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.05, 0.08, 0.12],
                'min_samples_leaf': [10, 20],
                'subsample': [0.8, 1.0],
            }

        grid_search = GridSearchCV(
            estimator,
            param_grid,
            cv=tscv,
            scoring='accuracy',
            n_jobs=1,
            verbose=1,
            refit=True,
        )
        grid_search.fit(X_scaled, y, sample_weight=sample_weight)
        self.best_params = grid_search.best_params_
        print(f"[Model] Beste Parameter: {self.best_params}")
        print(f"[Model] Walk-Forward CV-Genauigkeit (unkalibriert): {grid_search.best_score_*100:.2f}%")

        if _LGBM:
            # Refit best params then wrap with isotonic calibration.
            # Calibration corrects overconfident raw probabilities so Kelly sizing is reliable.
            best_lgbm = LGBMClassifier(
                **self.best_params,
                random_state=42,
                class_weight='balanced',
                verbose=-1,
                n_jobs=-1,
            )
            self.model = CalibratedClassifierCV(best_lgbm, cv=tscv, method='isotonic')
            self.model.fit(X_scaled, y, sample_weight=sample_weight)
            print("[Model] Isotonic Calibration abgeschlossen.")
        else:
            self.model = grid_search.best_estimator_

        self.is_trained = True

    def predict_probabilities(
        self,
        elo_diff: float, elo_avg: float, poisson_diff: float, form_diff: float,
        att_form_diff: float, def_form_diff: float,
        att_diff: float, mid_diff: float, def_diff: float,
        is_neutral: int = 1, tournament_weight: float = 0.5,
    ) -> dict:
        if not self.is_trained:
            raise RuntimeError("Modell muss trainiert werden!")

        X_pred = pd.DataFrame(
            [[elo_diff, elo_avg, poisson_diff, form_diff,
              att_form_diff, def_form_diff,
              att_diff, mid_diff, def_diff,
              is_neutral, tournament_weight]],
            columns=self.feature_names,
        )
        X_pred_scaled = pd.DataFrame(self.scaler.transform(X_pred), columns=self.feature_names)
        probs = self.model.predict_proba(X_pred_scaled)[0]

        classes = self.model.classes_
        prob_dict = {0: 0.0, 1: 0.0, 2: 0.0}
        for c, p in zip(classes, probs):
            prob_dict[c] = p

        return {"away_win": prob_dict[0], "draw": prob_dict[1], "home_win": prob_dict[2]}

    def get_feature_importances(self) -> dict:
        if not self.is_trained:
            return {}
        # CalibratedClassifierCV wraps the base estimator(s) — extract importances from first calibrated classifier.
        try:
            estimators = getattr(self.model, 'calibrated_classifiers_', None)
            if estimators:
                base = estimators[0].estimator
            else:
                base = self.model
            importances = base.feature_importances_
            return dict(zip(self.feature_names, importances.tolist()))
        except AttributeError:
            return {}
