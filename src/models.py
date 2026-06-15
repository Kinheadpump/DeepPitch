import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier

try:
    from lightgbm import LGBMClassifier
    _LGBM = True
except ImportError:
    _LGBM = False

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
        grid_search.fit(X_scaled, y)
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
            self.model.fit(X_scaled, y)
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
