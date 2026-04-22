def fix():
    with open("extreme_price_movements/meta_model.py", "r") as f:
        content = f.read()

    # Regression
    old_reg = """        baseline_p = prev_params if prev_params else params
        _, _, baseline_metrics, _ = self._cv_evaluate(
            kind, baseline_p, X_hpo, y_hpo, sw_hpo
        )"""
    new_reg = """        baseline_p = prev_params if prev_params else params
        _, _, baseline_metrics, _ = self._cv_evaluate(
            kind, baseline_p, X_hpo, y_hpo, sw_hpo, n_splits=self.cv_splits
        )"""
    content = content.replace(old_reg, new_reg)

    # Regression score_params
    old_reg_score = """        def _score_params(_p, _X, _y, _sw):
            _, ic, metrics, _ = self._cv_evaluate(
                kind,
                _p,
                _X,
                _y,
                _sw,
            )"""
    new_reg_score = """        def _score_params(_p, _X, _y, _sw):
            _, ic, metrics, _ = self._cv_evaluate(
                kind,
                _p,
                _X,
                _y,
                _sw,
                n_splits=self.cv_splits,
            )"""
    content = content.replace(old_reg_score, new_reg_score)

    # Classification cv splits
    old_clf_cv = """        baseline_p = prev_params if prev_params else params
        _, _, baseline_metrics, _ = self._cv_evaluate(
            kind, baseline_p, X_hpo, y_hpo, sw_hpo, n_splits=self.cv_splits
        )"""
    new_clf_cv = """        baseline_p = prev_params if prev_params else params
        _, _, baseline_metrics, _ = self._cv_evaluate(
            kind, baseline_p, X_hpo, y_hpo, sw_hpo, n_splits=self.cv_splits
        )"""

    old_clf_cv2 = """        def _score_params(_p, _X, _y, _sw):
            _, _, metrics, _ = self._cv_evaluate(
                kind,
                _p,
                _X,
                _y,
                _sw,
                n_splits=self.cv_splits,
            )"""

    with open("extreme_price_movements/meta_model.py", "w") as f:
        f.write(content)

fix()
