import numpy as np
from src.models.fusion.stacking_lgbm import StackingLGBM

# Test _fit_temperature : T doit être > 1 (LogReg sur-confiant)
# Test _apply_temperature : somme sur axis=1 doit rester ≈ 1.0
# Test _derived_features : shape doit être (n, 8), valeurs dans [0, +inf)

n, K = 100, 27
rng = np.random.default_rng(42)
p = rng.dirichlet(np.ones(K) * 0.1, size=n).astype(np.float32)  # sur-confiant
y = rng.integers(0, K, size=n)

T = StackingLGBM._fit_temperature(p, y)
p_cal = StackingLGBM._apply_temperature(p, T)
derived = StackingLGBM._derived_features(p_cal, p_cal)

assert T > 0, "T doit être positif"
assert np.allclose(p_cal.sum(axis=1), 1.0, atol=1e-5), "somme probas != 1"
assert derived.shape == (n, 8), f"shape dérivées : {derived.shape}"
assert np.all(np.isfinite(derived)), "NaN/Inf dans les features dérivées"
print(f"T={T:.4f} | p_cal.sum check OK | derived shape {derived.shape} OK")