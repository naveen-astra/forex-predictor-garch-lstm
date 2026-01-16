"""
Quick test script to verify GARCH model implementation.
This can be run to ensure the model works before opening the notebook.

Usage:
    python test_garch.py
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import numpy as np
import pandas as pd
from src.utils.config import set_random_seeds, RANDOM_SEED
from src.models.garch_model import GARCHModel

# Set random seed
set_random_seeds(RANDOM_SEED)

print("="*70)
print("GARCH MODEL IMPLEMENTATION TEST")
print("="*70)

# Generate synthetic GARCH data
print("\n1. Generating synthetic GARCH(1,1) data...")
n = 1000
omega, alpha, beta = 0.01, 0.1, 0.85

returns = np.zeros(n)
sigma2 = np.zeros(n)
sigma2[0] = omega / (1 - alpha - beta)

for t in range(1, n):
    sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
    returns[t] = np.sqrt(sigma2[t]) * np.random.normal()

returns_series = pd.Series(returns, index=pd.date_range('2020-01-01', periods=n))
print(f"   ✓ Generated {n} observations")

# Test 1: Model fitting
print("\n2. Testing model fitting...")
model = GARCHModel(p=1, q=1, dist='normal')
model.fit(returns_series)
print("   ✓ Model fitted successfully")

# Test 2: Parameter extraction
print("\n3. Testing parameter extraction...")
params = model.results.params
if 'omega' in params:
    print(f"   ✓ ω = {params['omega']:.6f}")
    print(f"   ✓ α = {params['alpha[1]']:.6f}")
    print(f"   ✓ β = {params['beta[1]']:.6f}")

# Test 3: Conditional volatility
print("\n4. Testing conditional volatility extraction...")
cond_vol = model.get_conditional_volatility()
print(f"   ✓ Mean volatility: {cond_vol.mean():.6f}")
print(f"   ✓ Volatility range: [{cond_vol.min():.6f}, {cond_vol.max():.6f}]")

# Test 4: Diagnostic tests
print("\n5. Testing diagnostic functions...")
diagnostics = model.diagnostic_tests()
print(f"   ✓ Ljung-Box: {diagnostics['ljung_box']['interpretation']}")
print(f"   ✓ ARCH LM: {diagnostics['arch_lm']['interpretation']}")
print(f"   ✓ Jarque-Bera: {diagnostics['jarque_bera']['interpretation']}")

# Test 5: Model comparison
print("\n6. Testing model comparison...")
from src.models.garch_model import compare_garch_models
specs = [
    {'p': 1, 'q': 1, 'dist': 'normal'},
    {'p': 2, 'q': 1, 'dist': 'normal'},
]
comparison = compare_garch_models(returns_series[:800], specs)
print("   ✓ Model comparison completed")

# Test 6: Out-of-sample volatility
print("\n7. Testing out-of-sample volatility generation...")
test_returns = returns_series[800:]
test_vol = model.generate_insample_volatility(test_returns)
print(f"   ✓ Generated volatility for {len(test_vol)} test observations")

print("\n" + "="*70)
print("ALL TESTS PASSED ✓")
print("="*70)
print("\nGARCH model implementation is ready for use!")
print("Proceed to notebooks/03_garch_modeling.ipynb for full analysis.")
