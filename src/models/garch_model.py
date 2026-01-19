"""
GARCH(1,1) Model Implementation for FOREX Volatility Forecasting

This module implements GARCH (Generalized Autoregressive Conditional Heteroskedasticity)
models for modeling time-varying volatility in FOREX returns. GARCH captures volatility
clustering and mean reversion, fundamental properties of financial time series.

Mathematical Formulation:
    Return equation:    r_t = μ + ε_t
    Variance equation:  σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    
    Where:
        r_t: Log return at time t
        σ²_t: Conditional variance (volatility²) at time t
        ε_t: Innovation (shock) ~ N(0, σ²_t)
        ω, α, β: Model parameters (α + β < 1 for stationarity)

Key Features:
    - GARCH(1,1) baseline implementation using `arch` package
    - Alternative specifications: GARCH(2,1), EGARCH for robustness
    - Comprehensive model diagnostics (Ljung-Box, ARCH LM, normality tests)
    - Out-of-sample volatility forecasting
    - Model selection via AIC/BIC

Author: Research Team
Date: January 2026
License: MIT
"""

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import Normal, StudentsT
from scipy import stats
from typing import Dict, Tuple, Optional, List
import warnings
import pickle
from pathlib import Path

# Suppress arch package warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


class GARCHModel:
    """
    GARCH model wrapper with diagnostics and forecasting capabilities.
    
    This class provides a clean API for:
        1. Fitting GARCH models on training data
        2. Generating conditional volatility estimates
        3. Performing statistical diagnostics
        4. Comparing alternative specifications
        5. Saving/loading trained models
    
    Attributes:
        p (int): GARCH lag order
        q (int): ARCH lag order
        dist (str): Error distribution ('normal' or 't' for Student's t)
        model: Fitted arch model object
        results: Model estimation results
    """
    
    def __init__(self, p: int = 2, q: int = 1, dist: str = 'normal', 
                 mean_model: str = 'Constant'):
        """
        Initialize GARCH model specification.
        
        Args:
            p: GARCH lag order (default: 2 - optimal from ACF/PACF)
            q: ARCH lag order (default: 1 - optimal from ACF/PACF)
            dist: Error distribution ('normal' or 't')
            mean_model: Mean equation specification ('Constant', 'Zero', 'AR')
        """
        self.p = p
        self.q = q
        self.dist = dist
        self.mean_model = mean_model
        self.model = None
        self.results = None
        self.train_data = None
        
    def fit(self, returns: pd.Series, rescale: bool = True) -> None:
        """
        Fit GARCH model to training data.
        
        Args:
            returns: Time series of log returns (must be stationary)
            rescale: Whether to rescale returns (improves numerical stability)
        
        Notes:
            - Model is fit using Maximum Likelihood Estimation (MLE)
            - Returns should be demeaned or have zero mean
            - Rescaling by 100 helps with convergence
        """
        self.train_data = returns.copy()
        
        # Rescale returns for numerical stability (standard practice)
        if rescale:
            train_returns = returns * 100
        else:
            train_returns = returns
            
        # Specify GARCH model
        self.model = arch_model(
            train_returns,
            mean=self.mean_model,
            vol='GARCH',
            p=self.p,
            q=self.q,
            dist=self.dist,
            rescale=False  # We already rescaled manually
        )
        
        # Fit model using MLE
        self.results = self.model.fit(disp='off', show_warning=False)
        
        print(f"GARCH({self.p},{self.q}) Model Fitted Successfully")
        print(f"Log-Likelihood: {self.results.loglikelihood:.2f}")
        print(f"AIC: {self.results.aic:.2f}")
        print(f"BIC: {self.results.bic:.2f}")
        
    def get_conditional_volatility(self, rescale: bool = True) -> pd.Series:
        """
        Extract conditional volatility (σ_t) from fitted model.
        
        Args:
            rescale: If True, rescale back to original return scale
            
        Returns:
            Time series of conditional volatility estimates
            
        Notes:
            - Returns volatility (σ_t), not variance (σ²_t)
            - Aligned with training data timestamps
        """
        if self.results is None:
            raise ValueError("Model must be fitted first. Call fit() method.")
        
        # Extract conditional volatility (standard deviation)
        cond_vol = self.results.conditional_volatility
        
        # Rescale back if needed
        if rescale:
            cond_vol = cond_vol / 100
            
        return cond_vol
    
    def forecast_volatility(self, horizon: int = 1, method: str = 'analytic') -> pd.DataFrame:
        """
        Forecast volatility for future periods.
        
        Args:
            horizon: Number of periods ahead to forecast
            method: Forecasting method ('analytic' or 'simulation')
            
        Returns:
            DataFrame with forecasted volatilities and confidence intervals
            
        Notes:
            - Analytic method: Closed-form GARCH forecasting equations
            - Forecasts assume no new information (unconditional forecasts)
        """
        if self.results is None:
            raise ValueError("Model must be fitted first.")
        
        # Generate forecasts
        forecasts = self.results.forecast(horizon=horizon, method=method)
        
        # Extract variance forecasts and convert to volatility
        variance_forecast = forecasts.variance.iloc[-1]
        volatility_forecast = np.sqrt(variance_forecast) / 100  # Rescale
        
        return volatility_forecast
    
    def generate_insample_volatility(self, returns: pd.Series, 
                                     rescale: bool = True) -> pd.Series:
        """
        Generate conditional volatility for validation/test data.
        
        This uses a rolling one-step-ahead approach:
            - At each time t, use data up to t-1 to estimate σ_t
            - This prevents data leakage
        
        Args:
            returns: Validation or test return series
            rescale: Whether to rescale returns
            
        Returns:
            Conditional volatility series for given returns
            
        Notes:
            - This is computationally expensive for long series
            - For research purposes, ensures no look-ahead bias
        """
        if self.results is None:
            raise ValueError("Model must be fitted first.")
        
        if rescale:
            returns_scaled = returns * 100
        else:
            returns_scaled = returns
            
        # Get parameters from fitted model
        params = self.results.params
        
        # Use model to generate conditional volatility
        # This applies the fitted parameters to new data
        model_new = arch_model(
            returns_scaled,
            mean=self.mean_model,
            vol='GARCH',
            p=self.p,
            q=self.q,
            rescale=False
        )
        
        # Fix parameters to previously estimated values
        # This prevents refitting and maintains consistency
        fixed_results = model_new.fix(params)
        cond_vol = fixed_results.conditional_volatility
        
        if rescale:
            cond_vol = cond_vol / 100
            
        return cond_vol
    
    def diagnostic_tests(self) -> Dict[str, Dict]:
        """
        Perform comprehensive model diagnostic tests.
        
        Returns:
            Dictionary containing test statistics and p-values for:
                1. Ljung-Box test (serial correlation in standardized residuals)
                2. ARCH LM test (remaining ARCH effects)
                3. Jarque-Bera test (residual normality)
        
        Interpretation:
            - Ljung-Box: p > 0.05 → No serial correlation (GOOD)
            - ARCH LM: p > 0.05 → No remaining ARCH effects (GOOD)
            - Jarque-Bera: p > 0.05 → Residuals are normal (GOOD, but often violated)
        """
        if self.results is None:
            raise ValueError("Model must be fitted first.")
        
        # Standardized residuals: ε_t / σ_t
        std_resid = self.results.std_resid
        
        diagnostics = {}
        
        # 1. Ljung-Box Test on standardized residuals
        # H0: No autocorrelation in standardized residuals
        lb_stat, lb_pvalue = self._ljung_box_test(std_resid, lags=10)
        diagnostics['ljung_box'] = {
            'statistic': lb_stat,
            'p_value': lb_pvalue,
            'interpretation': 'PASS' if lb_pvalue > 0.05 else 'FAIL',
            'note': 'Tests for serial correlation in standardized residuals'
        }
        
        # 2. ARCH LM Test on standardized residuals
        # H0: No ARCH effects remaining
        lm_stat, lm_pvalue = self._arch_lm_test(std_resid, lags=10)
        diagnostics['arch_lm'] = {
            'statistic': lm_stat,
            'p_value': lm_pvalue,
            'interpretation': 'PASS' if lm_pvalue > 0.05 else 'FAIL',
            'note': 'Tests for remaining conditional heteroskedasticity'
        }
        
        # 3. Jarque-Bera Normality Test
        # H0: Residuals are normally distributed
        jb_stat, jb_pvalue = stats.jarque_bera(std_resid.dropna())
        diagnostics['jarque_bera'] = {
            'statistic': jb_stat,
            'p_value': jb_pvalue,
            'interpretation': 'PASS' if jb_pvalue > 0.05 else 'FAIL',
            'note': 'Tests for normality; often fails for financial returns (fat tails)'
        }
        
        return diagnostics
    
    def _ljung_box_test(self, residuals: pd.Series, lags: int = 10) -> Tuple[float, float]:
        """
        Ljung-Box test for autocorrelation in residuals.
        
        Args:
            residuals: Standardized residuals
            lags: Number of lags to test
            
        Returns:
            Test statistic and p-value
        """
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        result = acorr_ljungbox(residuals.dropna(), lags=lags, return_df=True)
        # Return the test statistic and p-value for the specified lag
        return result.iloc[-1]['lb_stat'], result.iloc[-1]['lb_pvalue']
    
    def _arch_lm_test(self, residuals: pd.Series, lags: int = 10) -> Tuple[float, float]:
        """
        ARCH LM test for remaining ARCH effects.
        
        Args:
            residuals: Standardized residuals
            lags: Number of lags to test
            
        Returns:
            Test statistic and p-value
        """
        from statsmodels.stats.diagnostic import het_arch
        
        # Test on squared residuals
        lm_stat, lm_pvalue, _, _ = het_arch(residuals.dropna(), nlags=lags)
        return lm_stat, lm_pvalue
    
    def summary(self) -> str:
        """Print model summary with parameter estimates."""
        if self.results is None:
            raise ValueError("Model must be fitted first.")
        
        return self.results.summary()
    
    def save_model(self, filepath: Path) -> None:
        """Save fitted model to disk."""
        if self.results is None:
            raise ValueError("Model must be fitted first.")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'results': self.results,
                'p': self.p,
                'q': self.q,
                'dist': self.dist,
                'mean_model': self.mean_model
            }, f)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: Path):
        """Load fitted model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(p=data['p'], q=data['q'], dist=data['dist'], 
                   mean_model=data['mean_model'])
        model.results = data['results']
        return model


def compare_garch_models(returns: pd.Series, 
                        specifications: List[Dict]) -> pd.DataFrame:
    """
    Compare multiple GARCH specifications using AIC/BIC.
    
    Args:
        returns: Training return series
        specifications: List of model specs, e.g., [{'p': 1, 'q': 1}, {'p': 2, 'q': 1}]
        
    Returns:
        DataFrame comparing models by AIC, BIC, and log-likelihood
        
    Example:
        >>> specs = [
        ...     {'p': 1, 'q': 1, 'dist': 'normal'},
        ...     {'p': 2, 'q': 1, 'dist': 'normal'},
        ...     {'p': 1, 'q': 1, 'dist': 't'}
        ... ]
        >>> comparison = compare_garch_models(train_returns, specs)
    """
    results_list = []
    
    for spec in specifications:
        model = GARCHModel(**spec)
        model.fit(returns)
        
        results_list.append({
            'Specification': f"GARCH({spec['p']},{spec['q']}) - {spec.get('dist', 'normal')}",
            'Log-Likelihood': model.results.loglikelihood,
            'AIC': model.results.aic,
            'BIC': model.results.bic,
            'Num_Params': model.results.num_params
        })
    
    comparison_df = pd.DataFrame(results_list)
    
    # Sort by AIC (lower is better)
    comparison_df = comparison_df.sort_values('AIC')
    
    print("\nModel Comparison (sorted by AIC):")
    print("=" * 70)
    print(comparison_df.to_string(index=False))
    print("\nBest Model (lowest AIC):", comparison_df.iloc[0]['Specification'])
    
    return comparison_df


if __name__ == "__main__":
    # Example usage and testing
    print("GARCH Model Module - Example Usage\n")
    print("=" * 70)
    
    # Generate synthetic GARCH data for testing
    np.random.seed(42)
    n = 1000
    omega, alpha, beta = 0.01, 0.1, 0.85
    
    # Simulate GARCH(1,1) process
    returns = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)
    
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        returns[t] = np.sqrt(sigma2[t]) * np.random.normal()
    
    # Convert to pandas Series
    returns_series = pd.Series(returns, index=pd.date_range('2020-01-01', periods=n))
    
    print("Fitting GARCH(1,1) model to synthetic data...")
    model = GARCHModel(p=1, q=1)
    model.fit(returns_series)
    
    print("\n" + "=" * 70)
    print("Conditional Volatility Statistics:")
    cond_vol = model.get_conditional_volatility()
    print(cond_vol.describe())
    
    print("\n" + "=" * 70)
    print("Diagnostic Tests:")
    diagnostics = model.diagnostic_tests()
    for test_name, test_results in diagnostics.items():
        print(f"\n{test_name.upper()}:")
        print(f"  Statistic: {test_results['statistic']:.4f}")
        print(f"  P-value: {test_results['p_value']:.4f}")
        print(f"  Result: {test_results['interpretation']}")
        print(f"  Note: {test_results['note']}")

