"""Auditable filing transforms for FusionFinance.

This package is a namespaced, safety-hardened port of selected MIT-licensed
pure transforms from filing-alpha-cleanroom 0.6.0.  It intentionally excludes
the archive's data downloader, model/training stack, and rejected strategy.
"""

from .filing_fusion import (
    FilingAvailabilityConfig,
    FittedFusionModel,
    add_filing_decay_features,
    fit_fusion_model,
    map_filings_to_available_sessions,
    point_in_time_filing_join,
    robust_cross_sectional_standardize,
)
from .filing_text import build_longitudinal_text_features, extract_text_statistics
from .metrics import (
    annualized_return,
    annualized_sharpe,
    equity_curve,
    information_coefficient,
    max_drawdown,
    summarize_daily_returns,
)
from .schemas import (
    validate_filing_features,
    validate_market_data,
    validate_predictions,
)
from .xbrl import DEFAULT_CONCEPT_MAP, build_xbrl_features

__all__ = [
    "DEFAULT_CONCEPT_MAP",
    "FilingAvailabilityConfig",
    "FittedFusionModel",
    "add_filing_decay_features",
    "annualized_return",
    "annualized_sharpe",
    "build_longitudinal_text_features",
    "build_xbrl_features",
    "equity_curve",
    "extract_text_statistics",
    "fit_fusion_model",
    "information_coefficient",
    "map_filings_to_available_sessions",
    "max_drawdown",
    "point_in_time_filing_join",
    "robust_cross_sectional_standardize",
    "summarize_daily_returns",
    "validate_filing_features",
    "validate_market_data",
    "validate_predictions",
]

