"""Transparent filing-level XBRL ratio and change features.

Adapted from filing-alpha-cleanroom 0.6.0 under the MIT License.  See NOTICE.
The caller remains responsible for point-in-time context selection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_CONCEPT_MAP: dict[str, Sequence[str]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ),
    "gross_profit": ("GrossProfit",),
    "operating_income": ("OperatingIncomeLoss",),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "capex": (
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForAdditionsToPropertyPlantAndEquipment",
    ),
    "total_assets": ("Assets",),
    "current_assets": ("AssetsCurrent",),
    "current_liabilities": ("LiabilitiesCurrent",),
    "cash": (
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ),
    "debt_current": ("LongTermDebtCurrent", "ShortTermBorrowings", "DebtCurrent"),
    "debt_noncurrent": (
        "LongTermDebtNoncurrent",
        "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
    ),
    "equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    "receivables": (
        "AccountsReceivableNetCurrent",
        "AccountsNotesAndLoansReceivableNetCurrent",
    ),
    "inventory": ("InventoryNet",),
    "shares_outstanding": (
        "EntityCommonStockSharesOutstanding",
        "CommonStockSharesOutstanding",
    ),
    "share_repurchases": ("PaymentsForRepurchaseOfCommonStock",),
    "share_issuance": (
        "ProceedsFromStockOptionsExercised",
        "ProceedsFromIssuanceOfCommonStock",
    ),
}


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.where(denominator.abs() > 1e-12)


def _canonicalize_facts(
    facts: pd.DataFrame,
    concept_map: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    reverse_map: dict[str, tuple[str, int]] = {}
    for canonical, concepts in concept_map.items():
        for priority, concept in enumerate(concepts):
            reverse_map[concept] = (canonical, priority)
    frame = facts.copy(deep=True)
    mapped = frame["concept"].map(reverse_map)
    frame["canonical_concept"] = mapped.map(
        lambda value: value[0] if isinstance(value, tuple) else np.nan
    )
    frame["concept_priority"] = mapped.map(
        lambda value: value[1] if isinstance(value, tuple) else np.nan
    )
    frame = frame.loc[frame["canonical_concept"].notna()].copy()
    if frame.empty:
        return frame
    frame["concept_priority"] = frame["concept_priority"].astype(int)
    identifiers = ["ticker", "accepted_at", "form"]
    if "period_end" in frame.columns:
        identifiers.append("period_end")
    frame = frame.sort_values([*identifiers, "canonical_concept", "concept_priority"])
    return frame.drop_duplicates([*identifiers, "canonical_concept"], keep="first")


def build_xbrl_features(
    facts: pd.DataFrame,
    concept_map: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Return accounting ratios and sequential/comparable filing deltas.

    Required columns are ``ticker``, ``accepted_at``, ``form``, ``concept``,
    and ``value``.  The caller must first select the correct XBRL context and
    period; this transform only resolves aliases and derives features.
    """
    required = {"ticker", "accepted_at", "form", "concept", "value"}
    missing = sorted(required.difference(facts.columns))
    if missing:
        raise ValueError(f"XBRL facts are missing required columns: {missing}")
    frame = facts.copy(deep=True)
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["accepted_at"] = pd.to_datetime(frame["accepted_at"], errors="raise", utc=True)
    frame["form"] = frame["form"].astype(str).str.upper().str.strip()
    frame["concept"] = frame["concept"].astype(str).str.strip()
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.loc[np.isfinite(frame["value"])].copy()
    if "period_end" in frame.columns:
        frame["period_end"] = pd.to_datetime(frame["period_end"], errors="coerce").dt.date

    mapped = _canonicalize_facts(frame, concept_map or DEFAULT_CONCEPT_MAP)
    identifiers = ["ticker", "accepted_at", "form"]
    if "period_end" in mapped.columns:
        identifiers.append("period_end")
    if mapped.empty:
        return pd.DataFrame(columns=identifiers)
    wide = mapped.pivot(
        index=identifiers,
        columns="canonical_concept",
        values="value",
    ).reset_index()
    wide.columns.name = None

    def ensure(column: str) -> pd.Series:
        if column not in wide.columns:
            wide[column] = np.nan
        return wide[column]

    revenue = ensure("revenue")
    gross_profit = ensure("gross_profit")
    operating_income = ensure("operating_income")
    net_income = ensure("net_income")
    operating_cash_flow = ensure("operating_cash_flow")
    capex = ensure("capex").abs()
    assets = ensure("total_assets")
    current_assets = ensure("current_assets")
    current_liabilities = ensure("current_liabilities")
    cash = ensure("cash")
    debt_current = ensure("debt_current")
    debt_noncurrent = ensure("debt_noncurrent")
    receivables = ensure("receivables")
    inventory = ensure("inventory")

    wide["free_cash_flow"] = operating_cash_flow - capex
    # Preserve unknown debt instead of fabricating a zero when both concepts
    # are absent.  One reported component remains useful via min_count=1.
    wide["total_debt"] = pd.concat([debt_current, debt_noncurrent], axis=1).sum(
        axis=1,
        min_count=1,
    )
    wide["net_debt"] = wide["total_debt"] - cash
    wide["working_capital"] = current_assets - current_liabilities
    wide["gross_margin"] = _safe_divide(gross_profit, revenue)
    wide["operating_margin"] = _safe_divide(operating_income, revenue)
    wide["net_margin"] = _safe_divide(net_income, revenue)
    wide["operating_cash_flow_margin"] = _safe_divide(operating_cash_flow, revenue)
    wide["free_cash_flow_margin"] = _safe_divide(wide["free_cash_flow"], revenue)
    wide["current_ratio"] = _safe_divide(current_assets, current_liabilities)
    wide["debt_to_assets"] = _safe_divide(wide["total_debt"], assets)
    wide["net_debt_to_operating_income"] = _safe_divide(
        wide["net_debt"],
        operating_income.abs(),
    )
    wide["cash_to_assets"] = _safe_divide(cash, assets)
    wide["accruals_to_assets"] = _safe_divide(net_income - operating_cash_flow, assets)
    wide["receivables_to_revenue"] = _safe_divide(receivables, revenue)
    wide["inventory_to_revenue"] = _safe_divide(inventory, revenue)

    numeric_columns = [column for column in wide.columns if column not in identifiers]
    wide = wide.sort_values(["ticker", "form", "accepted_at"]).reset_index(drop=True)
    group_keys = ["ticker", "form"]
    comparable_lag = pd.Series(
        np.where(wide["form"].str.startswith("10-Q"), 4, 1),
        index=wide.index,
    )
    change_features: dict[str, pd.Series] = {}
    for column in numeric_columns:
        prior = wide.groupby(group_keys, sort=False)[column].shift(1)
        sequential_delta = wide[column] - prior
        sequential_percent = sequential_delta / prior.abs().where(prior.abs() > 1e-12)
        prior_change = sequential_percent.groupby(
            [wide[key] for key in group_keys],
            sort=False,
        ).shift(1)
        change_features[f"seq_delta__{column}"] = sequential_delta
        change_features[f"seq_pct__{column}"] = sequential_percent
        change_features[f"seq_accel__{column}"] = sequential_percent - prior_change

        comparable_prior = pd.Series(np.nan, index=wide.index, dtype=float)
        for lag_value in sorted(comparable_lag.unique()):
            lag = int(lag_value)
            shifted = wide.groupby(group_keys, sort=False)[column].shift(lag)
            comparable_prior.loc[comparable_lag == lag] = shifted.loc[comparable_lag == lag]
        comparable_delta = wide[column] - comparable_prior
        change_features[f"comparable_delta__{column}"] = comparable_delta
        change_features[f"comparable_pct__{column}"] = comparable_delta / comparable_prior.abs().where(
            comparable_prior.abs() > 1e-12
        )

    wide = pd.concat([wide, pd.DataFrame(change_features, index=wide.index)], axis=1)
    return wide.sort_values(["ticker", "accepted_at"]).reset_index(drop=True)
