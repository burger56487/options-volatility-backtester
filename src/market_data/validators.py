"""Validation rules for underlying bars and option quotes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .schemas import OptionQuote, OptionType, UnderlyingBar
from .spread_rules import (
    DEFAULT_MAX_RELATIVE_SPREAD,
    DEFAULT_MINIMUM_ABSOLUTE_SPREAD,
)


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass
class ValidationIssue:
    code: str
    message: str
    severity: Severity
    field: str | None = None
    value: Any = None


@dataclass
class ValidationResult:
    valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [
            issue
            for issue in self.issues
            if issue.severity == Severity.ERROR
        ]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [
            issue
            for issue in self.issues
            if issue.severity == Severity.WARNING
        ]


def _report(
    issues: list[ValidationIssue],
) -> ValidationResult:
    return ValidationResult(
        valid=not any(
            issue.severity == Severity.ERROR
            for issue in issues
        ),
        issues=issues,
    )


def validate_underlying_bar(
    bar: UnderlyingBar,
) -> ValidationResult:
    issues: list[ValidationIssue] = []

    numeric_fields = {
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "adjusted_close": bar.adjusted_close,
        "volume": bar.volume,
    }
    for field_name, value in numeric_fields.items():
        if not math.isfinite(value):
            issues.append(
                ValidationIssue(
                    code="NON_FINITE_VALUE",
                    message=f"{field_name} must be finite.",
                    severity=Severity.ERROR,
                    field=field_name,
                    value=value,
                )
            )

    for field_name in [
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
    ]:
        value = getattr(bar, field_name)
        if value <= 0:
            issues.append(
                ValidationIssue(
                    code="NON_POSITIVE_PRICE",
                    message=f"{field_name} must be positive.",
                    severity=Severity.ERROR,
                    field=field_name,
                    value=value,
                )
            )

    if bar.volume < 0:
        issues.append(
            ValidationIssue(
                code="NEGATIVE_VOLUME",
                message="Volume cannot be negative.",
                severity=Severity.ERROR,
                field="volume",
                value=bar.volume,
            )
        )

    if bar.high < max(bar.open, bar.close, bar.low):
        issues.append(
            ValidationIssue(
                code="INVALID_HIGH",
                message="High below open, close or low.",
                severity=Severity.ERROR,
                field="high",
                value=bar.high,
            )
        )

    if bar.low > min(bar.open, bar.close, bar.high):
        issues.append(
            ValidationIssue(
                code="INVALID_LOW",
                message="Low above open, close or high.",
                severity=Severity.ERROR,
                field="low",
                value=bar.low,
            )
        )

    return _report(issues)


def option_no_arbitrage_bounds(
    quote: OptionQuote,
) -> tuple[float, float]:
    """European option no-arbitrage bounds."""
    t = quote.maturity_days / 365.0
    discounted_spot = quote.spot * math.exp(
        -quote.dividend_yield * t
    )
    discounted_strike = quote.strike * math.exp(
        -quote.risk_free_rate * t
    )

    if quote.option_type == OptionType.CALL:
        lower = max(0.0, discounted_spot - discounted_strike)
        upper = discounted_spot
    else:
        lower = max(0.0, discounted_strike - discounted_spot)
        upper = discounted_strike
    return lower, upper


def validate_option_quote(
    quote: OptionQuote,
    max_relative_spread: float = DEFAULT_MAX_RELATIVE_SPREAD,
    minimum_absolute_spread: float = DEFAULT_MINIMUM_ABSOLUTE_SPREAD,
    arbitrage_tolerance: float = 1e-8,
) -> ValidationResult:
    issues: list[ValidationIssue] = []

    finite_fields = {
        "strike": quote.strike,
        "bid": quote.bid,
        "ask": quote.ask,
        "spot": quote.spot,
        "risk_free_rate": quote.risk_free_rate,
        "dividend_yield": quote.dividend_yield,
    }
    for field_name, value in finite_fields.items():
        if not math.isfinite(value):
            issues.append(
                ValidationIssue(
                    code="NON_FINITE_VALUE",
                    message=f"{field_name} must be finite.",
                    severity=Severity.ERROR,
                    field=field_name,
                    value=value,
                )
            )

    if quote.strike <= 0:
        issues.append(
            ValidationIssue(
                code="NON_POSITIVE_STRIKE",
                message="Strike must be positive.",
                severity=Severity.ERROR,
                field="strike",
                value=quote.strike,
            )
        )

    if quote.spot <= 0:
        issues.append(
            ValidationIssue(
                code="NON_POSITIVE_SPOT",
                message="Spot must be positive.",
                severity=Severity.ERROR,
                field="spot",
                value=quote.spot,
            )
        )

    if quote.bid < 0:
        issues.append(
            ValidationIssue(
                code="NEGATIVE_BID",
                message="Bid cannot be negative.",
                severity=Severity.ERROR,
                field="bid",
                value=quote.bid,
            )
        )

    if quote.ask < quote.bid:
        issues.append(
            ValidationIssue(
                code="CROSSED_MARKET",
                message="Ask below bid.",
                severity=Severity.ERROR,
                field="ask",
                value=quote.ask,
            )
        )

    if quote.maturity_days <= 0:
        issues.append(
            ValidationIssue(
                code="EXPIRED_OPTION",
                message="Option expired or expiry invalid.",
                severity=Severity.ERROR,
                field="expiry",
                value=quote.expiry.isoformat(),
            )
        )

    if quote.multiplier <= 0:
        issues.append(
            ValidationIssue(
                code="INVALID_MULTIPLIER",
                message="Multiplier must be positive.",
                severity=Severity.ERROR,
                field="multiplier",
                value=quote.multiplier,
            )
        )

    if (
        quote.mid > 0
        and quote.spread > minimum_absolute_spread
        and quote.relative_spread > max_relative_spread
    ):
        issues.append(
            ValidationIssue(
                code="WIDE_SPREAD",
                message="Relative spread too wide.",
                severity=Severity.WARNING,
                field="relative_spread",
                value=quote.relative_spread,
            )
        )

    if (
        quote.maturity_days > 0
        and quote.spot > 0
        and quote.strike > 0
    ):
        lower, upper = option_no_arbitrage_bounds(quote)
        if quote.ask < lower - arbitrage_tolerance:
            issues.append(
                ValidationIssue(
                    code="BELOW_LOWER_BOUND",
                    message="Ask below European lower bound.",
                    severity=Severity.ERROR,
                    field="ask",
                    value=quote.ask,
                )
            )
        if quote.bid > upper + arbitrage_tolerance:
            issues.append(
                ValidationIssue(
                    code="ABOVE_UPPER_BOUND",
                    message="Bid above European upper bound.",
                    severity=Severity.ERROR,
                    field="bid",
                    value=quote.bid,
                )
            )
        if quote.mid < lower - arbitrage_tolerance:
            issues.append(
                ValidationIssue(
                    code="MID_BELOW_LOWER_BOUND",
                    message="Mid below European lower bound.",
                    severity=Severity.ERROR,
                    field="mid",
                    value=quote.mid,
                )
            )
        if quote.mid > upper + arbitrage_tolerance:
            issues.append(
                ValidationIssue(
                    code="MID_ABOVE_UPPER_BOUND",
                    message="Mid above European upper bound.",
                    severity=Severity.ERROR,
                    field="mid",
                    value=quote.mid,
                )
            )

    return _report(issues)
