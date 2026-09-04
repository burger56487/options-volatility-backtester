"""Same-accuracy timing benchmark across pricing engines."""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.domain.enums import OptionType
from src.pricing.black_scholes import option_price
from src.pricing.registry import price
from src.pricing.requests import PricingRequest


def main() -> None:
    request = PricingRequest(
        spot=100.0,
        strike=100.0,
        time_to_expiry=0.5,
        risk_free_rate=0.04,
        dividend_yield=0.01,
        volatility=0.25,
        option_type=OptionType.CALL,
        steps=400,
    )
    reference = option_price(
        spot=100.0,
        strike=100.0,
        time_to_expiry=0.5,
        risk_free_rate=0.04,
        volatility=0.25,
        option_type="call",
        dividend_yield=0.01,
    )
    results = []
    for method in ("black_scholes", "crr", "crank_nicolson"):
        start = time.perf_counter()
        result = price(request, method)
        elapsed = time.perf_counter() - start
        results.append(
            {
                "method": method,
                "elapsed_seconds": round(elapsed, 6),
                "absolute_error": round(
                    abs(result.price - reference), 6
                ),
            }
        )
    for row in results:
        print(row)


if __name__ == "__main__":
    main()
