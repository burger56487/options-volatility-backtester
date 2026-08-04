from src.pricing.black_scholes import option_price
from src.pricing.implied_volatility import implied_volatility


def main() -> None:
    spot = 100.0
    strike = 105.0
    time_to_expiry = 0.75
    risk_free_rate = 0.03
    dividend_yield = 0.01
    true_volatility = 0.28

    print("=" * 48)
    print("IMPLIED VOLATILITY RECOVERY DEMO")
    print("=" * 48)
    print()

    for option_type in ["call", "put"]:
        market_price = option_price(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=true_volatility,
            option_type=option_type,  # type: ignore
            dividend_yield=dividend_yield,
        )

        recovered_newton = implied_volatility(
            market_price=market_price,
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            option_type=option_type,  # type: ignore
            dividend_yield=dividend_yield,
            method="newton",
        )

        recovered_bisection = implied_volatility(
            market_price=market_price,
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            option_type=option_type,  # type: ignore
            dividend_yield=dividend_yield,
            method="bisection",
        )

        print(f"{option_type.upper()} OPTION")
        print(f"Market price:         {market_price:.6f}")
        print(f"True volatility:      {true_volatility:.2%}")
        print(f"Newton IV:            {recovered_newton:.6%}")
        print(f"Bisection IV:         {recovered_bisection:.6%}")
        print()


if __name__ == "__main__":
    main()
