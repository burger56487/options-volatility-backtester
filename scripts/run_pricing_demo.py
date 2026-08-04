from src.pricing.black_scholes import price_and_greeks


def print_option_result(option_type: str) -> None:
    result = price_and_greeks(
        spot=100.0,
        strike=100.0,
        time_to_expiry=1.0,
        risk_free_rate=0.05,
        volatility=0.20,
        option_type=option_type,  # type: ignore
        dividend_yield=0.0,
    )

    print(f"{option_type.upper()} OPTION")
    print("-" * 36)
    print(f"Price: {result.price:.4f}")
    print(f"Delta: {result.delta:.4f}")
    print(f"Gamma: {result.gamma:.6f}")
    print(f"Vega:  {result.vega:.4f}")
    print(f"Theta: {result.theta:.4f}")
    print(f"Rho:   {result.rho:.4f}")
    print()


def main() -> None:
    print("=" * 36)
    print("BLACK-SCHOLES-MERTON PRICING DEMO")
    print("=" * 36)
    print()

    print_option_result("call")
    print_option_result("put")


if __name__ == "__main__":
    main()
