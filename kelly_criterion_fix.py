# Kelly Criterion Fix for Position Sizer
# This file contains the corrected implementation of the Kelly criterion


def calculate_correct_kelly_position_size(
    price_target_confidences: dict[str, float],
    adversarial_confidences: dict[str, float],
    kelly_multiplier: float=0.25,
    min_position_size: float=0.01,
    max_position_size: float=0.5,
) -> float:
    """
    Calculate position size using the CORRECT Kelly criterion formula.

    The Kelly criterion formula is: f=(bp - q) / b
    where:
    - b=odds received (1 for 1:1 odds)
    - p=probability of win
    - q = probability of loss

    Args:
        price_target_confidences: Dict of confidence scores for price targets
        adversarial_confidences: Dict of confidence scores for adverse scenarios
        kelly_multiplier: Conservative multiplier for Kelly fraction (0-1)
        min_position_size: Minimum position size
        max_position_size: Maximum position size

    Returns:
        float: Calculated position size within bounds
    """
    try:
        # Get average confidence for target levels (0.5% to 2.0%)
        target_levels = [0.5, 1.0, 1.5, 2.0]
        confidences: list[float] = []

        for level in target_levels:
            if price_target_confidences:
                closest_key = min(
                    price_target_confidences.keys(),
                    key=lambda x: abs(float(x.replace("%", "")) - level),
                )
                confidence = float(price_target_confidences.get(closest_key, 0.5))
            else:
                confidence = 0.5
            confidences.append(confidence)

        # Calculate average confidence (probability of win)
        avg_confidence = sum(confidences) / len(confidences)

        # Get average adverse risk (probability of loss)
        adverse_risks: list[float] = []
        for level in target_levels:
            if adversarial_confidences:
                closest_key = min(
                    adversarial_confidences.keys(),
                    key=lambda x: abs(float(x.replace("%", "")) - level),
                )
                risk = float(adversarial_confidences.get(closest_key, 0.3))
            else:
                risk = 0.3
            adverse_risks.append(risk)

        avg_adverse_risk = sum(adverse_risks) / len(adverse_risks)

        # CORRECT Kelly criterion: f=(bp - q) / b, here b=1 => f=p - q
        p = max(0.0, min(1.0, avg_confidence))
        q = max(0.0, min(1.0, avg_adverse_risk))

        # If p + q > 1, normalize to a valid simplex
        if p + q > 1.0:
            total = p + q
            p = p / total
            q = q / total

        # Kelly fraction
        kelly_fraction = max(0.0, p - q)

        # Neutralize tiny edges to avoid churn
        if abs(p - q) < 0.02:
            kelly_fraction = 0.0

        # Apply conservative multiplier
        kelly_position_size = kelly_fraction * float(kelly_multiplier)

        # Ensure within bounds
        return max(
            float(min_position_size), min(float(max_position_size), kelly_position_size),
        )

    except (ValueError, TypeError, KeyError) as e:
        print(f"Error calculating Kelly position size: {e}")
        return float(min_position_size)
    except ZeroDivisionError as e:
        print(f"Division by zero in Kelly calculation: {e}")
        return float(min_position_size)


def calculate_enhanced_kelly_position_size(
    price_target_confidences: dict[str, float],
    adversarial_confidences: dict[str, float],
    market_volatility: float=0.02,
    account_balance: float=1000.0,
    kelly_multiplier: float=0.25,
    min_position_size: float=0.01,
    max_position_size: float=0.5,
) -> dict[str, float]:
    """
    Enhanced Kelly criterion with additional risk factors.

    Args:
        price_target_confidences: Dict of confidence scores for price targets
        adversarial_confidences: Dict of confidence scores for adverse scenarios
        market_volatility: Current market volatility (0-1)
        account_balance: Current account balance
        kelly_multiplier: Conservative multiplier for Kelly fraction (0-1)
        min_position_size: Minimum position size
        max_position_size: Maximum position size

    Returns:
        dict: Enhanced position sizing analysis
    """
    try:
        # Calculate base Kelly position size (correct inputs)
        base_kelly_size = calculate_correct_kelly_position_size(
            price_target_confidences=price_target_confidences,
            adversarial_confidences=adversarial_confidences,
            kelly_multiplier=kelly_multiplier,
            min_position_size=min_position_size,
            max_position_size=max_position_size,
        )

        # Volatility adjustment
        # Higher volatility should reduce position size
        volatility_adjustment=max(0.1, 1.0 - (market_volatility * 2.0))

        # Account balance adjustment
        # Larger accounts can take slightly larger positions
        balance_adjustment=min(1.2, max(0.8, (account_balance / 10000.0) ** 0.1))

        # Apply adjustments
        adjusted_size = base_kelly_size * volatility_adjustment * balance_adjustment

        # Ensure within bounds
        final_size = max(
            float(min_position_size), min(float(max_position_size), float(adjusted_size)),
        )

        return {
            "base_kelly_size": base_kelly_size, "volatility_adjustment": volatility_adjustment,
            "balance_adjustment": balance_adjustment, "final_position_size": final_size,
            "market_volatility": market_volatility, "account_balance": account_balance,
        }

    except (ValueError, TypeError, KeyError) as e:
        print(f"Error calculating enhanced Kelly position size: {e}")
        return {
            "base_kelly_size": float(min_position_size), "volatility_adjustment": 1.0,
            "balance_adjustment": 1.0,
            "final_position_size": float(min_position_size), "market_volatility": market_volatility,
            "account_balance": account_balance,
        }
    except ZeroDivisionError as e:
        print(f"Division by zero in enhanced Kelly calculation: {e}")
        return {
            "base_kelly_size": float(min_position_size), "volatility_adjustment": 1.0,
            "balance_adjustment": 1.0,
            "final_position_size": float(min_position_size), "market_volatility": market_volatility,
            "account_balance": account_balance,
        }


# Example usage and testing
if __name__== "__main__":
    # Test data
    price_target_confidences = {
        "0.5%": 0.7,
        "1.0%": 0.65,
        "1.5%": 0.6,
        "2.0%": 0.55,
    }

    adversarial_confidences={
        "0.5%": 0.3,
        "1.0%": 0.35,
        "1.5%": 0.4,
        "2.0%": 0.45,
    }

    # Test basic Kelly calculation
    basic_size=calculate_correct_kelly_position_size(
        price_target_confidences = adversarial_confidences,
    )
    print(f"Basic Kelly position size: {basic_size:.4f}")

    # Test enhanced Kelly calculation
    enhanced_result=calculate_enhanced_kelly_position_size(
        price_target_confidences = adversarial_confidences,
        market_volatility=0.03,
        account_balance=5000.0,
    )
    print(f"Enhanced Kelly result: {enhanced_result}")
