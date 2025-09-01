"""
Example: How the Tactician uses the Kelly Criterion Multiplier

This example demonstrates how the Tactician integrates the Kelly criterion
multiplier into its position sizing logic.
"""

from kelly_criterion_formula import calculate_kelly_multiplier


class TacticianPositionSizer:
    """
    Example Tactician position sizer that uses Kelly criterion multiplier.
    """
    
    def __init__(self, account_balance: float = 10000.0):
        self.account_balance = account_balance
    
    def calculate_position_size(
        self,
        price_target_confidences: dict[str, float],
        adversarial_confidences: dict[str, float],
        base_position_size: float = 0.1,  # 10% of account as base
        max_position_size: float = 0.5,  # 50% of account maximum
    ) -> dict[str, float]:
        """
        Calculate position size using Kelly criterion multiplier.
        
        Args:
            price_target_confidences: Dict of confidence scores for price targets
            adversarial_confidences: Dict of confidence scores for adverse scenarios
            base_position_size: Base position size as fraction of account
            max_position_size: Maximum position size as fraction of account
        
        Returns:
            dict: Position sizing information
        """
        
        # Get Kelly multiplier from the Kelly criterion formula
        kelly_multiplier = calculate_kelly_multiplier(
            price_target_confidences=price_target_confidences,
            adversarial_confidences=adversarial_confidences,
            kelly_multiplier=0.25,  # Conservative multiplier
        )
        
        # Calculate position size using Kelly multiplier
        # The Kelly multiplier acts as a risk adjustment factor
        position_size_fraction = base_position_size * kelly_multiplier
        
        # Ensure position size is within bounds
        position_size_fraction = max(0.01, min(max_position_size, position_size_fraction))
        
        # Calculate actual position size in currency
        position_size_currency = self.account_balance * position_size_fraction
        
        return {
            "kelly_multiplier": kelly_multiplier,
            "position_size_fraction": position_size_fraction,
            "position_size_currency": position_size_currency,
            "account_balance": self.account_balance,
            "base_position_size": base_position_size,
            "max_position_size": max_position_size,
        }


# Example usage
if __name__ == "__main__":
    # Test data
    price_target_confidences = {
        "0.5%": 0.7,
        "1.0%": 0.65,
        "1.5%": 0.6,
        "2.0%": 0.55,
    }

    adversarial_confidences = {
        "0.5%": 0.3,
        "1.0%": 0.35,
        "1.5%": 0.4,
        "2.0%": 0.45,
    }

    # Create Tactician position sizer
    tactician = TacticianPositionSizer(account_balance=10000.0)
    
    # Calculate position size
    result = tactician.calculate_position_size(
        price_target_confidences=price_target_confidences,
        adversarial_confidences=adversarial_confidences,
    )
    
    print("Tactician Position Sizing Result:")
    print(f"Kelly Multiplier: {result['kelly_multiplier']:.4f}")
    print(f"Position Size Fraction: {result['position_size_fraction']:.4f}")
    print(f"Position Size Currency: ${result['position_size_currency']:.2f}")
    print(f"Account Balance: ${result['account_balance']:.2f}")
    print(f"Base Position Size: {result['base_position_size']:.2f}")
    print(f"Max Position Size: {result['max_position_size']:.2f}")