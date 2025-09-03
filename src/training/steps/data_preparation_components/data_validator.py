"""Data Validator Component
Handles validation and verification of data integrity, including column verification and calculation of missing features.
Extracted from step01_5_data_converter.py
"""
from typing import Any
import pandas as pd
from src.utils.logger import system_logger


class DataValidator:
    """Utility class for validating data integrity and calculating missing columns.
    
    This class provides functionality for:
    - Verifying required and optional columns
    - Checking calculation feasibility for missing columns
    - Calculating missing technical indicators and features
    """

    def __init__(self, logger=None):
        self.logger = logger or system_logger.getChild("DataValidator")

        # Define required columns for different data types
        self.required_klines_columns = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        self.required_aggtrades_columns = ["timestamp", "price", "quantity"]
        self.required_futures_columns = ["timestamp", "fundingRate"]

        # Define optional calculated columns
        self.optional_calculated_columns = {
            "price_returns": [
                "close_return",
                "open_return",
                "high_return",
                "low_return",
            ],
            "vwap": ["vwap", "vwap_return", "price_vwap_ratio", "price_vwap_deviation"],
            "volume_features": ["volume_return", "volume_ma", "volume_ratio"],
            "technical_indicators": ["sma_20", "ema_12", "rsi", "macd"],
        }

    def verify_missing_columns(
        self, df: pd.DataFrame, data_type: str = "unified"
    ) -> dict[str, Any]:
        """
        Verify which columns are missing from the dataframe.

        Args:
            df: DataFrame to check
            data_type: Type of data ("klines", "aggtrades", "futures", "unified")

        Returns:
            Dictionary with missing columns information
        """
        try:
            self.logger.info(f"🔍 Verifying missing columns for {data_type} data...")

            missing_info = {
                "data_type": data_type,
                "total_columns": len(df.columns),
                "existing_columns": list(df.columns),
                "missing_required": [],
                "missing_optional": {},
                "can_calculate": {},
                "verification_passed": True,
            }

            # Check required columns based on data type
            if data_type == "klines":
                required_columns = self.required_klines_columns
            elif data_type == "aggtrades":
                required_columns = self.required_aggtrades_columns
            elif data_type == "futures":
                required_columns = self.required_futures_columns
            else:  # unified
                required_columns = self.required_klines_columns  # Base requirement

            # Check for missing required columns
            missing_required = [
                col for col in required_columns if col not in df.columns
            ]
            missing_info["missing_required"] = missing_required

            if missing_required:
                missing_info["verification_passed"] = False
                self.logger.warning(f"⚠️ Missing required columns: {missing_required}")

            # Check for missing optional calculated columns
            for category, columns in self.optional_calculated_columns.items():
                missing_optional = [col for col in columns if col not in df.columns]
                missing_info["missing_optional"][category] = missing_optional

                # Check if we can calculate these columns
                can_calculate = self._check_calculation_feasibility(
                    df, category, missing_optional
                )
                missing_info["can_calculate"][category] = can_calculate

                if missing_optional:
                    self.logger.info(
                        f"📊 Missing {category} columns: {missing_optional}"
                    )
                    if can_calculate:
                        self.logger.info(f"   ✅ Can calculate: {can_calculate}")
                    else:
                        self.logger.warning(
                            f"   ❌ Cannot calculate: {[col for col in missing_optional if col not in can_calculate]}"
                        )

            self.logger.info(
                f"✅ Column verification completed. Verification passed: {missing_info['verification_passed']}"
            )
            return missing_info

        except Exception as e:
            self.logger.exception(f"❌ Error during column verification: {e}")
            return {
                "data_type": data_type,
                "verification_passed": False,
                "error": str(e),
            }

    def _check_calculation_feasibility(
        self, df: pd.DataFrame, category: str, missing_columns: list[str]
    ) -> list[str]:
        """
        Check which missing columns can be calculated based on available data.

        Args:
            df: DataFrame with available data
            category: Category of columns to check
            missing_columns: List of missing columns

        Returns:
            List of columns that can be calculated
        """
        can_calculate = []

        if category == "price_returns":
            # Check if we have price columns for returns calculation
            price_columns = ["close", "open", "high", "low"]
            available_prices = [col for col in price_columns if col in df.columns]

            for col in missing_columns:
                if col.endswith("_return"):
                    base_col = col.replace("_return", "")
                    if base_col in available_prices:
                        can_calculate.append(col)

        elif category == "vwap":
            # Check if we have required columns for VWAP calculation
            if "close" in df.columns and "volume" in df.columns:
                can_calculate.extend(
                    [
                        col
                        for col in missing_columns
                        if col
                        in [
                            "vwap",
                            "vwap_return",
                            "price_vwap_ratio",
                            "price_vwap_deviation",
                        ]
                    ]
                )

        elif category == "volume_features":
            # Check if we have volume column
            if "volume" in df.columns:
                can_calculate.extend(
                    [
                        col
                        for col in missing_columns
                        if col in ["volume_return", "volume_ma", "volume_ratio"]
                    ]
                )

        elif category == "technical_indicators":
            # Check if we have price column for technical indicators
            if "close" in df.columns:
                can_calculate.extend(
                    [
                        col
                        for col in missing_columns
                        if col in ["sma_20", "ema_12", "rsi", "macd"]
                    ]
                )

        return can_calculate

    def calculate_missing_columns(
        self, df: pd.DataFrame, missing_info: dict[str, Any]
    ) -> pd.DataFrame:
        """
        Calculate missing columns that can be computed.

        Args:
            df: DataFrame to enhance
            missing_info: Output from verify_missing_columns

        Returns:
            Enhanced DataFrame with calculated columns
        """
        try:
            self.logger.info("🔄 Calculating missing columns...")

            # Create a copy to avoid modifying original
            enhanced_df = df.copy()
            calculated_columns = []

            # Calculate price returns
            if "price_returns" in missing_info["can_calculate"]:
                calculated_returns = self._calculate_price_returns(
                    enhanced_df, missing_info["can_calculate"]["price_returns"]
                )
                enhanced_df = pd.concat([enhanced_df, calculated_returns], axis=1)
                calculated_columns.extend(calculated_returns.columns)

            # Calculate VWAP features
            if "vwap" in missing_info["can_calculate"]:
                calculated_vwap = self._calculate_vwap_features(
                    enhanced_df, missing_info["can_calculate"]["vwap"]
                )
                enhanced_df = pd.concat([enhanced_df, calculated_vwap], axis=1)
                calculated_columns.extend(calculated_vwap.columns)

            # Calculate volume features
            if "volume_features" in missing_info["can_calculate"]:
                calculated_volume = self._calculate_volume_features(
                    enhanced_df, missing_info["can_calculate"]["volume_features"]
                )
                enhanced_df = pd.concat([enhanced_df, calculated_volume], axis=1)
                calculated_columns.extend(calculated_volume.columns)

            # Calculate technical indicators
            if "technical_indicators" in missing_info["can_calculate"]:
                calculated_technical = self._calculate_technical_indicators(
                    enhanced_df, missing_info["can_calculate"]["technical_indicators"]
                )
                enhanced_df = pd.concat([enhanced_df, calculated_technical], axis=1)
                calculated_columns.extend(calculated_technical.columns)

            if calculated_columns:
                self.logger.info(
                    f"✅ Calculated {len(calculated_columns)} columns: {calculated_columns}"
                )
            else:
                self.logger.info("ℹ️ No columns were calculated")

            return enhanced_df

        except Exception as e:
            self.logger.exception(f"❌ Error calculating missing columns: {e}")
            return df

    def _calculate_price_returns(
        self, df: pd.DataFrame, missing_returns: list[str]
    ) -> pd.DataFrame:
        """Calculate price return columns."""
        calculated = pd.DataFrame(index=df.index)

        for col in missing_returns:
            if col.endswith("_return"):
                base_col = col.replace("_return", "")
                if base_col in df.columns:
                    calculated[col] = df[base_col].pct_change()

        return calculated

    def _calculate_vwap_features(
        self, df: pd.DataFrame, missing_vwap: list[str]
    ) -> pd.DataFrame:
        """Calculate VWAP-related features."""
        calculated = pd.DataFrame(index=df.index)

        # Calculate VWAP if needed
        if "vwap" in missing_vwap and "close" in df.columns and "volume" in df.columns:
            calculated["vwap"] = (df["close"] * df["volume"]).rolling(
                window=20
            ).sum() / df["volume"].rolling(window=20).sum()

        # Calculate VWAP return if needed
        if "vwap_return" in missing_vwap and "vwap" in calculated.columns:
            calculated["vwap_return"] = calculated["vwap"].pct_change()

        # Calculate price-VWAP ratio if needed
        if (
            "price_vwap_ratio" in missing_vwap
            and "vwap" in calculated.columns
            and "close" in df.columns
        ):
            calculated["price_vwap_ratio"] = df["close"] / calculated["vwap"]

        # Calculate price-VWAP deviation if needed
        if (
            "price_vwap_deviation" in missing_vwap
            and "vwap" in calculated.columns
            and "close" in df.columns
        ):
            calculated["price_vwap_deviation"] = (
                df["close"] - calculated["vwap"]
            ) / calculated["vwap"]

        return calculated

    def _calculate_volume_features(
        self, df: pd.DataFrame, missing_volume: list[str]
    ) -> pd.DataFrame:
        """Calculate volume-related features."""
        calculated = pd.DataFrame(index=df.index)

        if "volume_return" in missing_volume and "volume" in df.columns:
            calculated["volume_return"] = df["volume"].pct_change()

        if "volume_ma" in missing_volume and "volume" in df.columns:
            calculated["volume_ma"] = df["volume"].rolling(window=20).mean()

        if "volume_ratio" in missing_volume and "volume" in df.columns:
            calculated["volume_ratio"] = (
                df["volume"] / df["volume"].rolling(window=20).mean()
            )

        return calculated

    def _calculate_technical_indicators(
        self, df: pd.DataFrame, missing_technical: list[str]
    ) -> pd.DataFrame:
        """Calculate technical indicators."""
        calculated = pd.DataFrame(index=df.index)

        if "sma_20" in missing_technical and "close" in df.columns:
            calculated["sma_20"] = df["close"].rolling(window=20).mean()

        if "ema_12" in missing_technical and "close" in df.columns:
            calculated["ema_12"] = df["close"].ewm(span=12).mean()

        if "rsi" in missing_technical and "close" in df.columns:
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            calculated["rsi"] = 100 - (100 / (1 + rs))

        if "macd" in missing_technical and "close" in df.columns:
            ema_12 = df["close"].ewm(span=12).mean()
            ema_26 = df["close"].ewm(span=26).mean()
            calculated["macd"] = ema_12 - ema_26

        return calculated

    def validate_data_types(self, df: pd.DataFrame, schema: dict[str, str]) -> tuple[bool, list[str]]:
        """
        Validate that DataFrame columns match expected data types.
        
        Args:
            df: DataFrame to validate
            schema: Dictionary mapping column names to expected dtypes
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        for col, expected_dtype in schema.items():
            if col not in df.columns:
                continue
                
            actual_dtype = str(df[col].dtype)
            
            # Handle flexible dtype matching
            if expected_dtype == "float64" and actual_dtype.startswith("float"):
                continue
            elif expected_dtype == "int64" and actual_dtype.startswith("int"):
                continue
            elif expected_dtype == "string" and actual_dtype == "object":
                continue
            elif expected_dtype != actual_dtype:
                errors.append(
                    f"Column '{col}' has dtype '{actual_dtype}', expected '{expected_dtype}'"
                )
        
        return len(errors) == 0, errors

    def validate_data_ranges(self, df: pd.DataFrame, constraints: dict[str, dict]) -> tuple[bool, list[str]]:
        """
        Validate that DataFrame values fall within expected ranges.
        
        Args:
            df: DataFrame to validate
            constraints: Dictionary mapping column names to constraint dicts with 'min' and/or 'max'
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        for col, constraint in constraints.items():
            if col not in df.columns:
                continue
                
            if "min" in constraint:
                min_val = df[col].min()
                if pd.notna(min_val) and min_val < constraint["min"]:
                    errors.append(
                        f"Column '{col}' has minimum value {min_val}, "
                        f"expected >= {constraint['min']}"
                    )
                    
            if "max" in constraint:
                max_val = df[col].max()
                if pd.notna(max_val) and max_val > constraint["max"]:
                    errors.append(
                        f"Column '{col}' has maximum value {max_val}, "
                        f"expected <= {constraint['max']}"
                    )
        
        return len(errors) == 0, errors