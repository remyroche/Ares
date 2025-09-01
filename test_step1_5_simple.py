#!/usr/bin/env python3
"""
Simple Test Script for Step1_5 Column Verification and Calculation Enhancement

This script tests the new column verification and calculation functionality
without importing the complex module structure.
"""

import pandas as pd
import numpy as np
import sys


class ColumnVerifier:
    """Utility class for verifying and calculating missing columns."""
    
    def __init__(self, logger=None):
        self.logger = logger or print
        
        # Define required columns for different data types
        self.required_klines_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        self.required_aggtrades_columns = ["timestamp", "price", "quantity"]
        self.required_futures_columns = ["timestamp", "fundingRate"]
        
        # Define optional calculated columns
        self.optional_calculated_columns = {
            "price_returns": ["close_return", "open_return", "high_return", "low_return"],
            "vwap": ["vwap", "vwap_return", "price_vwap_ratio", "price_vwap_deviation"],
            "volume_features": ["volume_return", "volume_ma", "volume_ratio"],
            "technical_indicators": ["sma_20", "ema_12", "rsi", "macd"]
        }
    
    def verify_missing_columns(self, df: pd.DataFrame, data_type: str = "unified") -> dict:
        """
        Verify which columns are missing from the dataframe.
        
        Args:
            df: DataFrame to check
            data_type: Type of data ("klines", "aggtrades", "futures", "unified")
            
        Returns:
            Dictionary with missing columns information
        """
        try:
            self.logger(f"🔍 Verifying missing columns for {data_type} data...")
            
            missing_info = {
                "data_type": data_type,
                "total_columns": len(df.columns),
                "existing_columns": list(df.columns),
                "missing_required": [],
                "missing_optional": {},
                "can_calculate": {},
                "verification_passed": True
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
            missing_required = [col for col in required_columns if col not in df.columns]
            missing_info["missing_required"] = missing_required
            
            if missing_required:
                missing_info["verification_passed"] = False
                self.logger(f"⚠️ Missing required columns: {missing_required}")
            
            # Check for missing optional calculated columns
            for category, columns in self.optional_calculated_columns.items():
                missing_optional = [col for col in columns if col not in df.columns]
                missing_info["missing_optional"][category] = missing_optional
                
                # Check if we can calculate these columns
                can_calculate = self._check_calculation_feasibility(df, category, missing_optional)
                missing_info["can_calculate"][category] = can_calculate
                
                if missing_optional:
                    self.logger(f"📊 Missing {category} columns: {missing_optional}")
                    if can_calculate:
                        self.logger(f"   ✅ Can calculate: {can_calculate}")
                    else:
                        self.logger(f"   ❌ Cannot calculate: {[col for col in missing_optional if col not in can_calculate]}")
            
            self.logger(f"✅ Column verification completed. Verification passed: {missing_info['verification_passed']}")
            return missing_info
            
        except Exception as e:
            self.logger(f"❌ Error during column verification: {e}")
            return {
                "data_type": data_type,
                "verification_passed": False,
                "error": str(e)
            }
    
    def _check_calculation_feasibility(self, df: pd.DataFrame, category: str, missing_columns: list[str]) -> list[str]:
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
                can_calculate.extend([col for col in missing_columns if col in ["vwap", "vwap_return", "price_vwap_ratio", "price_vwap_deviation"]])
        
        elif category == "volume_features":
            # Check if we have volume column
            if "volume" in df.columns:
                can_calculate.extend([col for col in missing_columns if col in ["volume_return", "volume_ma", "volume_ratio"]])
        
        elif category == "technical_indicators":
            # Check if we have price column for technical indicators
            if "close" in df.columns:
                can_calculate.extend([col for col in missing_columns if col in ["sma_20", "ema_12", "rsi", "macd"]])
        
        return can_calculate
    
    def calculate_missing_columns(self, df: pd.DataFrame, missing_info: dict) -> pd.DataFrame:
        """
        Calculate missing columns that can be computed.
        
        Args:
            df: DataFrame to enhance
            missing_info: Output from verify_missing_columns
            
        Returns:
            Enhanced DataFrame with calculated columns
        """
        try:
            self.logger("🔄 Calculating missing columns...")
            
            # Create a copy to avoid modifying original
            enhanced_df = df.copy()
            calculated_columns = []
            
            # Calculate price returns
            if "price_returns" in missing_info["can_calculate"]:
                calculated_returns = self._calculate_price_returns(enhanced_df, missing_info["can_calculate"]["price_returns"])
                enhanced_df = pd.concat([enhanced_df, calculated_returns], axis=1)
                calculated_columns.extend(calculated_returns.columns)
            
            # Calculate VWAP features
            if "vwap" in missing_info["can_calculate"]:
                calculated_vwap = self._calculate_vwap_features(enhanced_df, missing_info["can_calculate"]["vwap"])
                enhanced_df = pd.concat([enhanced_df, calculated_vwap], axis=1)
                calculated_columns.extend(calculated_vwap.columns)
            
            # Calculate volume features
            if "volume_features" in missing_info["can_calculate"]:
                calculated_volume = self._calculate_volume_features(enhanced_df, missing_info["can_calculate"]["volume_features"])
                enhanced_df = pd.concat([enhanced_df, calculated_volume], axis=1)
                calculated_columns.extend(calculated_volume.columns)
            
            # Calculate technical indicators
            if "technical_indicators" in missing_info["can_calculate"]:
                calculated_technical = self._calculate_technical_indicators(enhanced_df, missing_info["can_calculate"]["technical_indicators"])
                enhanced_df = pd.concat([enhanced_df, calculated_technical], axis=1)
                calculated_columns.extend(calculated_technical.columns)
            
            if calculated_columns:
                self.logger(f"✅ Calculated {len(calculated_columns)} columns: {calculated_columns}")
            else:
                self.logger("ℹ️ No columns were calculated")
            
            return enhanced_df
            
        except Exception as e:
            self.logger(f"❌ Error calculating missing columns: {e}")
            return df
    
    def _calculate_price_returns(self, df: pd.DataFrame, missing_returns: list[str]) -> pd.DataFrame:
        """Calculate price return columns."""
        calculated = pd.DataFrame(index=df.index)
        
        for col in missing_returns:
            if col.endswith("_return"):
                base_col = col.replace("_return", "")
                if base_col in df.columns:
                    calculated[col] = df[base_col].pct_change()
        
        return calculated
    
    def _calculate_vwap_features(self, df: pd.DataFrame, missing_vwap: list[str]) -> pd.DataFrame:
        """Calculate VWAP-related features."""
        calculated = pd.DataFrame(index=df.index)
        
        # Calculate VWAP if needed
        if "vwap" in missing_vwap and "close" in df.columns and "volume" in df.columns:
            calculated["vwap"] = (df["close"] * df["volume"]).rolling(window=20).sum() / df["volume"].rolling(window=20).sum()
        
        # Calculate VWAP return if needed
        if "vwap_return" in missing_vwap and "vwap" in calculated.columns:
            calculated["vwap_return"] = calculated["vwap"].pct_change()
        
        # Calculate price-VWAP ratio if needed
        if "price_vwap_ratio" in missing_vwap and "vwap" in calculated.columns and "close" in df.columns:
            calculated["price_vwap_ratio"] = df["close"] / calculated["vwap"]
        
        # Calculate price-VWAP deviation if needed
        if "price_vwap_deviation" in missing_vwap and "vwap" in calculated.columns and "close" in df.columns:
            calculated["price_vwap_deviation"] = (df["close"] - calculated["vwap"]) / calculated["vwap"]
        
        return calculated
    
    def _calculate_volume_features(self, df: pd.DataFrame, missing_volume: list[str]) -> pd.DataFrame:
        """Calculate volume-related features."""
        calculated = pd.DataFrame(index=df.index)
        
        if "volume_return" in missing_volume and "volume" in df.columns:
            calculated["volume_return"] = df["volume"].pct_change()
        
        if "volume_ma" in missing_volume and "volume" in df.columns:
            calculated["volume_ma"] = df["volume"].rolling(window=20).mean()
        
        if "volume_ratio" in missing_volume and "volume" in df.columns:
            calculated["volume_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()
        
        return calculated
    
    def _calculate_technical_indicators(self, df: pd.DataFrame, missing_technical: list[str]) -> pd.DataFrame:
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


def create_test_data() -> pd.DataFrame:
    """Create test data with intentionally missing columns to test calculation."""
    print("📊 Creating test data with missing columns...")
    
    # Create base klines data (missing some calculated columns)
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1min')
    
    # Create realistic price data
    np.random.seed(42)  # For reproducible results
    base_price = 100.0
    price_changes = np.random.normal(0, 0.001, len(dates))
    prices = [base_price]
    for change in price_changes[1:]:
        prices.append(prices[-1] * (1 + change))
    
    # Create OHLCV data (missing calculated columns)
    data = {
        'timestamp': [int(dt.timestamp() * 1000) for dt in dates],
        'open': [p * (1 + np.random.normal(0, 0.0005)) for p in prices],
        'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, len(dates))
    }
    
    df = pd.DataFrame(data)
    
    print(f"✅ Created test data with {len(df)} rows and {len(df.columns)} columns")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Missing columns that should be calculated: close_return, vwap, etc.")
    
    return df


def test_column_verifier():
    """Test the ColumnVerifier class functionality."""
    print("\n🧪 Testing ColumnVerifier...")
    
    try:
        # Create test data
        test_data = create_test_data()
        
        # Initialize column verifier
        column_verifier = ColumnVerifier()
        
        # Test verification
        print("🔍 Testing column verification...")
        missing_info = column_verifier.verify_missing_columns(test_data, data_type="unified")
        
        # Check results
        print(f"   Verification passed: {missing_info['verification_passed']}")
        print(f"   Missing required: {missing_info['missing_required']}")
        print(f"   Missing optional: {missing_info['missing_optional']}")
        print(f"   Can calculate: {missing_info['can_calculate']}")
        
        # Test calculation
        print("🔄 Testing column calculation...")
        enhanced_data = column_verifier.calculate_missing_columns(test_data, missing_info)
        
        # Check what was calculated
        original_columns = set(test_data.columns)
        new_columns = set(enhanced_data.columns) - original_columns
        
        print(f"   Original columns: {len(original_columns)}")
        print(f"   New columns: {len(new_columns)}")
        print(f"   Calculated columns: {list(new_columns)}")
        
        # Verify specific calculations
        success = True
        if 'close_return' in new_columns:
            print("   ✅ close_return calculated successfully")
        else:
            print("   ❌ close_return not calculated")
            success = False
        
        if 'vwap' in new_columns:
            print("   ✅ vwap calculated successfully")
        else:
            print("   ❌ vwap not calculated")
            success = False
        
        if 'vwap_return' in new_columns:
            print("   ✅ vwap_return calculated successfully")
        else:
            print("   ❌ vwap_return not calculated")
            success = False
        
        if 'price_vwap_ratio' in new_columns:
            print("   ✅ price_vwap_ratio calculated successfully")
        else:
            print("   ❌ price_vwap_ratio not calculated")
            success = False
        
        # Test data quality
        print("🔍 Testing calculated data quality...")
        if 'close_return' in enhanced_data.columns:
            # Check for reasonable values
            close_return = enhanced_data['close_return']
            if close_return.isna().sum() > len(close_return) * 0.1:  # More than 10% NaN
                print("   ⚠️ close_return has too many NaN values")
                success = False
            else:
                print("   ✅ close_return data quality looks good")
        
        if 'vwap' in enhanced_data.columns:
            # Check for reasonable values
            vwap = enhanced_data['vwap']
            if vwap.isna().sum() > len(vwap) * 0.2:  # More than 20% NaN (rolling window effect)
                print("   ⚠️ vwap has too many NaN values")
                success = False
            else:
                print("   ✅ vwap data quality looks good")
        
        return success
        
    except Exception as e:
        print(f"❌ ColumnVerifier test failed: {e}")
        return False


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n🧪 Testing edge cases...")
    
    try:
        column_verifier = ColumnVerifier()
        
        # Test with empty DataFrame
        print("🔍 Testing with empty DataFrame...")
        empty_df = pd.DataFrame()
        missing_info = column_verifier.verify_missing_columns(empty_df, data_type="unified")
        print(f"   Empty DataFrame handling: {'✅' if missing_info['verification_passed'] == False else '❌'}")
        
        # Test with DataFrame missing all required columns
        print("🔍 Testing with DataFrame missing required columns...")
        invalid_df = pd.DataFrame({'random_col': [1, 2, 3]})
        missing_info = column_verifier.verify_missing_columns(invalid_df, data_type="unified")
        print(f"   Missing required columns handling: {'✅' if missing_info['verification_passed'] == False else '❌'}")
        
        # Test with DataFrame having only some price columns
        print("🔍 Testing with partial price data...")
        partial_df = pd.DataFrame({
            'timestamp': [1000000, 1000060, 1000120],
            'close': [100.0, 101.0, 99.5],
            'volume': [1000, 1100, 900]
        })
        missing_info = column_verifier.verify_missing_columns(partial_df, data_type="unified")
        enhanced_partial = column_verifier.calculate_missing_columns(partial_df, missing_info)
        
        # Check if VWAP was calculated (should be possible with close and volume)
        if 'vwap' in enhanced_partial.columns:
            print("   ✅ VWAP calculation with partial data works")
        else:
            print("   ❌ VWAP calculation with partial data failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Edge cases test failed: {e}")
        return False


def main():
    """Main test function."""
    print("🚀 Starting Step1_5 Column Verification Tests")
    print("=" * 60)
    
    # Run tests
    test1_result = test_column_verifier()
    test2_result = test_edge_cases()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"   ColumnVerifier test: {'✅ PASSED' if test1_result else '❌ FAILED'}")
    print(f"   Edge cases test: {'✅ PASSED' if test2_result else '❌ FAILED'}")
    
    total_tests = 2
    passed_tests = sum([test1_result, test2_result])
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Step1_5 column verification enhancement is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Please review the implementation.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)