# Enhanced prediction section for ml_momentum_persistence_step.py

            # Generate predictions with binary output
            predictions = model.predict(features)
            probabilities = model.predict_proba(features)[:, 1]
            
            # Convert to single 0/1 scalar output
            binary_predictions = (probabilities >= 0.5).astype(int)
            
            # Create output DataFrame with enhanced metrics
            output_df = features.copy()
            output_df['ml_momentum_persistence_step_prediction'] = binary_predictions
            output_df['ml_momentum_persistence_step_probability'] = probabilities
            output_df['ml_momentum_persistence_step_label'] = labels
            
            # Add MI/HSIC metrics
            try:
                from sklearn.feature_selection import mutual_info_regression
                pred_mi = mutual_info_regression(binary_predictions.reshape(-1, 1), labels)[0]
                metrics['prediction_mi_to_target'] = pred_mi
                tprint_info(f"📊 Prediction MI to target: {pred_mi:.4f}")
            except:
                metrics['prediction_mi_to_target'] = 0
            
            # Feature orthogonality check
            try:
                corr_matrix = features.corr().abs()
                high_corr = (corr_matrix > 0.7).sum().sum() - len(features.columns)
                metrics['high_correlation_pairs'] = high_corr // 2
                if high_corr > 0:
                    tprint_warning(f"⚠️ Found {high_corr//2} highly correlated feature pairs")
            except:
                metrics['high_correlation_pairs'] = 0
