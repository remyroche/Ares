# Feature Generation Report

## Summary
- **Symbol**: ETHUSDT
- **Exchange**: binance
- **Timeframe**: 15m
- **Direction**: DirectionType.LONGS
- **Generated At**: 2025-10-18 15:40:06
- **Status**: ✅ SUCCESS

## Feature Generation Results
- **Total Features Generated**: 280
- **Generation Time**: 648.452 seconds
- **Memory Usage**: 1279.23 MB
- **Cache Hit**: No

## Feature Categories
- **Acceleration**: 32 features generated
- **Entropy**: 25 features generated
- **Momentum**: 36 features generated
- **Oscillator**: 1 features generated
- **Returns**: 185 features generated
- **Spectral_Wavelet**: 1 features generated
- **Time**: 1 features generated
- **Trend**: 16 features generated
- **Volatility**: 45 features generated
- **Volume**: 16 features generated

## Detailed Feature List by Category

### Acceleration Features (32 features)
1. vectorbt_acceleration_correlation_20_price_returns
2. acceleration_features
3. vectorbt_acceleration_5_price_returns
4. vectorbt_acceleration_10_price_returns
5. vectorbt_volatility_acceleration_5_20_price_returns
6. vectorbt_volume_acceleration_5_volume_returns
7. vectorbt_momentum_acceleration_5_10_price_returns
8. vectorbt_acceleration_momentum_5_10_price_returns
9. vectorbt_acceleration_volatility_5_10_price_returns
10. vectorbt_acceleration_trend_strength_5_10_price_returns
11. vectorbt_acceleration_consistency_5_10_price_returns
12. vectorbt_acceleration_regime_5_10_price_returns
13. vectorbt_acceleration_volatility_5_20_price_returns
14. vectorbt_acceleration_trend_strength_5_20_price_returns
15. vectorbt_acceleration_momentum_5_20_price_returns
16. vectorbt_momentum_acceleration_5_20_price_returns
17. vectorbt_acceleration_consistency_5_20_price_returns
18. vectorbt_acceleration_regime_5_20_price_returns
19. vectorbt_acceleration_volatility_10_10_price_returns
20. vectorbt_acceleration_momentum_10_10_price_returns
21. vectorbt_momentum_acceleration_10_10_price_returns
22. vectorbt_acceleration_trend_strength_10_10_price_returns
23. vectorbt_acceleration_consistency_10_10_price_returns
24. vectorbt_momentum_acceleration_10_20_price_returns
25. vectorbt_acceleration_momentum_10_20_price_returns
26. vectorbt_acceleration_regime_10_10_price_returns
27. vectorbt_acceleration_volatility_10_20_price_returns
28. vectorbt_acceleration_trend_strength_10_20_price_returns
29. vectorbt_acceleration_regime_10_20_price_returns
30. vectorbt_multi_timeframe_acceleration_5_20_price_returns
31. vectorbt_acceleration_consistency_10_20_price_returns
32. vectorbt_acceleration_divergence_20_price_returns


### Entropy Features (25 features)
1. rsi_entropy_20_14
2. macd_entropy_20_12_26
3. price_entropy_5_price_returns
4. return_entropy_5_price_returns
5. price_entropy_ma_5_5_price_returns
6. return_entropy_ma_5_5_price_returns
7. price_entropy_ma_5_10_price_returns
8. return_entropy_ma_5_10_price_returns
9. price_entropy_10_price_returns
10. return_entropy_10_price_returns
11. price_entropy_ma_10_5_price_returns
12. return_entropy_ma_10_5_price_returns
13. price_entropy_ma_10_10_price_returns
14. return_entropy_ma_10_10_price_returns
15. price_entropy_20_price_returns
16. return_entropy_20_price_returns
17. price_entropy_ma_20_5_price_returns
18. return_entropy_ma_20_5_price_returns
19. price_entropy_ma_20_10_price_returns
20. return_entropy_ma_20_10_price_returns
21. shannon_entropy_20_10
22. permutation_entropy_20_3_1
23. sample_entropy_20_2_0.2
24. entropy_rate_20
25. spectral_entropy_20


### Momentum Features (36 features)
1. momentum_endpoints_sma_20
2. ctf_corr_momentum_5_15_20_price_returns
3. vectorbt_momentum_comprehensive_14
4. momentum_features
5. vectorbt_momentum_comprehensive_9
6. vectorbt_momentum_comprehensive_21
7. vectorbt_momentum_comprehensive_30
8. momentum_21_price_returns
9. momentum_30_price_returns
10. momentum_14_price_returns
11. advanced_momentum_5_20
12. advanced_momentum_10_30
13. analyst_momentum_1h
14. analyst_momentum_5m
15. analyst_momentum_15m
16. analyst_momentum_alignment
17. volume_momentum_5
18. volume_momentum_10
19. volume_momentum_20
20. ctf_15m_momentum_price_returns
21. ctf_5m_momentum_price_returns
22. ctf_30m_momentum_price_returns
23. ctf_ratio_momentum_5_20_price_returns
24. ctf_divergence_momentum_5_20_price_returns
25. vectorbt_momentum_5_price_returns
26. vectorbt_momentum_10_price_returns
27. vectorbt_momentum_20_price_returns
28. vectorbt_momentum_50_price_returns
29. vectorbt_momentum_acceleration_5_10_price_returns
30. vectorbt_acceleration_momentum_5_10_price_returns
31. vectorbt_acceleration_momentum_5_20_price_returns
32. vectorbt_momentum_acceleration_5_20_price_returns
33. vectorbt_acceleration_momentum_10_10_price_returns
34. vectorbt_momentum_acceleration_10_10_price_returns
35. vectorbt_momentum_acceleration_10_20_price_returns
36. vectorbt_acceleration_momentum_10_20_price_returns


### Oscillator Features (1 features)
1. ultimate_oscillator_7_14_28_returns_vwap


### Returns Features (185 features)
1. macd_12_26_9_returns_vwap
2. rsi_21_returns_vwap
3. rsi_30_returns_vwap
4. rsi_14_returns_vwap
5. sma_5_returns_vwap
6. sma_10_returns_vwap
7. sma_20_returns_vwap
8. sma_50_returns_vwap
9. sma_100_returns_vwap
10. ema_12_returns_vwap
11. ema_26_returns_vwap
12. ema_50_returns_vwap
13. dema_21_price_returns
14. tema_21_price_returns
15. ctf_ratio_sma_10_50_price_returns
16. ctf_corr_volatility_15_30_20_price_returns
17. ctf_corr_momentum_5_15_20_price_returns
18. vectorbt_acceleration_correlation_20_price_returns
19. log_returns_1_price_returns
20. log_returns_5_price_returns
21. log_returns_10_price_returns
22. simple_returns_1_price_returns
23. simple_returns_5_price_returns
24. simple_returns_10_price_returns
25. cumulative_returns_10_price_returns
26. cumulative_returns_20_price_returns
27. rolling_returns_10_price_returns
28. rolling_returns_20_price_returns
29. returns_skewness_20_price_returns
30. returns_kurtosis_20_price_returns
31. sharpe_ratio_20_0.0_price_returns
32. advanced_cumulative_returns_10
33. advanced_cumulative_returns_20
34. rolling_zscore_returns_20
35. stochastic_14_3_price_returns
36. williams_r_14_price_returns
37. roc_14_price_returns
38. stochastic_21_3_price_returns
39. williams_r_21_price_returns
40. roc_21_price_returns
41. stochastic_30_3_price_returns
42. williams_r_30_price_returns
43. roc_30_price_returns
44. wma_20_price_returns
45. mama_21_0.05_price_returns
46. vwma_20_price_returns
47. keltner_channels_20_14_price_returns
48. adx_14_returns_vwap
49. cci_20_returns_vwap
50. aroon_25_returns_vwap
51. ultimate_oscillator_7_14_28_returns_vwap
52. kst_10_15_20_30_10_10_10_15_returns_vwap
53. apo_12_26_returns_vwap
54. cmo_14_returns_vwap
55. natr_14_returns_vwap
56. pfe_12_returns_vwap
57. t3_14_0.7_returns_vwap
58. kama_30_2_30_returns_vwap
59. support_level_1_5_price_returns
60. support_level_2_5_price_returns
61. support_level_3_5_price_returns
62. support_level_4_5_price_returns
63. support_level_5_5_price_returns
64. resistance_level_1_5_price_returns
65. resistance_level_2_5_price_returns
66. resistance_level_3_5_price_returns
67. resistance_level_4_5_price_returns
68. resistance_level_5_5_price_returns
69. pivot_point_5_price_returns
70. support_level_1_10_price_returns
71. support_level_2_10_price_returns
72. support_level_3_10_price_returns
73. support_level_4_10_price_returns
74. support_level_5_10_price_returns
75. resistance_level_1_10_price_returns
76. resistance_level_2_10_price_returns
77. resistance_level_3_10_price_returns
78. resistance_level_4_10_price_returns
79. resistance_level_5_10_price_returns
80. pivot_point_10_price_returns
81. support_level_1_20_price_returns
82. support_level_2_20_price_returns
83. support_level_3_20_price_returns
84. support_level_4_20_price_returns
85. support_level_5_20_price_returns
86. resistance_level_1_20_price_returns
87. resistance_level_2_20_price_returns
88. resistance_level_3_20_price_returns
89. resistance_level_4_20_price_returns
90. resistance_level_5_20_price_returns
91. pivot_point_20_price_returns
92. fibonacci_0.236_5_price_returns
93. fibonacci_0.236_10_price_returns
94. fibonacci_0.236_20_price_returns
95. fibonacci_0.382_5_price_returns
96. fibonacci_0.382_10_price_returns
97. fibonacci_0.382_20_price_returns
98. fibonacci_0.5_5_price_returns
99. fibonacci_0.5_10_price_returns
100. fibonacci_0.5_20_price_returns
101. fibonacci_0.618_5_price_returns
102. fibonacci_0.618_10_price_returns
103. fibonacci_0.618_20_price_returns
104. fibonacci_0.786_5_price_returns
105. fibonacci_0.786_10_price_returns
106. fibonacci_0.786_20_price_returns
107. ctf_5m_trend_price_returns
108. ctf_15m_trend_price_returns
109. ctf_30m_trend_price_returns
110. price_entropy_5_price_returns
111. return_entropy_5_price_returns
112. price_entropy_ma_5_5_price_returns
113. return_entropy_ma_5_5_price_returns
114. price_entropy_ma_5_10_price_returns
115. return_entropy_ma_5_10_price_returns
116. price_entropy_10_price_returns
117. return_entropy_10_price_returns
118. price_entropy_ma_10_5_price_returns
119. return_entropy_ma_10_5_price_returns
120. price_entropy_ma_10_10_price_returns
121. return_entropy_ma_10_10_price_returns
122. price_entropy_20_price_returns
123. return_entropy_20_price_returns
124. price_entropy_ma_20_5_price_returns
125. return_entropy_ma_20_5_price_returns
126. price_entropy_ma_20_10_price_returns
127. return_entropy_ma_20_10_price_returns
128. returns_volatility_20_price_returns
129. momentum_21_price_returns
130. momentum_30_price_returns
131. momentum_14_price_returns
132. ctf_15m_momentum_price_returns
133. ctf_5m_momentum_price_returns
134. ctf_30m_momentum_price_returns
135. ctf_5m_volatility_price_returns
136. ctf_30m_volatility_price_returns
137. ctf_15m_volatility_price_returns
138. ctf_ratio_momentum_5_20_price_returns
139. ctf_ratio_volatility_5_20_price_returns
140. ctf_divergence_momentum_5_20_price_returns
141. vectorbt_momentum_5_price_returns
142. ctf_divergence_volatility_5_20_price_returns
143. vectorbt_momentum_10_price_returns
144. vectorbt_momentum_20_price_returns
145. vectorbt_momentum_50_price_returns
146. vectorbt_acceleration_5_price_returns
147. vectorbt_acceleration_10_price_returns
148. vectorbt_jerk_10_price_returns
149. vectorbt_jerk_5_price_returns
150. vectorbt_trend_strength_5_price_returns
151. vectorbt_trend_consistency_5_price_returns
152. vectorbt_trend_strength_10_price_returns
153. vectorbt_trend_consistency_10_price_returns
154. vectorbt_trend_strength_20_price_returns
155. vectorbt_trend_consistency_20_price_returns
156. vectorbt_trend_strength_50_price_returns
157. vectorbt_trend_consistency_50_price_returns
158. vectorbt_volatility_acceleration_5_20_price_returns
159. vectorbt_volume_acceleration_5_volume_returns
160. vectorbt_momentum_acceleration_5_10_price_returns
161. vectorbt_acceleration_momentum_5_10_price_returns
162. vectorbt_acceleration_volatility_5_10_price_returns
163. vectorbt_acceleration_trend_strength_5_10_price_returns
164. vectorbt_acceleration_consistency_5_10_price_returns
165. vectorbt_acceleration_regime_5_10_price_returns
166. vectorbt_acceleration_volatility_5_20_price_returns
167. vectorbt_acceleration_trend_strength_5_20_price_returns
168. vectorbt_acceleration_momentum_5_20_price_returns
169. vectorbt_momentum_acceleration_5_20_price_returns
170. vectorbt_acceleration_consistency_5_20_price_returns
171. vectorbt_acceleration_regime_5_20_price_returns
172. vectorbt_acceleration_volatility_10_10_price_returns
173. vectorbt_acceleration_momentum_10_10_price_returns
174. vectorbt_momentum_acceleration_10_10_price_returns
175. vectorbt_acceleration_trend_strength_10_10_price_returns
176. vectorbt_acceleration_consistency_10_10_price_returns
177. vectorbt_momentum_acceleration_10_20_price_returns
178. vectorbt_acceleration_momentum_10_20_price_returns
179. vectorbt_acceleration_regime_10_10_price_returns
180. vectorbt_acceleration_volatility_10_20_price_returns
181. vectorbt_acceleration_trend_strength_10_20_price_returns
182. vectorbt_acceleration_regime_10_20_price_returns
183. vectorbt_multi_timeframe_acceleration_5_20_price_returns
184. vectorbt_acceleration_consistency_10_20_price_returns
185. vectorbt_acceleration_divergence_20_price_returns


### Spectral_Wavelet Features (1 features)
1. vectorbt_spectral_wavelet_batch


### Time Features (1 features)
1. vectorbt_multi_timeframe_acceleration_5_20_price_returns


### Trend Features (16 features)
1. trend_score_14
2. ctf_5m_trend_price_returns
3. ctf_15m_trend_price_returns
4. ctf_30m_trend_price_returns
5. vectorbt_trend_strength_5_price_returns
6. vectorbt_trend_consistency_5_price_returns
7. vectorbt_trend_strength_10_price_returns
8. vectorbt_trend_consistency_10_price_returns
9. vectorbt_trend_strength_20_price_returns
10. vectorbt_trend_consistency_20_price_returns
11. vectorbt_trend_strength_50_price_returns
12. vectorbt_trend_consistency_50_price_returns
13. vectorbt_acceleration_trend_strength_5_10_price_returns
14. vectorbt_acceleration_trend_strength_5_20_price_returns
15. vectorbt_acceleration_trend_strength_10_10_price_returns
16. vectorbt_acceleration_trend_strength_10_20_price_returns


### Volatility Features (45 features)
1. ctf_corr_volatility_15_30_20_price_returns
2. returns_volatility_20_price_returns
3. volume_volatility_elasticity_20
4. enhanced_volatility_20
5. enhanced_volatility_10
6. enhanced_volatility_50
7. enhanced_volatility_100
8. enhanced_volatility_14
9. enhanced_volatility_30
10. vectorbt_volatility_comprehensive_10
11. vectorbt_volatility_comprehensive_14
12. vectorbt_volatility_comprehensive_20
13. vectorbt_volatility_comprehensive_30
14. vectorbt_volatility_comprehensive_50
15. vectorbt_parkinson_volatility_10
16. vectorbt_yang_zhang_volatility_10
17. vectorbt_rogers_satchell_volatility_10
18. vectorbt_garman_klass_volatility_10
19. vectorbt_parkinson_volatility_14
20. vectorbt_garman_klass_volatility_14
21. vectorbt_rogers_satchell_volatility_14
22. vectorbt_yang_zhang_volatility_14
23. vectorbt_parkinson_volatility_20
24. vectorbt_rogers_satchell_volatility_20
25. vectorbt_garman_klass_volatility_20
26. vectorbt_parkinson_volatility_30
27. vectorbt_yang_zhang_volatility_20
28. vectorbt_rogers_satchell_volatility_30
29. vectorbt_rogers_satchell_volatility_50
30. vectorbt_yang_zhang_volatility_30
31. vectorbt_garman_klass_volatility_30
32. vectorbt_parkinson_volatility_50
33. vectorbt_yang_zhang_volatility_50
34. vectorbt_garman_klass_volatility_50
35. ctf_5m_volatility_price_returns
36. ctf_30m_volatility_price_returns
37. ctf_15m_volatility_price_returns
38. ctf_ratio_volatility_5_20_price_returns
39. ctf_divergence_volatility_5_20_price_returns
40. vectorbt_volatility_acceleration_5_20_price_returns
41. vectorbt_acceleration_volatility_5_10_price_returns
42. vectorbt_acceleration_volatility_5_20_price_returns
43. vectorbt_acceleration_volatility_10_10_price_returns
44. vectorbt_acceleration_volatility_10_20_price_returns
45. band_limited_volatility


### Volume Features (16 features)
1. volume_sma_5
2. volume_ema_5
3. volume_sma_10
4. volume_ema_10
5. volume_sma_20
6. volume_ema_20
7. volume_sma_50
8. volume_ema_50
9. volume_momentum_5
10. volume_momentum_10
11. volume_volatility_elasticity_20
12. volume_momentum_20
13. vectorbt_volume_weighted_ad_line_10
14. vectorbt_volume_weighted_ad_line_50
15. vectorbt_volume_weighted_ad_line_20
16. vectorbt_volume_acceleration_5_volume_returns


## Optimization Statistics
- **vectorbt_optimization_enabled**: True
- **memory_optimization_enabled**: True
- **gpu_acceleration_enabled**: True
- **parallel_processing_enabled**: True

## Data Quality
- **Data Shape**: (1162368, 286)
- **Success**: Yes

## Stored Data

## Technical Details
- **Feature Data Type**: DataFrame
- **Generated Features Type**: DataFrame
- **Metadata Available**: Yes

## Recommendations
- ✅ Feature generation completed successfully
- 📊 Consider analyzing feature importance for model training
- 🔍 Review feature categories for completeness
- 💾 Features are ready for model training pipeline
