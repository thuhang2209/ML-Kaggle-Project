## Experiment Tracker (Bảng Theo Dõi Thử Nghiệm - Regression)

> **Quy tắc Metric**: 
> - **RMSLE (Root Mean Squared Error of Log1p)**: **CÀNG THẤP CÀNG TỐT**.

|   Exp ID   | Người thực hiện |         Mô hình         |             Feature Set              |           Preprocessing           |      Hyperparameters      | Val RMSLE  | Thời gian chạy | Ghi chú & Kết luận                            |
| :--------: | :-------------: | :---------------------: | :----------------------------------: | :-------------------------------: | :-----------------------: | :--------: | :------------: | :-------------------------------------------- |
| **EXP001** |  Thành viên 2   | Weekday Median Baseline |          Store ID, Weekday           |  Fallback Store / Overall Median  |            N/A            | **0.2387** |     2 giây     | Baseline chuẩn từ starter notebook.           |
| **EXP002** |  Thành viên 2   |    Linear Regression    |    Basic Numeric + One-Hot Stores    | Log1p transformation, Impute Mean |          Default          | **0.2215** |     5 giây     | Cải thiện so với baseline.                    |
| **EXP003** |  Thành viên 3   |        LightGBM         |  Basic + Calendar (Month, Day, Dow)  |   Categorical Feature Encoding    | lr=0.05, n_estimators=500 | **0.1982** |    12 giây     | Giảm mạnh RMSLE nhờ cây quyết định.           |
| **EXP004** |  Thành viên 4   |        LightGBM         |  Advanced (Lag 7, 14, Rolling Mean)  |   Drop early rows with NaN lags   |  lr=0.03, num_leaves=31   | **0.1745** |    25 giây     | Feature Lag đem lại hiệu quả vượt trội.       |
| **EXP005** |  Thành viên 3   |        CatBoost         |  Advanced + Holiday Dist + Weather   |    Internal CatBoost Encoding     | depth=6, iterations=1000  | **0.1698** |    45 giây     | CatBoost xử lý biến cửa hàng tốt hơn.         |
| **EXP006** |  Thành viên 4   |    Weighted Ensemble    | Blending EXP004 (0.4) + EXP005 (0.6) |     Mean Prediction Log Scale     |            N/A            | **0.1632** |     3 giây     | Ensemble cải thiện CV so với các mô hình đơn. |
## Experiment Tracker (Bảng Theo Dõi Thử Nghiệm Classification)

> **Quy tắc Metric**: 
> - **ROC-AUC (Area Under ROC Curve)**: **CÀNG CAO CÀNG TỐT** (0.5: Ngẫu nhiên, 1.0: Hoàn hảo).

| Exp ID | Người thực hiện | Mô hình | Feature Set | Preprocessing | Hyperparameters | Val ROC-AUC (5-Fold) | Thời gian chạy | Ghi chú & Kết luận |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **EXP001** | Thành viên 2 | Logistic Regression | Numeric Features Only | Median Impute + StandardScaler | Default (max_iter=2000) | **0.7290** | 3 giây | Baseline từ starter notebook. |
| **EXP002** | Thành viên 2 | Random Forest | Numeric + Cleaned Categoricals | One-Hot Encoding + Median Impute | n_estimators=200, max_depth=8 | **0.7850** | 15 giây | Bắt được tương tác phi tuyến tính tốt hơn. |
| **EXP003** | Thành viên 3 | LightGBM | Cleaned All Features | Native Categorical Handling | lr=0.05, num_leaves=31, scale_pos_weight=3 | **0.8320** | 10 giây | Cải thiện mạnh điểm AUC nhờ Boosting. |
| **EXP004** | Thành viên 4 | LightGBM | Advanced (Util_ratio, Spend_diff) | Cleaned + Engineered Features | Optuna Tuned Params | **0.8515** | 20 giây | Feature Engineering đem lại hiệu quả vượt trội. |
| **EXP005** | Thành viên 3 | CatBoost | Advanced Features Set | CatBoost Internal Categorical | depth=6, iterations=1000, l2_leaf_reg=3 | **0.8560** | 35 giây | CatBoost xử lý biến định tính xuất sắc. |
| **EXP006** | Thành viên 4 | Rank Ensemble | Blending EXP004 (0.4) + EXP005 (0.6) | Rank Averaging Predictions | N/A | **0.8625** | 5 giây | Ensemble giúp ổn định và tăng AUC thêm 0.0065. |
