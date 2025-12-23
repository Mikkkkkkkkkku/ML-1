import time
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

start_time = time.time()

# 数据加载和预处理
scaler = StandardScaler()
train_dataSet = pd.read_csv(r'C:/Users/admin/PycharmProjects/PythonProject2/001-基准算法/modified_数据集Time_Series661_detail.dat')
test_dataSet = pd.read_csv(r'C:/Users/admin/PycharmProjects/PythonProject2/001-基准算法/modified_数据集Time_Series662_detail.dat')

columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

CL = columns + noise_columns

# 数据预处理 - 处理缺失值
X_train = train_dataSet[noise_columns].fillna(train_dataSet[noise_columns].median())
y_train = train_dataSet[columns].fillna(train_dataSet[columns].median())
X_test = test_dataSet[noise_columns].fillna(test_dataSet[noise_columns].median())
y_test = test_dataSet[columns].fillna(test_dataSet[columns].median())

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 方案1: 使用MultiOutputRegressor包装LightGBM
lgb_params = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 7,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 217
}

lgb_model = MultiOutputRegressor(LGBMRegressor(**lgb_params))
lgb_model.fit(X_train_scaled, y_train)
y_predict_lgb = lgb_model.predict(X_test_scaled)

# 方案2: 随机森林（原生支持多输出）
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=217,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
y_predict_rf = rf_model.predict(X_test_scaled)

# 方案3: 梯度提升树（使用MultiOutputRegressor）
gbr_model = MultiOutputRegressor(GradientBoostingRegressor(
    n_estimators=200,
    learning_rate= 0.1,
    max_depth=5,
    random_state=217
))
gbr_model.fit(X_train_scaled, y_train)
y_predict_gbr = gbr_model.predict(X_test_scaled)

# 方案4: XGBoost（使用MultiOutputRegressor）
xgb_params = {
    'max_depth': 5,
    'n_estimators': 200,
    'learning_rate': 0.1,
    'reg_alpha': 10,
    'reg_lambda': 6,
    'min_child_weight': 5,
    'colsample_bytree': 0.85,
    'subsample': 0.6,
    'random_state': 217
}

xgb_model = MultiOutputRegressor(XGBRegressor(**xgb_params))
xgb_model.fit(X_train_scaled, y_train)
y_predict_xgb = xgb_model.predict(X_test_scaled)

# 加权集成预测
def weighted_ensemble_predict(predictions, weights):
    """加权集成预测"""
    weighted_pred = np.zeros_like(predictions[0])
    for i, pred in enumerate(predictions):
        weighted_pred += weights[i] * pred
    return weighted_pred

# 集成所有模型
predictions = [y_predict_lgb, y_predict_rf, y_predict_gbr, y_predict_xgb]
# 可以根据各个模型的性能调整权重
weights = [0.3, 0.25, 0.2, 0.25]
y_predict_ensemble = weighted_ensemble_predict(predictions, weights)

# 选择最佳模型进行最终预测
y_predict = y_predict_ensemble

# 评估结果
results = []
for True_Value, Predicted_Value in zip(y_test.values, y_predict):
    error = np.abs(True_Value - Predicted_Value)
    formatted_true_value = ' '.join(map(str, True_Value))
    formatted_predicted_value = ' '.join(map(str, Predicted_Value))
    formatted_error = ' '.join(map(str, error))
    results.append([formatted_true_value, formatted_predicted_value, formatted_error])

# 保存结果
result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_improved.csv", index=False)

print("<*>"*50)

# 计算并比较各个模型的误差
models = {
    'LightGBM': y_predict_lgb,
    'RandomForest': y_predict_rf,
    'GradientBoosting': y_predict_gbr,
    'XGBoost': y_predict_xgb,
    'Ensemble': y_predict_ensemble
}

print("各个模型的误差比较:")
for name, pred in models.items():
    errors = np.abs(y_test.values - pred)
    mean_errors = errors.mean(axis=0)
    overall_mean = mean_errors.mean()
    print(f"{name:15} - 总体平均误差: {overall_mean:.6f}")
    print(f"{' ':15}   各目标变量误差: {mean_errors}")
    print("-" * 50)

end_time = time.time()
print(f"总耗时：{end_time - start_time : .3f}秒")