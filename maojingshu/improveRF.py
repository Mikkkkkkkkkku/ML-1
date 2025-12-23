import time
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
import gc

warnings.filterwarnings('ignore')

start_time = time.time()

# 加载数据集时直接选择需要的列，减少内存占用
print("加载数据集...")
columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

# 只读取需要的列
used_columns = columns + noise_columns
train_dataSet = pd.read_csv(
    r'C:/Users/admin/PycharmProjects/PythonProject2/001-基准算法/modified_数据集Time_Series661_detail.dat',
    usecols=used_columns)
test_dataSet = pd.read_csv(
    r'C:/Users/admin/PycharmProjects/PythonProject2/001-基准算法/modified_数据集Time_Series662_detail.dat',
    usecols=used_columns)


# 优化内存使用
def optimize_memory(df):
    """优化DataFrame的内存使用"""
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
    return df


train_dataSet = optimize_memory(train_dataSet)
test_dataSet = optimize_memory(test_dataSet)

# 快速异常值检测（可选，可注释掉以节省时间）
print("快速异常值检测...")
outlier_info = {}
for column in used_columns:
    data = train_dataSet[column].values
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((data < lower_bound) | (data > upper_bound)).sum()
    outlier_info[column] = outliers / len(data)

print("异常值比例:")
for col, ratio in outlier_info.items():
    if ratio > 0:
        print(f"  {col}: {ratio:.2%}")

# 划分训练集和测试集
X_train = train_dataSet[noise_columns]
y_train = train_dataSet[columns]
X_test = test_dataSet[noise_columns]
y_test = test_dataSet[columns]

# 数据标准化
print("数据标准化...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype('float32')
X_test_scaled = scaler.transform(X_test).astype('float32')

# 转换为numpy数组以提高效率
y_train_values = y_train.values.astype('float32')
y_test_values = y_test.values.astype('float32')

# 清理内存
del train_dataSet, test_dataSet
gc.collect()


class FastRandomForest:
    def __init__(self):
        self.models = {}

    def train_fast_models(self, X, y):
        """快速训练模型 - 优化版本"""
        print("快速训练各目标变量模型...")

        # 使用经验验证的高效参数
        base_params = {
            'n_estimators': 300,  # 进一步减少树的数量
            'max_depth': 20,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 0.9,
            'bootstrap': True,
            'max_samples': 0.9,
            'n_jobs': -1,  # 限制并行数
            'random_state': 217,
            'verbose': 0
        }

        predictions = np.zeros((X.shape[0], len(columns)), dtype='float32')

        for i, col in enumerate(columns):
            print(f"训练 {col}...")

            # 为不同目标变量微调参数
            if col in ['T_SONIC', 'CO2_sig_strgth']:
                params = base_params.copy()
                params['n_estimators'] = 100
                params['max_depth'] = 15
            elif col in ['CO2_density', 'H2O_sig_strgth']:
                params = base_params.copy()
                params['max_features'] = 'sqrt'
            elif col == 'CO2_density_fast_tmpr':
                params = base_params.copy()
                params['n_estimators'] = 60
                params['max_depth'] = 10
            else:  # H2O_density
                params = base_params.copy()
                params['max_samples'] = 0.9

            model = RandomForestRegressor(**params)
            model.fit(X, y[:, i])
            self.models[col] = model
            predictions[:, i] = model.predict(X)

            # 快速验证（使用部分数据）
            if X.shape[0] > 5000:
                sample_size = min(3000, X.shape[0])
                sample_idx = np.random.choice(X.shape[0], size=sample_size, replace=False)
                X_sample = X[sample_idx]
                y_sample = y[sample_idx, i]

                # 快速交叉验证（2折）
                try:
                    cv_scores = cross_val_score(model, X_sample, y_sample, cv=2,
                                                scoring='r2', n_jobs=1)
                    print(f"  {col} 快速R2: {cv_scores.mean():.4f}")
                except:
                    print(f"  {col} 训练完成")
            else:
                print(f"  {col} 训练完成")

        return predictions

    def predict_fast(self, X):
        """快速预测"""
        predictions = np.zeros((X.shape[0], len(columns)), dtype='float32')
        for i, col in enumerate(columns):
            predictions[:, i] = self.models[col].predict(X)
        return predictions


# 创建快速模型实例
fast_rf = FastRandomForest()

# 训练模型
print("=" * 50)
print("开始训练快速随机森林模型")
print("=" * 50)

train_start = time.time()
y_predict_train = fast_rf.train_fast_models(X_train_scaled, y_train_values)
train_time = time.time() - train_start
print(f"训练完成，耗时: {train_time:.2f}秒")

# 预测测试集
print("\n预测测试集...")
predict_start = time.time()
y_predict_test = fast_rf.predict_fast(X_test_scaled)
predict_time = time.time() - predict_start
print(f"预测完成，耗时: {predict_time:.2f}秒")

# 性能评估
print("\n" + "=" * 50)
print("模型性能评估")
print("=" * 50)

# 训练集性能
print("训练集性能:")
train_r2_scores = []
train_rmse_scores = []

for i, col in enumerate(columns):
    r2 = r2_score(y_train_values[:, i], y_predict_train[:, i])
    rmse = np.sqrt(mean_squared_error(y_train_values[:, i], y_predict_train[:, i]))
    train_r2_scores.append(r2)
    train_rmse_scores.append(rmse)
    print(f"  {col}: R2 = {r2:.4f}, RMSE = {rmse:.4f}")

print(f"训练集平均R2: {np.mean(train_r2_scores):.4f}")
print(f"训练集平均RMSE: {np.mean(train_rmse_scores):.4f}")

# 测试集性能
print("\n测试集性能:")
test_r2_scores = []
test_rmse_scores = []
test_mae_scores = []

for i, col in enumerate(columns):
    r2 = r2_score(y_test_values[:, i], y_predict_test[:, i])
    rmse = np.sqrt(mean_squared_error(y_test_values[:, i], y_predict_test[:, i]))
    mae = np.mean(np.abs(y_test_values[:, i] - y_predict_test[:, i]))

    test_r2_scores.append(r2)
    test_rmse_scores.append(rmse)
    test_mae_scores.append(mae)

    print(f"  {col}:")
    print(f"    R² = {r2:.4f}")
    print(f"    RMSE = {rmse:.4f}")
    print(f"    MAE = {mae:.4f}")

avg_r2 = np.mean(test_r2_scores)
avg_rmse = np.mean(test_rmse_scores)
avg_mae = np.mean(test_mae_scores)

print(f"\n测试集总体平均:")
print(f"  R² = {avg_r2:.4f}")
print(f"  RMSE = {avg_rmse:.4f}")
print(f"  MAE = {avg_mae:.4f}")

# 保存结果（优化版 - 只保存部分结果以避免IO瓶颈）
print("\n保存简化结果...")
# 只保存前1000个样本的结果用于分析
save_samples = min(1000, len(y_test))
results = []

for i in range(save_samples):
    true_val = y_test_values[i]
    pred_val = y_predict_test[i]
    error = np.abs(true_val - pred_val)

    formatted_true = ' '.join([f"{x:.6f}" for x in true_val])
    formatted_pred = ' '.join([f"{x:.6f}" for x in pred_val])
    formatted_error = ' '.join([f"{x:.6f}" for x in error])
    results.append([formatted_true, formatted_pred, formatted_error])

result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_FastRandomForest.csv", index=False)
print(f"已保存 {save_samples} 个样本的结果")

# 特征重要性分析（简化版）
print("\n" + "=" * 50)
print("特征重要性分析")
print("=" * 50)

# 计算平均特征重要性
avg_importance = np.zeros(len(noise_columns))
for col in columns:
    avg_importance += fast_rf.models[col].feature_importances_
avg_importance /= len(columns)

feature_importance = pd.DataFrame({
    'feature': noise_columns,
    'importance': avg_importance
}).sort_values('importance', ascending=False)

print(feature_importance)

# 最终误差分析
print("\n" + "=" * 50)
print("最终误差分析")
print("=" * 50)

print("各目标变量的平均绝对误差:")
for i, col in enumerate(columns):
    print(f"  {col}: {test_mae_scores[i]:.6f}")

print(f"\n当前总体平均绝对误差: {avg_mae:.6f}")

# 性能改进建议
improvement_suggestions = []
if avg_mae > 0.5:
    improvement_suggestions.append("考虑增加树的数量到100-150")
    improvement_suggestions.append("尝试调整max_depth到15-20")
if avg_r2 < 0.8:
    improvement_suggestions.append("检查特征工程，可能需要更多相关特征")
    improvement_suggestions.append("考虑使用特征选择方法")

if improvement_suggestions:
    print("\n性能改进建议:")
    for suggestion in improvement_suggestions:
        print(f"  • {suggestion}")

end_time = time.time()
total_time = end_time - start_time

print(f"\n=== 执行总结 ===")
print(f"总运行时间: {total_time:.2f}秒")
print(f"训练时间: {train_time:.2f}秒")
print(f"预测时间: {predict_time:.2f}秒")
print(f"最终MAE: {avg_mae:.6f}")
print(f"最终R²: {avg_r2:.4f}")

if total_time < 60:
    print("✅ 优化成功！运行时间大幅减少")
else:
    print("⚠️  运行时间仍较长，建议进一步优化参数")

print("快速随机森林算法完成！")