import time
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy import stats

start_time = time.time()

# 特征标准化
scaler = StandardScaler()
# 加载数据集
train_dataSet = pd.read_csv(r'C:/Users/admin/PycharmProjects/PythonProject2/001-基准算法/modified_数据集Time_Series661_detail.dat')
test_dataSet = pd.read_csv(r'C:/Users/admin/PycharmProjects/PythonProject2/001-基准算法/modified_数据集Time_Series662_detail.dat')
# columns表示原始列，noise_columns表示添加噪声的额列
columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth',]
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

CL = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth','Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

## 查看数据缺失情况
data = train_dataSet[CL]
missingDf=data.isnull().sum().sort_values(ascending=False).reset_index()
missingDf.columns=['feature','miss_num']
missingDf['miss_percentage']=missingDf['miss_num']/data.shape[0]  #缺失值比例
print("缺失值比例")
print(missingDf)

# 初始化一个字典来存储每一列的异常值比例
outlier_ratios = {}

# 遍历每一列
for column in CL:
    # 计算每一列的Z分数
    z_scores = np.abs(stats.zscore(train_dataSet[column]))

    # 找出异常值（假设Z分数大于2为异常值）
    outliers = (z_scores > 2)

    # 计算异常值的比例
    outlier_ratio = outliers.mean()

    # 存储异常值比例
    outlier_ratios[column] = outlier_ratio
print("*"*30)
# 打印结果
print("异常值的比例:")
for column, ratio in outlier_ratios.items():
    print(f"{column}: {ratio:.2%}")

# 划分训练集中X_Train和y_Train
X_train = train_dataSet[noise_columns]
y_train = train_dataSet[columns]
# 划分测试集中X_test和y_test
X_test = test_dataSet[noise_columns]
y_test = test_dataSet[columns]

# 数据标准化
print("数据标准化...")
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""随机森林模型参数设置"""
# 基础参数
base_params = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 0.6,
    'bootstrap': True,
    'random_state': 217,
    'n_jobs': -1,  # 使用所有CPU核心
    'verbose': 1
}

# 如果需要调参，可以使用以下参数网格
rf_params = {
    # 'n_estimators': [100, 200, 300],
    # 'max_depth': [10, 15, 20, None],
    # 'min_samples_split': [2, 5, 10],
    # 'min_samples_leaf': [1, 2, 4],
    # 'max_features': ['sqrt', 'log2', None]
}

print("训练随机森林模型...")
# 创建随机森林模型
rf_model = RandomForestRegressor(**base_params)

# 如果需要调参，取消注释以下代码
# print("进行参数调优...")
# rf_grid = GridSearchCV(rf_model, rf_params, cv=5, scoring='r2', n_jobs=-1, verbose=1)
# rf_grid.fit(X_train_scaled, y_train)
# print(f"最佳参数: {rf_grid.best_params_}")
# print(f"最佳得分: {rf_grid.best_score_:.4f}")
# rf_model = rf_grid.best_estimator_

# 直接训练模型
rf_model.fit(X_train_scaled, y_train)

# 交叉验证评估
print("进行交叉验证...")
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"交叉验证R2分数: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 预测
print("进行预测...")
y_predict = rf_model.predict(X_test_scaled)

# 计算模型性能指标
print("\n模型性能评估:")
for i, col in enumerate(columns):
    r2 = r2_score(y_test[col], y_predict[:, i])
    mse = mean_squared_error(y_test[col], y_predict[:, i])
    rmse = np.sqrt(mse)
    print(f"{col}: R2 = {r2:.4f}, RMSE = {rmse:.4f}")

# 计算总体平均指标
total_r2 = 0
total_rmse = 0
for i, col in enumerate(columns):
    r2 = r2_score(y_test[col], y_predict[:, i])
    mse = mean_squared_error(y_test[col], y_predict[:, i])
    rmse = np.sqrt(mse)
    total_r2 += r2
    total_rmse += rmse

avg_r2 = total_r2 / len(columns)
avg_rmse = total_rmse / len(columns)
print(f"\n总体平均: R2 = {avg_r2:.4f}, RMSE = {avg_rmse:.4f}")

results = []
# 遍历y_test和y_predict，并且计算误差
for True_Value, Predicted_Value in zip(y_test.values, y_predict):
    error = np.abs(True_Value - Predicted_Value)
    # 格式化True_Value和Predicted_Value为原始数据格式
    formatted_true_value = ' '.join([f"{x:.6f}" for x in True_Value])
    formatted_predicted_value = ' '.join([f"{x:.6f}" for x in Predicted_Value])
    formatted_error = ' '.join([f"{x:.6f}" for x in error])
    results.append([formatted_true_value, formatted_predicted_value, formatted_error])

# 结果写入CSV文件当中
result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_RandomForest.csv", index=False)

print("<*>"*50)

# 从CSV文件读取数据
data = pd.read_csv("result_RandomForest.csv")

# 提取第三列数据
column3 = data.iloc[:, 2]

# 将每行的6个数字拆分并转换为数字列表
numbers = column3.str.split(' ', expand=True).apply(pd.to_numeric)

# 计算平均值
means = numbers.mean()

# 打印结果
print("6个数据的平均绝对误差为：")
for i, col in enumerate(columns):
    print(f"  {col}: {means[i]:.6f}")
print(f"\n总体平均绝对误差: {means.mean():.6f}")

# 特征重要性分析
print("\n特征重要性分析:")
feature_importance = pd.DataFrame({
    'feature': noise_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)

end_time = time.time()
print(f"\n总耗时：{end_time - start_time:.3f}秒")
print("随机森林算法完成！")