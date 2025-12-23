# 时间：2024年6月8号  Date： June 16, 2024
# 文件名称 Filename： 03-main.py
# 编码实现 Coding by： Hongjie Liu , Suiwen Zhang 邮箱 Mailbox：redsocks1043@163.com
# 所属单位：中国 成都，西南民族大学（Southwest  University of Nationality，or Southwest Minzu University）, 计算机科学与工程学院.
# 指导老师：周伟老师
# coding=utf-8
import time
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from GCForest import gcForest

start_time = time.time()

# 加载数据集
train_dataSet = pd.read_csv(r'C:/Users/admin/PycharmProjects/PythonProject2/001-基准算法/modified_数据集Time_Series661_detail.dat')
test_dataSet = pd.read_csv(r'C:/Users/admin/PycharmProjects/PythonProject2/001-基准算法/modified_数据集Time_Series662_detail.dat')

columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

# 划分训练集和测试集
X_train = train_dataSet[noise_columns]
y_train = train_dataSet[columns]
X_test = test_dataSet[noise_columns]
y_test = test_dataSet[columns]

# GCForest配置
shape_1X = (1, len(noise_columns))

# 为每个输出变量训练一个GCForest模型
y_predict = np.zeros_like(y_test.values)

for i, column in enumerate(columns):
    print(f"训练第 {i + 1}/{len(columns)} 个输出变量: {column}")

    # 离散化目标变量用于分类
    n_bins = 50
    bins = np.percentile(y_train[column].values, np.linspace(0, 100, n_bins + 1))
    y_train_discrete = np.digitize(y_train[column].values, bins) - 1
    y_train_discrete = np.clip(y_train_discrete, 0, n_bins - 1)

    # 创建并训练GCForest模型
    gcf = gcForest(
        shape_1X=shape_1X,
        n_mgsRFtree=30,
        window=[3],
        stride=1,
        cascade_test_size=0.2,
        n_cascadeRF=2,
        n_cascadeRFtree=50,
        cascade_layer=5,
        min_samples_mgs=0.1,
        min_samples_cascade=0.05,
        tolerance=0.001,
        n_jobs=-1
    )

    gcf.fit(X_train.values, y_train_discrete)

    # 预测并转换回连续值
    pred_proba = gcf.predict_proba(X_test.values)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    y_predict[:, i] = np.sum(pred_proba * bin_centers, axis=1)

# 保存预测结果
results = []
for True_Value, Predicted_Value in zip(y_test.values, y_predict):
    error = np.abs(True_Value - Predicted_Value)
    formatted_true_value = ' '.join(map(str, True_Value))
    formatted_predicted_value = ' '.join(map(str, Predicted_Value))
    formatted_error = ' '.join(map(str, error))
    results.append([formatted_true_value, formatted_predicted_value, formatted_error])

result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_GCForest.csv", index=False)
print("GCForest结果已保存到: result_GCForest.csv")

# 计算平均误差
data = pd.read_csv("result_GCForest.csv")
column3 = data.iloc[:, 2]
numbers = column3.str.split(' ', expand=True).apply(pd.to_numeric)
means = numbers.mean()

print("\nGCForest 6个数据的平均值为：")
print(means)
print(f"总平均误差: {means.mean():.6f}")

# 计算总体R²分数
overall_r2 = r2_score(y_test, y_predict)
print(f"总体 R² 分数: {overall_r2:.4f}")

end_time = time.time()
print(f"\n总耗时：{end_time - start_time:.3f}秒")