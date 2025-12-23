# 时间：2024年6月8号  Date： June 16, 2024
# 文件名称 Filename： 03-main_improved.py
# 编码实现 Coding by： Hongjie Liu , Suiwen Zhang 邮箱 Mailbox：redsocks1043@163.com
# 所属单位：中国 成都，西南民族大学（Southwest University of Nationality，or Southwest Minzu University）, 计算机科学与工程学院.
# 指导老师：周伟老师
# coding=utf-8
import time
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy import stats

start_time = time.time()

# 特征标准化
scaler = StandardScaler()
# 加载数据集
train_dataSet = pd.read_csv(r'C:/Users/admin/PycharmProjects/PythonProject2/001-基准算法/modified_数据集Time_Series661_detail.dat')
test_dataSet = pd.read_csv(r'C:/Users/admin/PycharmProjects/PythonProject2/001-基准算法/modified_数据集Time_Series662_detail.dat')

# columns表示原始列，noise_columns表示添加噪声的列
columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

CL = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth',
      'Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
      'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

# 数据预处理（与原始代码相同）
data = train_dataSet[CL]
missingDf = data.isnull().sum().sort_values(ascending=False).reset_index()
missingDf.columns = ['feature', 'miss_num']
missingDf['miss_percentage'] = missingDf['miss_num'] / data.shape[0]
print("缺失值比例")
print(missingDf)

# 异常值检测（与原始代码相同）
outlier_ratios = {}
for column in CL:
    z_scores = np.abs(stats.zscore(train_dataSet[column]))
    outliers = (z_scores > 2)
    outlier_ratio = outliers.mean()
    outlier_ratios[column] = outlier_ratio

print("*" * 30)
print("异常值的比例:")
for column, ratio in outlier_ratios.items():
    print(f"{column}: {ratio:.2%}")

# 划分训练集和测试集
X_train = train_dataSet[noise_columns]
y_train = train_dataSet[columns]
X_test = test_dataSet[noise_columns]
y_test = test_dataSet[columns]


# 改进的深度森林模型
class ImprovedDeepForest:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.models = []

    def fit(self, X, y):
        # 第一层：多种树模型
        rf1 = RandomForestRegressor(n_estimators=self.n_estimators,
                                    max_depth=self.max_depth,
                                    random_state=self.random_state)
        rf2 = RandomForestRegressor(n_estimators=self.n_estimators,
                                    max_depth=self.max_depth,
                                    random_state=self.random_state + 1)
        et1 = ExtraTreesRegressor(n_estimators=self.n_estimators,
                                  max_depth=self.max_depth,
                                  random_state=self.random_state)

        self.models = [rf1, rf2, et1]

        # 使用K折交叉验证生成特征
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        X_enhanced_train = np.zeros((X.shape[0], len(self.models) * y.shape[1]))

        for train_idx, val_idx in kf.split(X):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]

            for i, model in enumerate(self.models):
                model.fit(X_train_fold, y_train_fold)
                preds = model.predict(X_val_fold)
                X_enhanced_train[val_idx, i * y.shape[1]:(i + 1) * y.shape[1]] = preds

        # 第二层：元学习器
        self.meta_learner = RandomForestRegressor(n_estimators=self.n_estimators,
                                                  max_depth=self.max_depth,
                                                  random_state=self.random_state)

        # 添加原始特征
        X_final = np.hstack([X_enhanced_train, X.values])
        self.meta_learner.fit(X_final, y)

        # 重新训练第一层模型在整个训练集上
        for model in self.models:
            model.fit(X, y)

    def predict(self, X):
        # 生成第一层预测特征
        X_enhanced_test = np.zeros((X.shape[0], len(self.models) * y_train.shape[1]))

        for i, model in enumerate(self.models):
            preds = model.predict(X)
            X_enhanced_test[:, i * y_train.shape[1]:(i + 1) * y_train.shape[1]] = preds

        # 添加原始特征并进行最终预测
        X_final = np.hstack([X_enhanced_test, X.values])
        return self.meta_learner.predict(X_final)


# 训练改进的深度森林模型
print("开始训练改进的深度森林模型...")
deep_forest = ImprovedDeepForest(n_estimators=150, max_depth=10, random_state=217)
deep_forest.fit(X_train, y_train)

# 预测
y_predict = deep_forest.predict(X_test)

# 结果处理和保存
results = []
for True_Value, Predicted_Value in zip(y_test.values, y_predict):
    error = np.abs(True_Value - Predicted_Value)
    formatted_true_value = ' '.join(map(str, True_Value))
    formatted_predicted_value = ' '.join(map(str, Predicted_Value))
    formatted_error = ' '.join(map(str, error))
    results.append([formatted_true_value, formatted_predicted_value, formatted_error])

# 保存结果
result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_ImprovedDeepForest.csv", index=False)

print("<*>" * 50)

# 计算平均误差
data = pd.read_csv("result_ImprovedDeepForest.csv")
column3 = data.iloc[:, 2]
numbers = column3.str.split(' ', expand=True).apply(pd.to_numeric)
means = numbers.mean()

print("6个数据的平均值为：\n", means)
print(f"总体平均误差: {means.mean():.6f}")

end_time = time.time()
print(f"总耗时：{end_time - start_time:.3f}秒")