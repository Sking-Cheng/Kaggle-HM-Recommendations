# Kaggle-H&M-Recommendations
H&amp;M Personalized Fashion Recommendations Solution (Original rank 🥉 170/2952)
## 方案思路和trick

- 只采用了数据集中的表格数据部分，并没有使用图像数据和文本数据，最终使用的模型是Catboost。
- 由于pandas读取csv文件数据较慢，在一开始就将所有表格数据转为了pickle格式保存，并在之后直接读取pkl文件，这让数据加载的时间**缩短了10倍以上**
- 特征工程部分是本次比赛的重点，主要通过以下四个步骤构建新的特征：
  1. 创建了user-item矩阵，并使用**LightFM库**，训练了**user的embedding**，作为后续模型输入的特征。
  2. 对各个商品的属性特征做**onehot编码**, 接着并入交易表, 然后groupby users聚合，构建了**user和商品属性之间的关系特征**
  3. 通过上述新特征，生成了**candidates**候选项，并且衍生出**少量rank特征**。
- 之后数据集切分上，将最后一周作为验证集，其余周做为训练集，使用**Catboost**进行训练

 

## Model

```python
train_dataset = catboost.Pool(data=train[feature_columns], label=train['y'], group_id=train['query_group'], cat_features=cat_features)
valid_dataset = catboost.Pool(data=valid[feature_columns], label=valid['y'], group_id=valid['query_group'], cat_features=cat_features)

params = {
    'loss_function': 'YetiRank',
    'use_best_model': True,
    'one_hot_max_size': 300,
    'iterations': 10000,
}
model = catboost.CatBoost(params) 
model.fit(train_dataset, eval_set=valid_dataset) 

plt.plot(model.get_evals_result()['validation']['PFound'])

feature_importance = model.get_feature_importance(train_dataset)
sorted_idx = np.argsort(feature_importance)
plt.figure(figsize=(8, 16))
plt.yticks(range(len(feature_columns)), np.array(feature_columns)[sorted_idx])
plt.barh(range(len(feature_columns)), feature_importance[sorted_idx])
```



## 最佳参数

```python
params = {
  'objective': 'regression',
  'boosting_type': 'gbdt',
  'n_jobs': -1, 
  'verbose': -1, 
  'seed': SEED,
  'feature_fraction_seed': SEED,
  'bagging_seed': SEED,
  'drop_seed': SEED,
  'data_random_seed': SEED,
  'max_bin':trial.suggest_int("max_bin", 80, 255), 
  'learning_rate': trial.suggest_loguniform("learning_rate", 0.003, 0.4), 
  'num_leaves': trial.suggest_int("num_leaves", 10, 400), 
  'max_depth': trial.suggest_int("max_depth", 3, 64), 
  'min_child_samples':trial.suggest_int("min_child_samples", 16, 600), 
  'min_child_weight':trial.suggest_uniform("min_child_weight", 7e-4, 2e-2),
  'feature_fraction': trial.suggest_discrete_uniform("feature_fraction", 0.05, 0.8, 0.1), 
  'feature_fraction_bynode': trial.suggest_discrete_uniform("feature_fraction_bynode", 0.2, 0.9, 0.1), 
  'bagging_fraction': trial.suggest_discrete_uniform("bagging_fraction", 0.2, 1.0, 0.1), 
  'bagging_freq': trial.suggest_int('bagging_freq', 10, 100), 
  'reg_alpha': trial.suggest_categorical("reg_alpha",  [0, 0.001, 0.01, 0.03, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]), 
  'reg_lambda': trial.suggest_categorical("reg_lambda",  [0, 0.001, 0.01, 0.03, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
}
```



## 冲榜历程

- **比赛结束前自己的方案**：
  1. 使用LightGBM Baseline训练模型。少量特征 + 粗糙的 candidates挑选，Public LB：0.0203
  2. 加入并调参 LightFM 生成user的embedding，Public LB：0.0231
  3. 加入商品属性特征onehot编码，并用user聚合，PB：0.0247   **(top6%🥉)**
- **比赛结束参考前排方案进行改进之后**：
  1. 增加了repurchase等多个candidates条件，Public LB：0.0271
  2. 增加了user-item对的静态特征和动态特征，Public LB：0.0285
  3. 使用Catboost 并调参，PB：0.0309
  4. 增加了user、item、user-item对新鲜度特征，Public LB：0.0315+



## 代码、数据集

- 代码
  - HM_get_data.ipynb # 生成pickle特征
  - HM_feature.ipynb # 特征工程
  - HM_train_infer # 训练和推理
- 数据集
  - [官方数据集](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)



## 写在后面

有一说一，SOLO打比赛是真的累🥺决定参加[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)时离比赛结束只有十几天⏰，同时还要兼顾考研复习🎯，因此特征工程和超参调优都比较粗糙，参赛阶段借鉴了较多的开源的kernel，虽然拿到了铜牌🥉但是离前排大神还有非常大的差距。赛后大佬们分享的思路对后续的提升型复盘有很大的帮助，此代码也是集合一些大神思路后的最终结果，没有写注释，若有幸垂阅🐭🐭的**shitcode**，并觉得对你有点帮助的话，请点一个**⭐**，🐭🐭**将感激不尽捏!!!**
