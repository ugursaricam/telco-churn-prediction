from helper_functions import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from scipy.stats import norm, skew
from scipy import stats
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

df = pd.read_csv('datasets/Telco-Customer-Churn.csv')

###################################
# Exploratory data analysis
###################################

df.head()
df.shape
df.isnull().sum()
df.info()

df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

df.info()
df.isnull().sum()

df.iloc[df[df["TotalCharges"].isnull()].index, 19] = df[df["TotalCharges"].isnull()]["MonthlyCharges"]

df.describe().T

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)

cat_cols
num_cols
cat_but_car

df.head()

###################################
# Visualisations
###################################

for col in cat_cols:
    cat_summary(df, col, plot=True)

plt.figure(figsize = (5, 5))
myexplode = (0.1, 0)
churn = df['Churn'].value_counts()
churn.plot(kind='pie', explode = myexplode, shadow = True, autopct = "%1.1f%%")
plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)

for col in cat_cols:
    target_summary_with_cat(df, 'Churn', col, plot=True)

for col in num_cols:
    target_summary_with_num(df, 'Churn', col, plot=True)

###################################
# Outliers
###################################

for col in num_cols:
    print(col, ':', check_outlier(df, col))

for col in num_cols:
    boxplot_outliers(df, col)

###################################
# Missing values and corr
###################################

missing_values_table(df)

df_corr(df)

df.corrwith(df["Churn"]).sort_values(ascending=False)

###################################
# Visualisations
###################################

for col in num_cols:
    plt.figure()
    sns.distplot(df[col], fit = norm)
    plt.show(block=True)

###################################
# Base ML Model
###################################

base_df = df.copy()

cat_cols = [col for col in cat_cols if col not in ["Churn"]]

base_df = one_hot_encoder(base_df, cat_cols)

y = base_df['Churn']
X = base_df.drop(['Churn', 'customerID'], axis=1)

SEED = 42

models = [('LR', LogisticRegression(random_state=SEED)),
          # ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=SEED)),
          ('RF', RandomForestClassifier(random_state=SEED)),
          ('XGB', XGBClassifier(random_state=SEED)),
          ("LightGBM", LGBMClassifier(random_state=SEED)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=SEED))]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

# ########## LR ##########
# Accuracy: 0.8031
# Auc: 0.8417
# Recall: 0.5383
# Precision: 0.6576
# F1: 0.5919
# ########## CART ##########
# Accuracy: 0.7265
# Auc: 0.6569
# Recall: 0.5056
# Precision: 0.4856
# F1: 0.4951
# ########## RF ##########
# Accuracy: 0.7923
# Auc: 0.8245
# Recall: 0.4853
# Precision: 0.6455
# F1: 0.5538
# ########## XGB ##########
# Accuracy: 0.7886
# Auc: 0.827
# Recall: 0.5131
# Precision: 0.6263
# F1: 0.5631
# ########## LightGBM ##########
# Accuracy: 0.7982
# Auc: 0.8373
# Recall: 0.5281
# Precision: 0.6482
# F1: 0.5816
# ########## CatBoost ##########
# Accuracy: 0.7968
# Auc: 0.8406
# Recall: 0.5083
# Precision: 0.6511
# F1: 0.5705

###################################
# Feature extraction
###################################

# Feature based on the customer's gender and senior
df.loc[((df['gender'] == 'Male') & (df["SeniorCitizen"]== 1)), 'SENIOR/YOUNG_GENDER'] ="senior_male"
df.loc[((df['gender'] == 'Male') & (df["SeniorCitizen"]== 0)), 'SENIOR/YOUNG_GENDER'] ="young_male"
df.loc[((df['gender'] == 'Female') & (df["SeniorCitizen"]== 1)), 'SENIOR/YOUNG_GENDER'] ="senior_female"
df.loc[((df['gender'] == 'Female') & (df["SeniorCitizen"]== 0)), 'SENIOR/YOUNG_GENDER'] ="young_female"
df.groupby("SENIOR/YOUNG_GENDER").agg({"Churn": ["mean","count"]})

# Creating annual categorical variable from Tenure variable
df.loc[(df["tenure"]>=0) & (df["tenure"]<=12),"NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["tenure"]>12) & (df["tenure"]<=24),"NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"]>24) & (df["tenure"]<=36),"NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"]>36) & (df["tenure"]<=48),"NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"]>48) & (df["tenure"]<=60),"NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"]>60) & (df["tenure"]<=72),"NEW_TENURE_YEAR"] = "5-6 Year"
df.groupby("NEW_TENURE_YEAR").agg({"Churn": ["mean","count"]})

# Specify 1 or 2 year contract customers as Engaged
df["NEW_Engaged"] = df["Contract"].apply(lambda x: 1 if x in ["One year","Two year"] else 0)
df.groupby("NEW_Engaged").agg({"Churn": ["mean","count"]})

# Customers who do not receive any support, backup or protection
df["NEW_noProt"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (x["TechSupport"] != "Yes") else 0, axis=1)
df.groupby("NEW_noProt").agg({"Churn": ["mean","count"]})

# Customers with monthly contracts and young
df["NEW_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0, axis=1)
df.groupby("NEW_Young_Not_Engaged").agg({"Churn": ["mean","count"]})

# Total number of services received by the customer
df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                               'OnlineBackup', 'DeviceProtection', 'TechSupport',
                               'StreamingTV', 'StreamingMovies']]== 'Yes').sum(axis=1)
df.groupby("NEW_TotalServices").agg({"Churn": ["mean","count"]})

# Customers who buy any streaming service
df["NEW_FLAG_ANY_STREAMING"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)
df.groupby("NEW_FLAG_ANY_STREAMING").agg({"Churn": ["mean","count"]})

# Does the customer make automatic payments?
df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(lambda x: 1 if x in ["Bank transfer (automatic)","Credit card (automatic)"] else 0)
df.groupby("NEW_FLAG_AutoPayment").agg({"Churn": ["mean","count"]})

# Average monthly payment
df["NEW_AVG_Charges"] = df["TotalCharges"] / (df["tenure"] + 1)
df["NEW_AVG_Charges"].mean()

# Customers who pay more than average pay
df["NEW_Over_AVG_Payment"] = df.apply(lambda x: 1 if (x["NEW_AVG_Charges"] > df["NEW_AVG_Charges"].mean()) else 0, axis=1)
df.groupby("NEW_Over_AVG_Payment").agg({"Churn": ["mean","count"]})

# Customers who do not have a contract and pay more than the average wage
df["NEW_not_Engaged_pay_more"] = df.apply(lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["MonthlyCharges"] > df["MonthlyCharges"].mean()) else 0, axis=1)
df.groupby("NEW_not_Engaged_pay_more").agg({"Churn": ["mean","count"]})

df.head()

###################################
# Encoding
###################################

binary_cols = binary_cols(df)
for col in binary_cols:
    df = label_encoder(df, col)

df.head()

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols)

df.shape

#############################################
# New ML model
#############################################

y = df['Churn']
X = df.drop(['Churn', 'customerID'], axis=1)

SEED = 42

models = [('LR', LogisticRegression(random_state=SEED)),
          # ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=SEED)),
          ('RF', RandomForestClassifier(random_state=SEED)),
          ('XGB', XGBClassifier(random_state=SEED)),
          ("LightGBM", LGBMClassifier(random_state=SEED)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=SEED))]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

# ########## LR ##########
# Accuracy: 0.8045
# Auc: 0.8461
# Recall: 0.5335
# Precision: 0.6648
# F1: 0.5915
# ########## CART ##########
# Accuracy: 0.7324
# Auc: 0.6605
# Recall: 0.5024
# Precision: 0.495
# F1: 0.4983
# ########## RF ##########
# Accuracy: 0.7924
# Auc: 0.8243
# Recall: 0.4955
# Precision: 0.6419
# F1: 0.559
# ########## XGB ##########
# Accuracy: 0.7897
# Auc: 0.8254
# Recall: 0.5174
# Precision: 0.626
# F1: 0.5664
# ########## LightGBM ##########
# Accuracy: 0.7936
# Auc: 0.8373
# Recall: 0.5158
# Precision: 0.6389
# F1: 0.5703
# ########## CatBoost ##########
# Accuracy: 0.7972
# Auc: 0.8416
# Recall: 0.5153
# Precision: 0.6498
# F1: 0.5744

#############################################
# Hyperparameter tuning
#############################################

# Random Forests
rf_model = RandomForestClassifier(random_state=17)
rf_model.get_params()
rf_params = {"max_depth": [5, 8, None], # Ağacın maksimum derinliği
             "max_features": [3, 5, 7, "auto"], # En iyi bölünmeyi ararken göz önünde bulundurulması gereken özelliklerin sayısı
             "min_samples_split": [2, 5, 8, 15, 20], # Bir node'u bölmek için gereken minimum örnek sayısı
             "n_estimators": [100, 200, 500]} # Ağaç sayısı

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
rf_best_grid.best_params_ # {'max_depth': 8, 'max_features': 7, 'min_samples_split': 20, 'n_estimators': 100}
# rf_final = RandomForestClassifier(max_depth=8, max_features=7, min_samples_split=20, n_estimators=100, random_state=17).fit(X, y)
rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

# XGBoost
xgboost_model = XGBClassifier(random_state=17)
xgboost_model.get_params()
xgboost_params = {"learning_rate": [0.1, 0.01, 0.001],
                  "max_depth": [5, 8, 12, 15, 20],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.5, 0.7, 1]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
xgboost_best_grid.best_params_ # {'colsample_bytree': 0.5, 'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 500}
xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

# LightGBM
lgbm_model = LGBMClassifier(random_state=17)
lgbm_model.get_params()
lgbm_params = {"learning_rate": [0.01, 0.1, 0.001],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
lgbm_best_grid.best_params_ # {'colsample_bytree': 0.5, 'learning_rate': 0.01, 'n_estimators': 500}
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

# CatBoost
catboost_model = CatBoostClassifier(random_state=17, verbose=False)
catboost_model.get_params()
catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
catboost_best_grid.best_params_ # {'depth': 3, 'iterations': 500, 'learning_rate': 0.01}
catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

#############################################
# Comparison of models after hyperparameter tuning
#############################################

final_models = [('RF', rf_final),
                ('XGB', xgboost_final),
                ("LightGBM", lgbm_final),
                ("CatBoost", catboost_final)]

for name, model in final_models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

# ########## RF ##########
# Accuracy: 0.8042
# Auc: 0.8467
# Recall: 0.5137
# Precision: 0.6718
# F1: 0.582
# ########## XGB ##########
# Accuracy: 0.8031
# Auc: 0.8474
# Recall: 0.5206
# Precision: 0.6655
# F1: 0.5837
# ########## LightGBM ##########
# Accuracy: 0.8026
# Auc: 0.8457
# Recall: 0.519
# Precision: 0.6651
# F1: 0.5826
# ########## CatBoost ##########
# Accuracy: 0.8035
# Auc: 0.8473
# Recall: 0.4992
# Precision: 0.6768
# F1: 0.5741


# BASE MODEL
# ########## RF ##########
# Accuracy: 0.7923
# Auc: 0.8245
# Recall: 0.4853
# Precision: 0.6455
# F1: 0.5538

# AFTER FEATURE ENGINEERING
# ########## RF ##########
# Accuracy: 0.7924
# Auc: 0.8243
# Recall: 0.4955
# Precision: 0.6419
# F1: 0.559

# AFTER HYPERPARAMETER TUNING
# ########## RF ##########
# Accuracy: 0.8042
# Auc: 0.8467
# Recall: 0.5137
# Precision: 0.6718
# F1: 0.582

# BASE MODEL
# ########## XGB ##########
# Accuracy: 0.7886
# Auc: 0.827
# Recall: 0.5131
# Precision: 0.6263
# F1: 0.5631

# AFTER FEATURE ENGINEERING
# ########## XGB ##########
# Accuracy: 0.7897
# Auc: 0.8254
# Recall: 0.5174
# Precision: 0.626
# F1: 0.5664

# AFTER HYPERPARAMETER TUNING
# ########## XGB ##########
# Accuracy: 0.8031
# Auc: 0.8474
# Recall: 0.5206
# Precision: 0.6655
# F1: 0.5837

# BASE MODEL
# ########## LightGBM ##########
# Accuracy: 0.7982
# Auc: 0.8373
# Recall: 0.5281
# Precision: 0.6482
# F1: 0.5816

# AFTER FEATURE ENGINEERING
# ########## LightGBM ##########
# Accuracy: 0.7936
# Auc: 0.8373
# Recall: 0.5158
# Precision: 0.6389
# F1: 0.5703

# AFTER HYPERPARAMETER TUNING
# ########## LightGBM ##########
# Accuracy: 0.8026
# Auc: 0.8457
# Recall: 0.519
# Precision: 0.6651
# F1: 0.5826

# BASE MODEL
# ########## CatBoost ##########
# Accuracy: 0.7968
# Auc: 0.8406
# Recall: 0.5083
# Precision: 0.6511
# F1: 0.5705

# AFTER FEATURE ENGINEERING
# ########## CatBoost ##########
# Accuracy: 0.7972
# Auc: 0.8416
# Recall: 0.5153
# Precision: 0.6498
# F1: 0.5744

# AFTER HYPERPARAMETER TUNING
# ########## CatBoost ##########
# Accuracy: 0.8035
# Auc: 0.8473
# Recall: 0.4992
# Precision: 0.6768
# F1: 0.5741

################################################
# Feature Importance
################################################

plot_importance(rf_final, X)
plot_importance(xgboost_final, X)
plot_importance(lgbm_final, X)
plot_importance(catboost_final, X)

