# Developed on Python 3.5 64-bit, Windows 10 64-bit
# Christopher Lang
# STA9792, Fall 2017
# Final Project
import csv
import os
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from datetime import datetime as dt
import numpy as np
import copy
import pandas as pd
from tqdm import tqdm
import itertools
import pdb


os.chdir('C:/Users/Christopher Lang/Dropbox/Education/Baruch College/Fall 2017/STA 9792 - Advanced Data Analysis/final project')

# Define a few convenience function for piping data processing
def p_str2date(df, column, fmt='%m/%d/%Y'):
    df[column] = df[column].map(lambda x: dt.strptime(x, fmt).date())

    return df


def p_growth(df, column, new_column, periods=1, sort_column=None):
    if sort_column is not None:
        df = df.sort_values(sort_column)

    df[new_column] = df[column] / df[column].shift(periods=periods)

    return df


def p_demean(df, columns, new_columns=None):
    if isinstance(columns, str):
        if new_columns is not None:
            df[new_columns] = df[columns] - np.mean(df[columns])
        else:
            df[columns] = df[columns] - np.mean(df[columns])

    elif isinstance(columns, list) or isinstance(columns, tuple):
        if new_columns is not None:
            for acol, ancol in zip(columns, new_columns):
                df[ancol] = df[acol] - np.mean(df[acol])
        else:
            for acol in columns:
                df[acol] = df[acol] - np.mean(df[acol])

    return df

# Load data for all questions =================================================
ffactors = (
    pd.read_csv("data/original/projectdata_famafrench.csv").
    pipe(p_str2date, column='Date').
    sort_values('Date').
    pipe(p_demean, columns=['Mkt-RF', 'SMB', 'HML'],
         new_columns=['dm_mkt_rf', 'dm_SMB', 'dm_HML'])
)
assetprices = (
    pd.read_csv("data/original/projectdata_dailyStockData.csv").
    pipe(p_str2date, column='Date').
    sort_values('Date')
)
riskfree = (
    pd.read_csv("data/original/projectdata_dailyStockData_riskfree.csv").
    pipe(p_str2date, column='Date').
    sort_values('Date')
)

impact = (
    pd.read_csv('data/original/projectdata_impact.csv')
)

riskfree['spy_lnreturn_rf'] = riskfree['SPY_Ret'] - riskfree['RF']
riskfree = riskfree[riskfree['spy_lnreturn_rf'].notnull()]

stocktick = [i for i in assetprices.columns if i != 'Date']

# Question 1 ==================================================================
stock_betas = list()
for astock in stocktick:
    astock_price = (
        assetprices[['Date', astock]].copy().
        pipe(p_growth, column=astock, new_column='returns')
    )

    astock_price = astock_price[astock_price.returns.notnull()]
    astock_price['lnreturn'] = astock_price.returns.map(lambda x: np.log(x))

    rf = riskfree[['Date', 'spy_lnreturn_rf', 'RF']]
    astock_price = pd.merge(astock_price, rf, how='left', on='Date')

    astock_price['lnreturn_rf'] = astock_price['lnreturn'] - astock_price['RF']

    design_matrix = astock_price['spy_lnreturn_rf'].as_matrix().reshape(-1, 1)
    response_vector = astock_price['lnreturn_rf'].as_matrix().reshape(-1, 1)

    capm_model = LinearRegression()
    capm_model.fit(design_matrix, response_vector)

    r = {
        'stock': astock,
        'beta1': capm_model.coef_[0][0],
        'beta0': capm_model.intercept_[0]
    }

    stock_betas.append(r)

stock_betas.sort(key=lambda x: x['stock'])

with open('data/output/question1_stock_betas.csv', 'w') as f:
    csvwriter = csv.DictWriter(f, ['stock', 'beta1', 'beta0'],
                               lineterminator='\n',
                               delimiter=',')

    csvwriter.writeheader()
    csvwriter.writerows(stock_betas)

# Question 1.2
# Calculate expect annual return of a stock
expected_returns = list()

for a_stock_beta in stock_betas:
    R_f = np.float64(0.02)
    Ex_R_m = np.float64(0.10)

    expect_rt = R_f + a_stock_beta['beta1'] * (Ex_R_m - R_f)

    r = {
        "stock": a_stock_beta['stock'],
        "expected_annual_return": expect_rt
    }

    expected_returns.append(r)

expected_returns.sort(key=lambda x: x['stock'])

with open('data/output/question1_stock_expected_return.csv', 'w') as f:
    csvwriter = csv.DictWriter(f, ['stock', 'expected_annual_return'],
                               lineterminator='\n',
                               delimiter=',')

    csvwriter.writeheader()
    csvwriter.writerows(expected_returns)

# Question 2
stock_betas = dict()
stock_betas['stock'] = list()
stock_betas['beta1'] = list()
stock_betas['beta2'] = list()
stock_betas['beta3'] = list()
stock_betas['beta0'] = list()
cov_mat_values = list()
for astock in stocktick:
    astock_price = (
        assetprices[['Date', astock]].copy().
        pipe(p_growth, column=astock, new_column='returns')
    )

    astock_price = astock_price[astock_price.returns.notnull()]
    astock_price['lnreturn'] = astock_price.returns.map(lambda x: np.log(x))

    astock_price = pd.merge(astock_price, ffactors[['Date', 'RF']], how='left',
                            on='Date')

    astock_price['lnreturn_rf'] = astock_price['lnreturn'] - astock_price['RF']

    astock_price = astock_price[['Date', 'lnreturn_rf']]

    fff = ffactors[['Date', 'dm_mkt_rf', 'dm_SMB', 'dm_HML']].copy()

    model_frame = pd.merge(fff, astock_price, on='Date', how='inner')
    model_frame = model_frame[['lnreturn_rf', 'dm_mkt_rf', 'dm_SMB', 'dm_HML']]

    model = LinearRegression()
    model.fit(model_frame[['dm_mkt_rf', 'dm_SMB', 'dm_HML']],
              model_frame['lnreturn_rf'])

    stock_betas['stock'].append(astock)
    stock_betas['beta1'].append(model.coef_[0])
    stock_betas['beta2'].append(model.coef_[1])
    stock_betas['beta3'].append(model.coef_[2])
    stock_betas['beta0'].append(model.intercept_)

stock_betas = pd.DataFrame(stock_betas)
beta_matrix = stock_betas.copy()
del beta_matrix['stock']
del beta_matrix['beta0']

beta_matrix = np.matrix(beta_matrix.as_matrix())

ff_cov = np.cov(ffactors[['dm_mkt_rf', 'dm_SMB', 'dm_HML']], rowvar=False)
ff_cov = np.matrix(ff_cov)

stock_cov = beta_matrix * ff_cov * beta_matrix.transpose() * np.float64(250)

# Question 3 ==================================================================
# Building a neural network to estimate stock impact

# Use this function to random decide how many layers, and how many nodes
# for testing model performance
def gen_layers(max_layer=3, max_nodes=10, times=10):
    result = list()

    for _ in range(times):
        nlayers = np.random.randint(1, max_layer + 1)
        result.append(tuple(np.random.randint(1, max_nodes + 1, nlayers)))

    return result


design_matrix = impact[['Size', "Volatility", 'POV']].as_matrix()
outcome = impact[['Cost']].as_matrix().ravel()

# For testing, we're going to use a 10-fold Cross Validation for
# selecting the Neural Network Parameters
matrix_split = KFold(n_splits=10, shuffle=True)
kfold_indices = matrix_split.split(design_matrix, outcome)

layer_fold_iterator = itertools.product(gen_layers(), kfold_indices)
layer_fold_iterator = list(layer_fold_iterator)

mlp_result = list()
for layer_struct, a_kfold in tqdm(layer_fold_iterator, ncols=80):
    training_indices = a_kfold[0]
    testing_indices = a_kfold[1]

    training_set = design_matrix[training_indices]
    training_outome = outcome[training_indices]
    testing_set = design_matrix[testing_indices]
    testing_outcome = outcome[testing_indices]

    mlp_model = MLPRegressor(layer_struct, max_iter=10000)
    mlp_model.fit(training_set, training_outome)

    predicted = mlp_model.predict(testing_set)

    mse = np.sum(np.power(predicted - testing_outcome, 2))

    mlp_result.append({'layer': layer_struct, 'mse': mse})

mlp_mse = list()
mlp_result.sort(key=lambda x: x['layer'])
for layer, group in itertools.groupby(mlp_result, lambda x: x['layer']):
    mse = np.mean([i['mse'] for i in group])
    mlp_mse.append((layer, mse))

mlp_mse.sort(key=lambda x: x[1])

best_layer = mlp_mse[0][0]  # grab layer structure

mlp_model = MLPRegressor(best_layer, max_iter=10000)
mlp_model.fit(design_matrix, outcome)

# For question 3.1, please see Excel book:
# Christopher Lang - Final Project Question partial Question 3.xlsm
a1 = 595.05
a2 = 0.33
a3 = 0.54
a4 = 0.96
b1 = 0.92

# Estimate MI using Istar model
size = impact['Size'].as_matrix().ravel()
volatility = impact['Volatility'].as_matrix().ravel()
pov = impact['POV'].as_matrix().ravel()
cost = impact['Cost'].as_matrix().ravel()

istar = a1 * np.power(size, a2) * np.power(volatility, a3)
mi_istar = b1 * istar * np.power(pov, a4) * (1 - b1) * istar

# Calculate Istar error
mi_istar_errorsd = np.sqrt(np.mean(np.power(mi_istar - cost, 2)))
mi_istar_sse = np.sum(np.power(mi_istar - cost, 2))

# Calculate NN error
predicted_cost = mlp_model.predict(design_matrix)
nn_errorsd = np.sqrt(np.mean(np.power(predicted_cost - cost, 2)))
nn_sse = np.sum(np.power(predicted_cost - cost, 2))

# NN had much lower error, SD
# istar SSE = 5757544593.1575203, NN SSE = 2748600.8163455906
# NN SD = 675.97927748114455, NN SD = 14.769662214559604

# Question 4 ==================================================================
{
    impact.groupby('Stock').
    agg({'Volatility': {'volatility_mean': 'mean'}}).
    to_csv('data/output/question4_stock_volatility.csv')
}

{
    pd.melt(assetprices, id_vars=['Date'], var_name='Stock', value_name='Price').
    sort_values('Date').groupby(['Stock']).
    agg({"Price": {"price": "last"}}).
    to_csv('data/output/question4_stockprice.csv')
}

pd.DataFrame(stock_cov).to_csv("data/output/question4_covariance_matrix.csv")

# Get MI from NN
cost = impact.query('Date == "8/31/2017"')
cost = cost[['Size', 'Volatility', 'POV']].as_matrix()

cost = pd.Series(mlp_model.predict(cost)).reset_index(drop=True)
stock_names = impact.query('Date == "8/31/2017"')[['Stock']].reset_index(drop=True)

pd.concat([stock_names, cost], 1).to_csv("data/output/question4_stock_MI.csv")
