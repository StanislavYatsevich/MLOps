import click
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import math


def get_weights(lst, deg):
    return [deg ** i for i in range(len(lst))]


def make_lags(ts, start, end):
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(start, end + 1)
        },
        axis=1)


def replace_neg_values(lst):
    for i in range(len(lst)):
        lst[i] = 0 if lst[i] < 0 else lst[i]
    return lst


class Dataset:

    def __init__(self, train_link, test_link):
        self.train_data = pd.read_csv(train_link)
        self.test_data = pd.read_csv(test_link)

    def prepare_data(self, deg):
        self.train_data.drop('Unnamed: 0', axis=1, inplace=True)
        self.train_data = self.train_data[['target', 'year', 'month']].groupby(['year', 'month']).sum()
        self.train_data.reset_index(inplace=True)

        self.test_data.drop('Unnamed: 0', axis=1, inplace=True)
        self.test_data = self.test_data[['target', 'year', 'month']].groupby(['year', 'month']).sum()
        self.test_data.reset_index(inplace=True)

        if self.train_data.month[len(self.train_data) - 1] == self.test_data.month[0]:
            self.train_data.target[len(self.train_data) - 1] += self.test_data.target[0]
            self.test_data = self.test_data[1:]

        train_years = [i for i in range(self.train_data.year.min(), self.train_data.year.max() + 1)]
        months = [i for i in range(1, 13)]
        theory_years_months = []

        for i in train_years:
            for j in months:
                theory_years_months.append((i, j))
        theory_years_months = theory_years_months[
                              self.train_data.month[0] - 1: -(12 - self.train_data.month[len(self.train_data) - 1])]

        fact_years_months = []
        for i in range(len(self.train_data.year)):
            fact_years_months.append((self.train_data.year[i], self.train_data.month[i]))

        restored_months = []
        missed_months = list(set(theory_years_months) - set(fact_years_months))
        for i in range(len(missed_months)):
            year = missed_months[i][0]
            month = missed_months[i][1]
            target = np.round(np.average(
                self.train_data[(self.train_data['month'] == month) & (self.train_data['year'] <= year - 1)]['target'],
                weights=get_weights(self.train_data[(self.train_data['month'] == month) &
                                                    (self.train_data['year'] <= year - 1)]['target'], deg)), 1)
            restored_months.append((year, month, target))

        for i in restored_months:
            self.train_data.loc[len(self.train_data.index)] = i

        self.train_data = self.train_data.sort_values(by=['year', 'month'])
        self.train_data['year'] = self.train_data['year'].astype('int')
        self.train_data['month'] = self.train_data['month'].astype('int')

        self.train_data.reset_index(inplace=True)
        self.train_data.drop(['index'], inplace=True, axis=1)
        self.test_data.index = [i for i in range(len(self.train_data), len(self.train_data) + len(self.test_data))]

        months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        seasons = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 1]
        self.train_data['season'] = self.train_data['month'].map(dict(zip(months, seasons)))
        self.test_data['season'] = self.test_data['month'].map(dict(zip(months, seasons)))

    def get_sales(self):
        a = self.train_data.copy()
        dates = []
        for i in range(len(a.year)):
            dates.append(datetime(a.year[i], a.month[i], 1))
        a.index = dates
        a.drop(['year', 'month', 'season'], inplace=True, axis=1)

        fig = px.line(a)
        fig.update_layout(width=1000, title_text='Month sales')
        fig.show()

    def get_sales_diff(self):
        a = self.train_data.copy()
        dates = []
        for i in range(len(a.year)):
            dates.append(datetime(a.year[i], a.month[i], 1))
        a.index = dates
        a.drop(['year', 'month', 'season'], inplace=True, axis=1)

        fig = px.line(a.diff())
        fig.update_layout(width=1000, title_text='Diff Month sales')
        fig.show()

    def get_seasonal_decompose(self):
        seas_decomp = seasonal_decompose(self.train_data['target'], period=12)
        plt.rc('figure', figsize=(14, 8))
        plt.rc('font', size=15)
        seas_decomp.plot();

    def get_acf(self, lags):
        plot_acf(self.train_data['target'], lags=lags);

    def get_pacf(self, lags):
        plot_pacf(self.train_data['target'], lags=lags);

    def get_qqplot(self):
        qqplot_data = qqplot(self.train_data['target'], line='s')

    def get_arima_predicts(self, order, diff):
        if not diff:
            y_train = self.train_data['target']
            y_test = self.test_data['target']
            df = pd.concat([y_train, y_test])

            arima = ARIMA(self.train_data['target'], order=order)
            y_pred = arima.fit().forecast(len(y_test))
            predict = pd.concat([y_train, y_pred])

            df = pd.concat([df, predict], axis=1)
            df.columns = ['Actual Sales', 'Predicted Sales']

            fig = px.line(df, y=df.columns, title='ARIMA Predictions')
            fig.show()

            print('MAE: ' + str(np.round(mean_absolute_error(y_test, y_pred), 2)))
            print('MAPE: ' + str(100 * np.round(mean_absolute_percentage_error(y_test, y_pred), 2)) + '%')

        else:
            a = pd.concat([self.train_data, self.test_data]).diff().fillna(0)
            true_target = pd.concat([self.train_data.target, self.test_data.target])

            y_train_diff = a.target[:-len(self.test_data)]
            y_test_diff = a.target[-len(self.test_data):]
            df = pd.concat([y_train_diff, y_test_diff])

            arima = ARIMA(y_train_diff, order=order)
            y_pred_diff = arima.fit().forecast(len(y_test_diff))
            predict_diff = pd.concat([y_train_diff, y_pred_diff])

            df = pd.concat([df, predict_diff, true_target], axis=1)
            df.columns = ['Actual Sales Diff', 'Predicted Sales Diff', 'True Target']

            pred_target = [0 for i in range(len(self.train_data) + len(self.test_data))]
            pred_target[0] = df['True Target'].iloc[0]

            for i in range(1, len(pred_target)):
                pred_target[i] = pred_target[i - 1] + df['Predicted Sales Diff'].iloc[i]

            pred_target = pd.DataFrame(pred_target)
            pred_target.index = [i for i in range(len(self.train_data) + len(self.test_data))]

            df = pd.concat([df, pred_target], axis=1)
            df.columns = ['Actual Sales Diff', 'Predicted Sales Diff', 'True Target', 'Predicted Target']
            df.drop(['Actual Sales Diff', 'Predicted Sales Diff'], inplace=True, axis=1)

            fig = px.line(df, y=df.columns, title='ARIMA Predictions')
            fig.show()

            print('MAE: ' + str(np.round(mean_absolute_error(df['True Target'][-len(self.test_data):],
                                                             df['Predicted Target'][-len(self.test_data):]), 2)))
            print('MAPE: ' + str(100 * np.round(mean_absolute_percentage_error(df['True Target'][-len(self.test_data):],
                                                                               df['Predicted Target'][
                                                                               -len(self.test_data):]), 2)) + '%')

    def get_sarima_predicts(self, order, seasonal_order, diff):
        if not diff:
            y_train = self.train_data['target']
            y_test = self.test_data['target']
            df = pd.concat([y_train, y_test])

            sarima = SARIMAX(self.train_data['target'], order=order, seasonal_order=seasonal_order)
            y_pred = sarima.fit().forecast(len(y_test))
            predict = pd.concat([y_train, y_pred])

            df = pd.concat([df, predict], axis=1)
            df.columns = ['Actual Sales', 'Predicted Sales']

            fig = px.line(df, y=df.columns, title='SARIMA Predictions')
            fig.show()

            print('MAE: ' + str(np.round(mean_absolute_error(y_test, y_pred), 2)))
            print('MAPE: ' + str(100 * np.round(mean_absolute_percentage_error(y_test, y_pred), 2)) + '%')

        else:
            a = pd.concat([self.train_data, self.test_data]).diff().fillna(0)
            true_target = pd.concat([self.train_data.target, self.test_data.target])

            y_train_diff = a.target[:-len(self.test_data)]
            y_test_diff = a.target[-len(self.test_data):]
            df = pd.concat([y_train_diff, y_test_diff])

            sarima = SARIMAX(y_train_diff, order=order, seasonal_order=seasonal_order)
            y_pred_diff = sarima.fit().forecast(len(y_test_diff))
            predict_diff = pd.concat([y_train_diff, y_pred_diff])

            df = pd.concat([df, predict_diff, true_target], axis=1)
            df.columns = ['Actual Sales Diff', 'Predicted Sales Diff', 'True Target']

            pred_target = [0 for i in range(len(self.train_data) + len(self.test_data))]
            pred_target[0] = df['True Target'].iloc[0]

            for i in range(1, len(pred_target)):
                pred_target[i] = pred_target[i - 1] + df['Predicted Sales Diff'].iloc[i]

            pred_target = pd.DataFrame(pred_target)
            pred_target.index = [i for i in range(len(self.train_data) + len(self.test_data))]

            df = pd.concat([df, pred_target], axis=1)
            df.columns = ['Actual Sales Diff', 'Predicted Sales Diff', 'True Target', 'Predicted Target']
            df.drop(['Actual Sales Diff', 'Predicted Sales Diff'], inplace=True, axis=1)

            fig = px.line(df, y=df.columns, title='SARIMA Predictions')
            fig.show()

            print('MAE: ' + str(np.round(mean_absolute_error(df['True Target'][-len(self.test_data):],
                                                             df['Predicted Target'][-len(self.test_data):]), 2)))
            print('MAPE: ' + str(100 * np.round(mean_absolute_percentage_error(df['True Target'][-len(self.test_data):],
                                                                               df['Predicted Target'][
                                                                               -len(self.test_data):]), 2)) + '%')

    def get_simple_exp_smoothing_predicts(self, alphas):
        df = pd.concat([self.train_data['target'], self.test_data['target']])

        model = SimpleExpSmoothing(self.train_data['target'])
        predictions1 = model.fit(smoothing_level=alphas[0]).forecast(len(self.test_data))
        predictions2 = model.fit(smoothing_level=alphas[1]).forecast(len(self.test_data))
        predictions3 = model.fit(smoothing_level=alphas[2]).forecast(len(self.test_data))

        df = pd.concat([df, pd.concat([self.train_data['target'], predictions1])], axis=1)
        df.columns = ['Actual target', f'Predicted target(alpha = {alphas[0]})']
        df = pd.concat([df, pd.concat([self.train_data['target'], predictions2])], axis=1)
        df.columns = ['Actual target', f'Predicted target(alpha = {alphas[0]})',
                      f'Predicted target(alpha = {alphas[1]})']
        df = pd.concat([df, pd.concat([self.train_data['target'], predictions3])], axis=1)
        df.columns = ['Actual target', f'Predicted target(alpha = {alphas[0]})',
                      f'Predicted target(alpha = {alphas[1]})', f'Predicted target(alpha = {alphas[2]})']

        fig = px.line(df, y=df.columns, title='Simple Exponential Smoothing Predictions')
        fig.show()

        mae1 = np.round(mean_absolute_error(df['Actual target'][-len(self.test_data):],
                                            df[f'Predicted target(alpha = {alphas[0]})'][-len(self.test_data):]), 2)
        mae2 = np.round(mean_absolute_error(df['Actual target'][-len(self.test_data):],
                                            df[f'Predicted target(alpha = {alphas[1]})'][-len(self.test_data):]), 2)
        mae3 = np.round(mean_absolute_error(df['Actual target'][-len(self.test_data):],
                                            df[f'Predicted target(alpha = {alphas[2]})'][-len(self.test_data):]), 2)
        mape1 = np.round(100 * mean_absolute_percentage_error(df['Actual target'][-len(self.test_data):],
                                                              df[f'Predicted target(alpha = {alphas[0]})'][
                                                              -len(self.test_data):]), 2)
        mape2 = np.round(100 * mean_absolute_percentage_error(df['Actual target'][-len(self.test_data):],
                                                              df[f'Predicted target(alpha = {alphas[1]})'][
                                                              -len(self.test_data):]), 2)
        mape3 = np.round(100 * mean_absolute_percentage_error(df['Actual target'][-len(self.test_data):],
                                                              df[f'Predicted target(alpha = {alphas[2]})'][
                                                              -len(self.test_data):]), 2)

        print(f'Alpha {alphas[0]}: MAE = ' + str(mae1) + ', MAPE = ' + str(mape1) + '%')
        print(f'Alpha {alphas[1]}: MAE = ' + str(mae2) + ', MAPE = ' + str(mape2) + '%')
        print(f'Alpha {alphas[2]}: MAE = ' + str(mae3) + ', MAPE = ' + str(mape3) + '%')

    def get_holt_predicts(self, alphas, betas):
        df = pd.concat([self.train_data['target'], self.test_data['target']])

        model = Holt(self.train_data['target'])
        predictions1 = model.fit(smoothing_level=alphas[0], smoothing_slope=betas[0]).forecast(len(self.test_data))
        predictions2 = model.fit(smoothing_level=alphas[1], smoothing_slope=betas[1]).forecast(len(self.test_data))
        predictions3 = model.fit(smoothing_level=alphas[2], smoothing_slope=betas[2]).forecast(len(self.test_data))

        df = pd.concat([df, pd.concat([self.train_data['target'], predictions1])], axis=1)
        df.columns = ['Actual target', f'Predicted target(alpha = {alphas[0]})']
        df = pd.concat([df, pd.concat([self.train_data['target'], predictions2])], axis=1)
        df.columns = ['Actual target', f'Predicted target(alpha = {alphas[0]}, beta = {betas[0]})',
                      f'Predicted target(alpha = {alphas[1]}, beta = {betas[1]})']
        df = pd.concat([df, pd.concat([self.train_data['target'], predictions3])], axis=1)
        df.columns = ['Actual target', f'Predicted target(alpha = {alphas[0]}, beta = {betas[0]})',
                      f'Predicted target(alpha = {alphas[1]}, beta = {betas[1]})',
                      f'Predicted target(alpha = {alphas[2]}, beta = {betas[2]})']

        fig = px.line(df, y=df.columns, title='Holt Predictions')
        fig.show()

        mae1 = np.round(mean_absolute_error(df['Actual target'][-len(self.test_data):],
                                            df[f'Predicted target(alpha = {alphas[0]}, beta = {betas[0]})'][
                                            -len(self.test_data):]), 2)
        mae2 = np.round(mean_absolute_error(df['Actual target'][-len(self.test_data):],
                                            df[f'Predicted target(alpha = {alphas[1]}, beta = {betas[1]})'][
                                            -len(self.test_data):]), 2)
        mae3 = np.round(mean_absolute_error(df['Actual target'][-len(self.test_data):],
                                            df[f'Predicted target(alpha = {alphas[2]}, beta = {betas[2]})'][
                                            -len(self.test_data):]), 2)
        mape1 = np.round(100 * mean_absolute_percentage_error(df['Actual target'][-len(self.test_data):], df[
                                                                                                              f'Predicted target(alpha = {alphas[0]}, beta = {betas[0]})'][
                                                                                                          -len(
                                                                                                              self.test_data):]),
                         2)
        mape2 = np.round(100 * mean_absolute_percentage_error(df['Actual target'][-len(self.test_data):], df[
                                                                                                              f'Predicted target(alpha = {alphas[1]}, beta = {betas[1]})'][
                                                                                                          -len(
                                                                                                              self.test_data):]),
                         2)
        mape3 = np.round(100 * mean_absolute_percentage_error(df['Actual target'][-len(self.test_data):], df[
                                                                                                              f'Predicted target(alpha = {alphas[2]}, beta = {betas[2]})'][
                                                                                                          -len(
                                                                                                              self.test_data):]),
                         2)

        print(f'Alpha {alphas[0]}, beta = {betas[0]}: MAE = ' + str(mae1) + ', MAPE = ' + str(mape1) + '%')
        print(f'Alpha {alphas[1]}, beta = {betas[1]}: MAE = ' + str(mae1) + ', MAPE = ' + str(mape2) + '%')
        print(f'Alpha {alphas[2]}, beta = {betas[2]}: MAE = ' + str(mae1) + ', MAPE = ' + str(mape3) + '%')

    def get_tree_predicts(self, start, end, models_list, models_names):
        y_train = self.train_data['target']
        y_test = self.test_data['target']
        X_train = pd.concat(
            [self.train_data.drop('target', axis=1), make_lags(self.train_data.target, start, end).fillna(0)], axis=1)
        X_test = pd.concat(
            [self.test_data.drop('target', axis=1), make_lags(self.test_data.target, start, end).fillna(0)], axis=1)

        df = pd.concat([y_train, y_test])
        mae_scores = []
        mape_scores = []

        for model in models_list:
            model.fit(X_train, y_train)
            y_pred = pd.Series(model.predict(X_test), index=y_test.index)
            predict = pd.concat([y_train, y_pred])
            df = pd.concat([df, predict], axis=1)

            mae_scores.append(np.round(mean_absolute_error(y_test, y_pred), 2))
            mape_scores.append(100 * np.round(mean_absolute_percentage_error(y_test, y_pred), 2))

        df.columns = ['Real'] + models_names

        fig = px.line(df, y=df.columns, title='Tree-based models predictions')
        fig.show()

        for i in range(len(models_list)):
            print(f'{models_names[i]}: MAE = {mae_scores[i]}, MAPE = {mape_scores[i]} %')

    def get_lr_predicts(self, start, end, models_list, models_names):
        train_data = self.train_data.copy()
        test_data = self.test_data.copy()

        train_data['sin_month'] = np.sin(2 * np.pi * train_data.month / 12)
        train_data['cos_month'] = np.cos(2 * np.pi * train_data.month / 12)
        train_data.drop('month', axis=1, inplace=True)

        test_data['sin_month'] = np.sin(2 * np.pi * test_data.month / 12)
        test_data['cos_month'] = np.cos(2 * np.pi * test_data.month / 12)
        test_data.drop('month', axis=1, inplace=True)

        train_data['sin_season'] = np.sin(2 * np.pi * train_data.season / 12)
        train_data['cos_season'] = np.cos(2 * np.pi * train_data.season / 12)
        train_data.drop('season', axis=1, inplace=True)

        test_data['sin_season'] = np.sin(2 * np.pi * test_data.season / 12)
        test_data['cos_season'] = np.cos(2 * np.pi * test_data.season / 12)
        test_data.drop('season', axis=1, inplace=True)

        y_train = train_data['target']
        y_test = test_data['target']
        X_train = pd.concat([train_data.drop('target', axis=1), make_lags(train_data.target, start, end).fillna(0)],
                            axis=1)
        X_test = pd.concat([test_data.drop('target', axis=1), make_lags(test_data.target, start, end).fillna(0)],
                           axis=1)

        scaler = StandardScaler()
        for i in range(start, end + 1):
            X_train[[f'y_lag_{i}']] = scaler.fit_transform(X_train[[f'y_lag_{i}']])
            X_test[[f'y_lag_{i}']] = scaler.transform(X_test[[f'y_lag_{i}']])

        df = pd.concat([y_train, y_test])
        mae_scores = []
        mape_scores = []

        for model in models_list:
            model.fit(X_train, y_train)
            y_pred = pd.Series(replace_neg_values(model.predict(X_test)), index=y_test.index)
            predict = pd.concat([y_train, y_pred])
            df = pd.concat([df, predict], axis=1)

            mae_scores.append(np.round(mean_absolute_error(y_test, y_pred), 2))
            mape_scores.append(100 * np.round(mean_absolute_percentage_error(y_test, y_pred), 2))

        df.columns = ['Real'] + models_names

        fig = px.line(df, y=df.columns, title='Linear models Predictions')
        fig.show()

        for i in range(len(models_list)):
            print(f'{models_names[i]}: MAE = {mae_scores[i]}, MAPE = {mape_scores[i]} %')