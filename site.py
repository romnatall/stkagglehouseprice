from math import exp
import numpy as np
import optuna
from optuna.samplers import TPESampler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from altair import Column
from category_encoders import OneHotEncoder, OrdinalEncoder,TargetEncoder
from scipy.stats import ttest_ind
import shap
import catboost as cat

from sklearn.metrics import mean_squared_log_error
import streamlit.components.v1 as components

from sklearn import pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.feature_selection import f_regression, chi2
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.neighbors import KNeighborsRegressor
import tempfile

import streamlit as st
import pandas as pd
import pickle
import sklearn
shap.initjs()
st.set_page_config(layout="wide")

sklearn.set_config(transform_output="pandas")
fec=['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
       'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
       'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
       'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond',
       'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
       'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
       'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType',
       'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC',
       'Fence', 'MiscFeature', 'SaleType', 'SaleCondition', 'Id',
       'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
       'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
       'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
       'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
       'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt',
       'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
       'MoSold', 'YrSold']
class mypreprocess(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.nanistype=[
            'Alley',
            'BsmtQual',
            'BsmtCond',
            'BsmtExposure',
            'BsmtFinType1',
            'BsmtFinType2',
            'FireplaceQu',
            'GarageType',
            'GarageFinish',
            'GarageQual',
            'GarageCond',
            'PoolQC',
            'Fence',
            'MiscFeature',
            'MasVnrType'
        ]
        self.rendict={
            'name':'MSSubClass',
            20: '1-STORY 1946 & NEWER ALL STYLES',
            30: '1-STORY 1945 & OLDER',
            40: '1-STORY W/FINISHED ATTIC ALL AGES',
            45: '1-1/2 STORY - UNFINISHED ALL AGES',
            50: '1-1/2 STORY FINISHED ALL AGES',
            60: '2-STORY 1946 & NEWER',
            70: '2-STORY 1945 & OLDER',
            75: '2-1/2 STORY ALL AGES',
            80: 'SPLIT OR MULTI-LEVEL',
            85: 'SPLIT FOYER',
            90: 'DUPLEX - ALL STYLES AND AGES',
            120: '1-STORY PUD (Planned Unit Development) - 1946 & NEWER',
            150: '1-1/2 STORY PUD - ALL AGES',
            160: '2-STORY PUD - 1946 & NEWER',
            180: 'PUD - MULTILEVEL - INCL SPLIT LEV/FOYER',
            190: '2 FAMILY CONVERSION - ALL STYLES AND AGES'
        }

    def preprocess( self, df):
       
        df[self.rendict['name']]=df[self.rendict['name']].map( self.rendict)    
        df[ self.nanistype]=df[self.nanistype].fillna('Empty')
        return df
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.preprocess(X)

# Загрузка предварительно сохраненного препроцессора и модели


def preprocess_data(df):
    # Применение предварительного обработчика к данным
    processed_df = preprocessor.transform(df)
    return processed_df

def predict_prices(df):
    # Предсказание цен с помощью модели
    predictions = model.predict(df)
    return predictions

def main():
    st.title('Прогнозирование цен на недвижимость')

    # Загрузка файла CSV
    
    if (st.radio("загрузка:", ("из файла", "sample file"))=="sample file"):
        uploaded_file = 'test.csv'
    else:
        uploaded_file = st.file_uploader("Загрузите файл CSV", type=['csv'])

    if uploaded_file is not None:
        # Считывание данных из файла CSV
        val = pd.DataFrame(pd.read_csv(uploaded_file))
        st.write('**Исходные данные:**')
        st.write(val)
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)

        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)

        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)

        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)

        valx= preprocessor.transform(val)
        val_pred = model.predict(valx)

        # Создание датафрейма с ответами
        results_df = pd.DataFrame({'Id': val['Id'], 'Predicted SalePrice': val_pred})

        st.write('**Результаты:**')
        st.write(results_df)

        # Сохранение файла с ответами
        csv = results_df.to_csv(index=False)
        st.download_button(label='Скачать результаты', data=csv, file_name='predicted_prices.csv', mime='text/csv')


        # Получение важности признаков из модели
        feature_importance = model.feature_importances_

        # Создание DataFrame с важностью признаков
        importance_df = pd.DataFrame({'Feature': fec,
                                    'Importance': feature_importance}).sort_values(by='Importance', ascending=False)

        # Вывод датафрейма с важностью признаков
        st.write('**Важность признаков:**')
        st.write(importance_df.T)

        selected_row = st.selectbox("Выберите строку:", val)

        st.write(val[val['Id'] == selected_row])
        v=val[val['Id'] == selected_row]
        X=preprocessor.transform(v)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        def st_shap(plot, height=None):
            shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
            components.html(shap_html, height=height)

        # Создание графика Shapley values с использованием matplotlib

        st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:]))

      
        


if __name__ == '__main__':
    main()
