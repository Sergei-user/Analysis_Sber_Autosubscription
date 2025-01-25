import pandas as pd
import dill

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from datetime import datetime


def filter_data(df):
    data = df.copy()
    columns_to_drop = [
        'device_model',
        'utm_keyword',
        'client_id',
        'session_id',
    ]

    return data.drop(columns_to_drop, axis=1)


def new_features(df):
    import pandas as pd
    import json

    data = df.copy()

    with open('data/city_lat_long.json', 'r') as file:
        city_lat_long = json.load(file)

    with open('data/distance_to_Moscow.json', 'r') as file:
        dist_to_Moscow = json.load(file)

    # Источники органического трафика из 'utm_medium'
    organic_traf_medium = ['organic', 'referral', '(none)']  # все остальные значения - платный трафик
    # Источники рекламмы в соцсетях из 'utm_source'
    social_media_source = ['QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt',
                           'ISrKoXQCxqqYvAZICvjs', 'IZEXUFLARCUMynmHNBGo',
                           'PlbkrSYoHuZBWfYjYnfw', 'gVRrcxiDQubJiljoTbGm']
    # создание новых признаков времени
    data['visit_time'] = pd.to_datetime(data['visit_time'], format='%H:%M:%S')
    data['visit_hour'] = data['visit_time'].dt.hour
    data['visit_day_night'] = data['visit_hour'].apply(lambda x: 'day' if 8 < x < 21 else 'night')

    # создание новых признаков даты
    data['visit_date'] = pd.to_datetime(data['visit_date'])
    data['visit_month'] = data['visit_date'].dt.month
    data['visit_day'] = data['visit_date'].dt.day
    data['dayofweek'] = data['visit_date'].dt.weekday

    # создание признака трафика из 'utm_medium'
    data['social_media_source'] = data['utm_source'].apply(lambda x:
                                                           'social' if x in social_media_source else 'no_social')
    # создание признака источников рекламмы в соцсетях из 'utm_source'
    data['traf_medium'] = data['utm_medium'].apply(lambda x:
                                                   'organic' if x in organic_traf_medium else 'no_organic')
    # создание признака площадь экрана
    data['device_screen_area'] = data['device_screen_resolution'].apply(
        lambda x: int(x.split('x')[0]) * int(x.split('x')[1]))
    # заменим Russia на Rusha, чтобы в дальнейшем если нет координат у города найти координаты страны,
    # а Russia определяеся как город в США и создадим признак Россия или нет.
    data.loc[:, 'geo_country'] = data['geo_country'].replace('Russia', 'Rusha')
    data['country'] = data['geo_country'].apply(lambda x: 1.0 if x == 'Rusha' else 0.0)

    # заменим строки с отсутствующим городом на страну и создадим признаки широта и долгота
    data['geo_city'] = data.apply(lambda x: (x['geo_country'] if x['geo_city'] == '(not set)'
                                             else x['geo_city']), axis=1)
    data.loc[:, 'lat_city'] = data['geo_city'].apply(
        lambda x: city_lat_long[x][0] if x != '(not set)' else city_lat_long['Moscow'][0])
    data.loc[:, 'long_city'] = data['geo_city'].apply(
        lambda x: city_lat_long[x][1] if x != '(not set)' else city_lat_long['Moscow'][1])
    # создание признака расстояние от города до Москвы
    data.loc[:, 'dist_to_Moscow'] = data['geo_city'].apply(lambda x: dist_to_Moscow[x] if x != '(not set)' else 0.0)

    return data


def pipeline():
    print('Старт Sber_car_sub Prediction Pipeline')
    df = pd.read_csv(f'data/df_session_target.csv')

    X = df.drop('target', axis=1)
    y = df['target']

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='(not set)')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', min_frequency=500))
    ])

    column_transformer = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer,
         make_column_selector(dtype_include=['int64', 'int32', 'float64', 'float32'])),
        ('categorical', categorical_transformer, make_column_selector(dtype_include=['object']))
    ])

    preprocessor = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data)),
        # ('imputer', FunctionTransformer(fill_missings)),
        ('feature_creator', FunctionTransformer(new_features)),
        ('column_transformer', column_transformer)
    ])

    models = [
        LogisticRegression(solver='liblinear', random_state=42, max_iter=200, class_weight='balanced')
        # RandomForestClassifier(random_state=42, max_features='sqrt', min_samples_leaf=23, n_estimators=150,
        #                        bootstrap=False, verbose=10),
        # MLPClassifier(random_state=42, hidden_layer_sizes=(100, 50, 50), activation='tanh'),
    ]

    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        print('Начало cross_val')
        score = cross_val_score(pipe, X, y, cv=4, scoring='roc_auc')
        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, roc_auc: {best_score:.4f}')
    print('Начало обучения на всем датасете')
    best_pipe.fit(X, y)

    with open('model/sber_cars_sub_pipe.pkl', 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'Sber_car_sub prediction model',
                'author': 'Sergei Kostrov',
                'version': 1,
                'date': datetime.now(),
                'type': type(best_pipe.named_steps["classifier"]).__name__,
                'accuracy': best_score
            }
        }, file)


if __name__ == '__main__':
    pipeline()
