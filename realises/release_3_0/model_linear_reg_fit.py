import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import os


def load_and_validate_data(filename: str):
    """Загрузка CSV и проверяем наличие всех колонок целевой переменной"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Файл '{filename}' не найден в текущем каталоге.")
    
    df = pd.read_csv(filename)
    required_cols = {'price_rent', 'price_log', 'price_log_scaled'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"В файле отсутствуют колонки: {missing}")
    
    if 'ad_id' in df.columns:
        df = df.drop(columns=['ad_id'])
    
    return df


def prepare_targets_and_features(df: pd.DataFrame):
    y_rent = df['price_rent']
    y_log = df['price_log']
    y_log_scaled = df['price_log_scaled']
    
    feature_cols = [col for col in df.columns if col not in {'price_rent', 'price_log', 'price_log_scaled'}]
    X = df[feature_cols]
    
    return X, y_rent, y_log, y_log_scaled


def split_data_consistently(X, y_rent, y_log, y_log_scaled, test_size=0.25, random_state=42):
    X_train, X_test, y_rent_train, y_rent_test = train_test_split(
        X, y_rent, test_size=test_size, random_state=random_state
    )
    
    train_idx = y_rent_train.index
    test_idx = y_rent_test.index
    
    y_log_train = y_log.loc[train_idx].values
    y_log_test = y_log.loc[test_idx].values
    y_log_scaled_train = y_log_scaled.loc[train_idx].values
    y_log_scaled_test = y_log_scaled.loc[test_idx].values
    
    return (X_train, X_test, 
            y_rent_train, y_rent_test,
            y_log_train, y_log_test,
            y_log_scaled_train, y_log_scaled_test)


def fit_scaler_on_log_target(y_log_train):
    scaler = StandardScaler()
    scaler.fit(y_log_train.reshape(-1, 1))
    return scaler


def evaluate_regression_model(
    X_train, X_test, y_train, y_test_true, 
    target_type: str, 
    scaler=None
):
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    if target_type == 'rent':
        pred_original = pred
    elif target_type == 'log':
        pred_original = np.expm1(pred)
    elif target_type == 'scaled':
        pred_log = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
        pred_original = np.expm1(pred_log)
    else:
        raise ValueError("target_type must be 'rent', 'log', or 'scaled'")
    
    pred_original = np.clip(pred_original, a_min=1.0, a_max=None)
    
    mape = mean_absolute_percentage_error(y_test_true, pred_original) * 100
    mae = mean_absolute_error(y_test_true, pred_original)
    
    return mape, mae


def main():
    DATA_FILE = 'stars25_dan_data.csv'
    
    # Load and validate
    df = load_and_validate_data(DATA_FILE)
    
    X, y_rent, y_log, y_log_scaled = prepare_targets_and_features(df)
    
    (X_train, X_test,
     y_rent_train, y_rent_test,
     y_log_train, y_log_test,
     y_log_scaled_train, y_log_scaled_test) = split_data_consistently(
        X, y_rent, y_log, y_log_scaled
    )
    
    # маштабируем обратно к логарифмированной цене
    scaler = fit_scaler_on_log_target(y_log_train)
    
    results = {}
    
    results['Прямое (price_rent)'] = evaluate_regression_model(
        X_train, X_test, y_rent_train, y_rent_test, 'rent'    )
    
    results['Log1p(price)'] = evaluate_regression_model(
        X_train, X_test, y_log_train, y_rent_test, 'log'    )
    
    results['Масштабированный Log1p(price)'] = evaluate_regression_model(
        X_train, X_test, y_log_scaled_train, y_rent_test, 'scaled', scaler    )
    
    print("\nРезультаты оценки моделей:")
    print(f"{'Модель':<35} {'MAPE (%)':<10} {'MAE (руб)':<12}")
    print("-" * 65)
    
    best_mape = float('inf')
    best_model_name = ""
    
    for name, (mape, mae) in results.items():
        print(f"{name:<35} {mape:<10.2f} {mae:<12.0f}")
        if mape < best_mape:
            best_mape = mape
            best_model_name = name
    
    print(f"\nЛучшая модель по MAPE: {best_model_name} ({best_mape:.2f}%)")


if __name__ == "__main__":
    main()