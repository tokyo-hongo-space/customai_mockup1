import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class ForecastEngine:
    """時系列予測エンジン"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        
    def prepare_data(self, df: pd.DataFrame, time_col: str, target_col: str, 
                    aggregation_level: str = "現在のままで良い", 
                    missing_strategy: str = "欠損値を除外して分析") -> pd.DataFrame:
        """データ前処理"""
        
        # データのコピーを作成
        data = df.copy()
        data[time_col] = pd.to_datetime(data[time_col])
        data = data.sort_values(time_col)
        
        # 欠損値処理
        if missing_strategy == "欠損値を除外して分析":
            data = data.dropna(subset=[target_col])
        elif missing_strategy == "0で埋める":
            data[target_col] = data[target_col].fillna(0)
        elif missing_strategy == "前後の値で補間":
            data[target_col] = data[target_col].interpolate(method='linear')
        elif missing_strategy == "平均値で補間":
            mean_value = data[target_col].mean()
            data[target_col] = data[target_col].fillna(mean_value)
        
        # データ集約
        if aggregation_level == "週単位で集約したい":
            data = self._aggregate_weekly(data, time_col, target_col)
        elif aggregation_level == "月単位で集約したい":
            data = self._aggregate_monthly(data, time_col, target_col)
        elif aggregation_level == "日単位で集約したい":
            data = self._aggregate_daily(data, time_col, target_col)
        
        # 最終的なデータクリーニング
        data = data.dropna(subset=[target_col])
        data = data.reset_index(drop=True)
        
        return data
    
    def _aggregate_weekly(self, df: pd.DataFrame, time_col: str, target_col: str) -> pd.DataFrame:
        """週次集約"""
        df['week'] = df[time_col].dt.to_period('W').dt.start_time
        weekly_data = df.groupby('week').agg({
            target_col: ['mean', 'sum', 'count']
        }).reset_index()
        
        # 列名を整理
        weekly_data.columns = ['date', 'mean_value', 'sum_value', 'count']
        weekly_data = weekly_data.rename(columns={'date': time_col, 'mean_value': target_col})
        
        # 最低3件以上のデータがある週のみを残す
        weekly_data = weekly_data[weekly_data['count'] >= 3]
        
        return weekly_data[[time_col, target_col]]
    
    def _aggregate_monthly(self, df: pd.DataFrame, time_col: str, target_col: str) -> pd.DataFrame:
        """月次集約"""
        df['month'] = df[time_col].dt.to_period('M').dt.start_time
        monthly_data = df.groupby('month').agg({
            target_col: ['mean', 'sum', 'count']
        }).reset_index()
        
        # 列名を整理
        monthly_data.columns = ['date', 'mean_value', 'sum_value', 'count']
        monthly_data = monthly_data.rename(columns={'date': time_col, 'mean_value': target_col})
        
        return monthly_data[[time_col, target_col]]
    
    def _aggregate_daily(self, df: pd.DataFrame, time_col: str, target_col: str) -> pd.DataFrame:
        """日次集約"""
        df['day'] = df[time_col].dt.date
        daily_data = df.groupby('day').agg({
            target_col: ['mean', 'sum', 'count']
        }).reset_index()
        
        # 列名を整理
        daily_data.columns = ['date', 'mean_value', 'sum_value', 'count']
        daily_data = daily_data.rename(columns={'date': time_col, 'mean_value': target_col})
        daily_data[time_col] = pd.to_datetime(daily_data[time_col])
        
        return daily_data[[time_col, target_col]]
    
    def split_train_test(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """訓練・テストデータの分割"""
        split_point = int(len(df) * (1 - test_size))
        train_data = df.iloc[:split_point].copy()
        test_data = df.iloc[split_point:].copy()
        
        return train_data, test_data
    
    def build_arima_model(self, train_data: pd.DataFrame, target_col: str) -> Dict:
        """ARIMAモデルの構築"""
        try:
            series = train_data[target_col]
            
            # パラメータ自動選択（簡易版）
            best_aic = float('inf')
            best_params = None
            best_model = None
            
            # グリッドサーチ（簡略化）
            for p in range(0, 3):
                for d in range(0, 2):
                    for q in range(0, 3):
                        try:
                            model = ARIMA(series, order=(p, d, q))
                            fitted_model = model.fit()
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_params = (p, d, q)
                                best_model = fitted_model
                        except:
                            continue
            
            if best_model is None:
                # フォールバック: シンプルなARIMA(1,1,1)
                model = ARIMA(series, order=(1, 1, 1))
                best_model = model.fit()
                best_params = (1, 1, 1)
            
            return {
                'model': best_model,
                'params': best_params,
                'aic': best_model.aic,
                'method': 'ARIMA'
            }
            
        except Exception as e:
            return {'error': str(e), 'method': 'ARIMA'}
    
    def build_sarima_model(self, train_data: pd.DataFrame, target_col: str) -> Dict:
        """SARIMAモデルの構築"""
        try:
            series = train_data[target_col]
            
            # 季節性の検出
            if len(series) >= 24:
                seasonal_period = min(12, len(series) // 4)  # 季節周期を推定
            else:
                seasonal_period = 4  # デフォルト
            
            # 簡易パラメータ選択
            try:
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                model = SARIMAX(series, 
                              order=(1, 1, 1), 
                              seasonal_order=(1, 1, 1, seasonal_period))
                fitted_model = model.fit(disp=False)
                
                return {
                    'model': fitted_model,
                    'params': f"(1,1,1)x(1,1,1,{seasonal_period})",
                    'aic': fitted_model.aic,
                    'method': 'SARIMA'
                }
            except:
                # SARIMAが失敗した場合はARIMAにフォールバック
                return self.build_arima_model(train_data, target_col)
                
        except Exception as e:
            return {'error': str(e), 'method': 'SARIMA'}
    
    def build_prophet_model(self, train_data: pd.DataFrame, time_col: str, target_col: str) -> Dict:
        """Prophetモデルの構築"""
        try:
            # Prophetの入力形式に変換
            prophet_data = train_data[[time_col, target_col]].copy()
            prophet_data.columns = ['ds', 'y']
            
            # Prophetモデル作成
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            
            model.fit(prophet_data)
            
            return {
                'model': model,
                'method': 'Prophet',
                'data_format': 'prophet'
            }
            
        except Exception as e:
            return {'error': str(e), 'method': 'Prophet'}
    
    def evaluate_model(self, model_result: Dict, train_data: pd.DataFrame, test_data: pd.DataFrame, 
                      time_col: str, target_col: str) -> Dict:
        """モデルの評価"""
        
        if 'error' in model_result:
            return {'error': model_result['error'], 'mape': 100}
        
        try:
            actual = test_data[target_col].values
            
            if model_result['method'] in ['ARIMA', 'SARIMA']:
                # ARIMA/SARIMAの予測
                forecast = model_result['model'].forecast(steps=len(test_data))
                predicted = forecast.values if hasattr(forecast, 'values') else forecast
                
            elif model_result['method'] == 'Prophet':
                # Prophetの予測
                future = model_result['model'].make_future_dataframe(periods=len(test_data))
                forecast = model_result['model'].predict(future)
                predicted = forecast['yhat'].iloc[-len(test_data):].values
            
            # MAPE計算
            mape = self.calculate_mape(actual, predicted)
            
            # その他の評価指標
            mae = np.mean(np.abs(actual - predicted))
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            
            return {
                'mape': mape,
                'mae': mae,
                'rmse': rmse,
                'actual': actual.tolist(),
                'predicted': predicted.tolist()
            }
            
        except Exception as e:
            return {'error': str(e), 'mape': 100}
    
    def calculate_mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """MAPE（平均絶対パーセント誤差）の計算"""
        # ゼロ値の処理
        actual_non_zero = np.where(actual == 0, 1e-7, actual)
        mape = np.mean(np.abs((actual - predicted) / actual_non_zero)) * 100
        return float(mape)
    
    def generate_forecast(self, best_model: Dict, periods: int, time_col: str, 
                         last_date: pd.Timestamp) -> pd.DataFrame:
        """将来予測の生成"""
        
        try:
            if best_model['method'] in ['ARIMA', 'SARIMA']:
                # ARIMA/SARIMAの予測
                forecast = best_model['model'].forecast(steps=periods)
                forecast_values = forecast.values if hasattr(forecast, 'values') else forecast
                
                # 信頼区間の取得
                conf_int = best_model['model'].get_forecast(steps=periods).conf_int()
                lower_bound = conf_int.iloc[:, 0].values
                upper_bound = conf_int.iloc[:, 1].values
                
            elif best_model['method'] == 'Prophet':
                # Prophetの予測
                future = best_model['model'].make_future_dataframe(periods=periods)
                forecast = best_model['model'].predict(future)
                
                forecast_values = forecast['yhat'].iloc[-periods:].values
                lower_bound = forecast['yhat_lower'].iloc[-periods:].values
                upper_bound = forecast['yhat_upper'].iloc[-periods:].values
            
            # 日付の生成
            freq_map = {
                '日次': 'D',
                '週次': 'W',
                '月次': 'M'
            }
            
            # 頻度を推定（簡易版）
            freq = 'D'  # デフォルト
            
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=periods,
                freq=freq
            )
            
            # 結果のDataFrame作成
            forecast_df = pd.DataFrame({
                time_col: future_dates,
                'forecast': forecast_values,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            })
            
            return forecast_df
            
        except Exception as e:
            # エラー時は空のDataFrameを返す
            return pd.DataFrame(columns=[time_col, 'forecast', 'lower_bound', 'upper_bound'])
    
    def run_full_analysis(self, df: pd.DataFrame, time_col: str, target_col: str,
                         config: Dict) -> Dict:
        """完全な分析の実行"""
        
        results = {
            'data_preparation': {},
            'models': {},
            'best_model': None,
            'forecast': None,
            'visualization_paths': []
        }
        
        try:
            # 1. データ準備
            prepared_data = self.prepare_data(
                df, time_col, target_col,
                config.get('aggregation_level', '現在のままで良い'),
                config.get('missing_strategy', '欠損値を除外して分析')
            )
            
            results['data_preparation'] = {
                'original_rows': len(df),
                'prepared_rows': len(prepared_data),
                'date_range': {
                    'start': str(prepared_data[time_col].min()),
                    'end': str(prepared_data[time_col].max())
                }
            }
            
            # 2. 訓練・テストデータ分割
            train_data, test_data = self.split_train_test(prepared_data)
            
            # 3. 複数モデルの構築・評価
            models_to_try = ['ARIMA', 'SARIMA', 'Prophet']
            
            for model_type in models_to_try:
                try:
                    if model_type == 'ARIMA':
                        model_result = self.build_arima_model(train_data, target_col)
                    elif model_type == 'SARIMA':
                        model_result = self.build_sarima_model(train_data, target_col)
                    elif model_type == 'Prophet':
                        model_result = self.build_prophet_model(train_data, time_col, target_col)
                    
                    if 'error' not in model_result:
                        # モデル評価
                        evaluation = self.evaluate_model(model_result, train_data, test_data, time_col, target_col)
                        model_result['evaluation'] = evaluation
                        
                        results['models'][model_type] = model_result
                except Exception as e:
                    results['models'][model_type] = {'error': str(e)}
            
            # 4. 最良モデルの選択
            best_mape = float('inf')
            best_model_name = None
            
            for model_name, model_result in results['models'].items():
                if 'evaluation' in model_result and 'mape' in model_result['evaluation']:
                    mape = model_result['evaluation']['mape']
                    if mape < best_mape:
                        best_mape = mape
                        best_model_name = model_name
            
            if best_model_name:
                results['best_model'] = {
                    'name': best_model_name,
                    'mape': best_mape,
                    'model_data': results['models'][best_model_name]
                }
                
                # 5. 将来予測
                periods = self._get_forecast_periods(config.get('prediction_period', '1ヶ月先'))
                last_date = prepared_data[time_col].max()
                
                forecast_df = self.generate_forecast(
                    results['models'][best_model_name], 
                    periods, 
                    time_col, 
                    last_date
                )
                
                results['forecast'] = forecast_df
            
            # 6. 可視化
            viz_path = self.create_forecast_visualization(
                prepared_data, train_data, test_data, results, time_col, target_col
            )
            results['visualization_paths'].append(viz_path)
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _get_forecast_periods(self, prediction_period: str) -> int:
        """予測期間から期間数を計算"""
        period_map = {
            '1週間先': 7,
            '1ヶ月先': 30,
            '3ヶ月先': 90,
            '6ヶ月先': 180,
            '1年先': 365
        }
        return period_map.get(prediction_period, 30)
    
    def create_forecast_visualization(self, data: pd.DataFrame, train_data: pd.DataFrame, 
                                    test_data: pd.DataFrame, results: Dict, 
                                    time_col: str, target_col: str) -> str:
        """予測結果の可視化"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('予測分析結果', fontsize=16)
        
        # 1. 全体時系列とテスト結果
        axes[0,0].plot(train_data[time_col], train_data[target_col], label='訓練データ', color='blue')
        axes[0,0].plot(test_data[time_col], test_data[target_col], label='実際の値', color='green')
        
        if results['best_model'] and 'evaluation' in results['best_model']['model_data']:
            predicted = results['best_model']['model_data']['evaluation'].get('predicted', [])
            if predicted:
                axes[0,0].plot(test_data[time_col], predicted, label='予測値', color='red', linestyle='--')
        
        axes[0,0].set_title('訓練・テストデータと予測結果')
        axes[0,0].legend()
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. 将来予測
        if results['forecast'] is not None and len(results['forecast']) > 0:
            axes[0,1].plot(data[time_col], data[target_col], label='実績データ', color='blue')
            axes[0,1].plot(results['forecast'][time_col], results['forecast']['forecast'], 
                          label='将来予測', color='red', linestyle='--')
            axes[0,1].fill_between(results['forecast'][time_col], 
                                  results['forecast']['lower_bound'],
                                  results['forecast']['upper_bound'],
                                  alpha=0.3, color='red', label='信頼区間')
        
        axes[0,1].set_title('将来予測')
        axes[0,1].legend()
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. モデル性能比較
        model_names = []
        mape_scores = []
        
        for model_name, model_result in results['models'].items():
            if 'evaluation' in model_result and 'mape' in model_result['evaluation']:
                model_names.append(model_name)
                mape_scores.append(model_result['evaluation']['mape'])
        
        if model_names:
            bars = axes[1,0].bar(model_names, mape_scores, color=['blue', 'green', 'red'])
            axes[1,0].set_title('モデル精度比較（MAPE）')
            axes[1,0].set_ylabel('MAPE (%)')
            
            # 目標精度ライン（10%）
            axes[1,0].axhline(y=10, color='orange', linestyle='--', label='目標精度(10%)')
            axes[1,0].legend()
            
            # 数値ラベル追加
            for bar, score in zip(bars, mape_scores):
                axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                              f'{score:.1f}%', ha='center')
        
        # 4. 残差分析（最良モデル）
        if results['best_model'] and 'evaluation' in results['best_model']['model_data']:
            evaluation = results['best_model']['model_data']['evaluation']
            actual = evaluation.get('actual', [])
            predicted = evaluation.get('predicted', [])
            
            if actual and predicted:
                residuals = np.array(actual) - np.array(predicted)
                axes[1,1].scatter(predicted, residuals, alpha=0.6)
                axes[1,1].axhline(y=0, color='red', linestyle='--')
                axes[1,1].set_xlabel('予測値')
                axes[1,1].set_ylabel('残差')
                axes[1,1].set_title(f'残差分析 ({results["best_model"]["name"]})')
        
        plt.tight_layout()
        
        # 図を保存
        viz_path = 'forecast_analysis_result.png'
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return viz_path