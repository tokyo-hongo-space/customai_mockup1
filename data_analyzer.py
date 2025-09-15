import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DataAnalyzer:
    """データ分析・特徴抽出クラス"""
    
    def __init__(self):
        plt.rcParams['font.family'] = 'DejaVu Sans'
    
    def analyze_data_structure(self, df: pd.DataFrame) -> Dict:
        """データ構造の分析"""
        analysis = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
        }
        
        # 時間列の推定
        time_candidates = []
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].dropna().iloc[:10])
                    time_candidates.append(col)
                except:
                    pass
            elif 'date' in col.lower() or 'time' in col.lower():
                time_candidates.append(col)
        
        analysis['time_column_candidates'] = time_candidates
        
        # 数値列の特定
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        analysis['numeric_columns'] = numeric_columns
        
        return analysis
    
    def detect_time_patterns(self, df: pd.DataFrame, time_col: str, target_col: str) -> Dict:
        """時系列パターンの検出"""
        # 時間列を datetime に変換
        df_copy = df.copy()
        df_copy[time_col] = pd.to_datetime(df_copy[time_col])
        df_copy = df_copy.sort_values(time_col)
        
        patterns = {
            'date_range': {
                'start': str(df_copy[time_col].min()),
                'end': str(df_copy[time_col].max()),
                'days': (df_copy[time_col].max() - df_copy[time_col].min()).days
            },
            'frequency': self._detect_frequency(df_copy[time_col]),
            'missing_pattern': self._analyze_missing_pattern(df_copy, time_col, target_col),
            'trend': self._detect_trend(df_copy[target_col].dropna()),
            'seasonality': self._detect_seasonality(df_copy, time_col, target_col),
            'outliers': self._detect_outliers(df_copy[target_col].dropna()),
            'summary_stats': df_copy[target_col].describe().to_dict()
        }
        
        return patterns
    
    def _detect_frequency(self, time_series: pd.Series) -> str:
        """データ頻度の推定"""
        time_diff = time_series.diff().dropna()
        mode_diff = time_diff.mode()
        
        if len(mode_diff) == 0:
            return "不明"
        
        days = mode_diff.iloc[0].days
        
        if days == 1:
            return "日次"
        elif 6 <= days <= 8:
            return "週次"
        elif 28 <= days <= 32:
            return "月次"
        elif 88 <= days <= 95:
            return "四半期"
        elif 360 <= days <= 370:
            return "年次"
        else:
            return f"不規則（平均{days}日間隔）"
    
    def _analyze_missing_pattern(self, df: pd.DataFrame, time_col: str, target_col: str) -> List[str]:
        """欠損値パターンの分析"""
        patterns = []
        missing_data = df[df[target_col].isnull()].copy()
        
        if len(missing_data) == 0:
            return ["欠損値なし"]
        
        # 曜日パターン
        missing_data['weekday'] = missing_data[time_col].dt.day_name()
        weekday_counts = missing_data['weekday'].value_counts()
        if len(weekday_counts) > 0:
            top_weekday = weekday_counts.index[0]
            if weekday_counts[top_weekday] > len(missing_data) * 0.3:
                patterns.append(f"{top_weekday}に欠損が集中")
        
        # 月パターン
        missing_data['month'] = missing_data[time_col].dt.month
        month_counts = missing_data['month'].value_counts()
        if len(month_counts) > 0:
            top_month = month_counts.index[0]
            if month_counts[top_month] > len(missing_data) * 0.3:
                patterns.append(f"{top_month}月に欠損が集中")
        
        # 期間パターン
        if len(missing_data) > 0:
            patterns.append(f"全{len(df)}行中{len(missing_data)}行が欠損（{len(missing_data)/len(df)*100:.1f}%）")
        
        return patterns
    
    def _detect_trend(self, series: pd.Series) -> str:
        """トレンドの検出"""
        if len(series) < 3:
            return "データ不足"
        
        # 線形回帰での傾きを計算
        x = np.arange(len(series))
        slope = np.polyfit(x, series, 1)[0]
        
        if abs(slope) < series.std() * 0.01:
            return "横ばい"
        elif slope > 0:
            return f"上昇傾向（傾き: {slope:.2f}）"
        else:
            return f"下降傾向（傾き: {slope:.2f}）"
    
    def _detect_seasonality(self, df: pd.DataFrame, time_col: str, target_col: str) -> Dict:
        """季節性の検出"""
        df_clean = df.dropna(subset=[target_col])
        
        if len(df_clean) < 12:  # 最低12データポイント必要
            return {"月別": "データ不足", "曜日別": "データ不足"}
        
        seasonality = {}
        
        # 月別季節性
        df_clean['month'] = df_clean[time_col].dt.month
        monthly_avg = df_clean.groupby('month')[target_col].mean()
        monthly_std = monthly_avg.std()
        monthly_peak = monthly_avg.idxmax()
        monthly_trough = monthly_avg.idxmin()
        
        if monthly_std > monthly_avg.mean() * 0.1:
            seasonality['月別'] = f"季節性あり（ピーク: {monthly_peak}月, 最低: {monthly_trough}月）"
        else:
            seasonality['月別'] = "明確な季節性なし"
        
        # 曜日別パターン
        df_clean['weekday'] = df_clean[time_col].dt.dayofweek
        weekly_avg = df_clean.groupby('weekday')[target_col].mean()
        weekly_std = weekly_avg.std()
        
        weekday_names = ['月', '火', '水', '木', '金', '土', '日']
        weekly_peak_day = weekday_names[weekly_avg.idxmax()]
        weekly_trough_day = weekday_names[weekly_avg.idxmin()]
        
        if weekly_std > weekly_avg.mean() * 0.1:
            seasonality['曜日別'] = f"曜日パターンあり（高: {weekly_peak_day}曜日, 低: {weekly_trough_day}曜日）"
        else:
            seasonality['曜日別'] = "明確な曜日パターンなし"
        
        return seasonality
    
    def _detect_outliers(self, series: pd.Series) -> Dict:
        """外れ値の検出"""
        if len(series) < 4:
            return {"count": 0, "method": "データ不足"}
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        outlier_mask = (series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)
        outlier_count = outlier_mask.sum()
        
        return {
            "count": int(outlier_count),
            "percentage": outlier_count / len(series) * 100,
            "method": "IQR法"
        }
    
    def create_summary_visualization(self, df: pd.DataFrame, time_col: str, target_col: str) -> str:
        """データサマリーの可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('データ分析サマリー', fontsize=16)
        
        # データをコピーして日付変換
        df_plot = df.copy()
        df_plot[time_col] = pd.to_datetime(df_plot[time_col])
        df_plot = df_plot.sort_values(time_col)
        
        # 1. 時系列プロット
        axes[0,0].plot(df_plot[time_col], df_plot[target_col], linewidth=1, alpha=0.7)
        axes[0,0].set_title('時系列データ')
        axes[0,0].set_xlabel('時間')
        axes[0,0].set_ylabel(target_col)
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. ヒストグラム
        axes[0,1].hist(df_plot[target_col].dropna(), bins=30, alpha=0.7, edgecolor='black')
        axes[0,1].set_title('値の分布')
        axes[0,1].set_xlabel(target_col)
        axes[0,1].set_ylabel('頻度')
        
        # 3. 月別ボックスプロット
        df_plot['month'] = df_plot[time_col].dt.month
        monthly_data = [df_plot[df_plot['month'] == m][target_col].dropna() for m in range(1, 13)]
        monthly_data = [data for data in monthly_data if len(data) > 0]  # 空のデータを除外
        
        if len(monthly_data) > 0:
            axes[1,0].boxplot(monthly_data, labels=[f'{i+1}月' for i in range(len(monthly_data))])
            axes[1,0].set_title('月別分布')
            axes[1,0].set_xlabel('月')
            axes[1,0].set_ylabel(target_col)
            axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. 欠損値パターン
        missing_by_month = df_plot.groupby('month')[target_col].apply(lambda x: x.isnull().sum())
        axes[1,1].bar(missing_by_month.index, missing_by_month.values)
        axes[1,1].set_title('月別欠損値数')
        axes[1,1].set_xlabel('月')
        axes[1,1].set_ylabel('欠損値数')
        
        plt.tight_layout()
        
        # 図を保存
        viz_path = 'data_summary_visualization.png'
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return viz_path