import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import traceback

# utilsモジュールのインポート
try:
    from utils.llm_client import LocalLLMClient
    from utils.data_analyzer import DataAnalyzer
    from utils.interactive_questioner import InteractiveQuestioner
    from utils.forecast_engine import ForecastEngine
    from utils.report_generator import ReportGenerator
except ImportError as e:
    st.error(f"モジュールのインポートエラー: {e}")
    st.info("utils/ディレクトリ内のファイルが正しく配置されているか確認してください")
    st.stop()

# ページ設定
st.set_page_config(
    page_title="ローカルLLM売上予測ツール",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-header {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# セッション状態の初期化
def initialize_session_state():
    """セッション状態の初期化"""
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'data_analysis' not in st.session_state:
        st.session_state.data_analysis = None
    if 'business_context' not in st.session_state:
        st.session_state.business_context = {}
    if 'data_context' not in st.session_state:
        st.session_state.data_context = {}
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'llm_client' not in st.session_state:
        st.session_state.llm_client = None

def check_llm_connection():
    """LLM接続チェック"""
    try:
        if st.session_state.llm_client is None:
            st.session_state.llm_client = LocalLLMClient()
        
        if st.session_state.llm_client.check_connection():
            return True
        else:
            st.error("❌ ローカルLLMサーバーに接続できません")
            st.info("Ollamaサーバーが起動しているか確認してください: `ollama serve`")
            return False
    except Exception as e:
        st.error(f"LLM接続エラー: {e}")
        return False

def main():
    """メインアプリケーション"""
    initialize_session_state()
    
    # ヘッダー
    st.markdown('<h1 class="main-header">📊 ローカルLLM売上予測ツール</h1>', unsafe_allow_html=True)
    
    # サイドバー
    with st.sidebar:
        st.header("🔧 システム状態")
        
        # LLM接続状態
        if check_llm_connection():
            st.success("✅ LLM接続: 正常")
        else:
            st.error("❌ LLM接続: 失敗")
        
        st.markdown("---")
        
        # 進行状況
        st.header("📋 進行状況")
        steps = [
            "データ アップロード",
            "ビジネス目的 設定", 
            "データ内容 確認",
            "分析実行",
            "結果確認・ダウンロード"
        ]
        
        for i, step_name in enumerate(steps, 1):
            if st.session_state.step == i:
                st.markdown(f"**→ {i}. {step_name}**")
            elif st.session_state.step > i:
                st.markdown(f"✅ {i}. {step_name}")
            else:
                st.markdown(f"⭕ {i}. {step_name}")
        
        st.markdown("---")
        
        # リセットボタン
        if st.button("🔄 最初からやり直し", type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # メインコンテンツ
    if st.session_state.step == 1:
        step1_data_upload()
    elif st.session_state.step == 2:
        step2_business_questions()
    elif st.session_state.step == 3:
        step3_data_questions()
    elif st.session_state.step == 4:
        step4_analysis_execution()
    elif st.session_state.step == 5:
        step5_results_and_download()

def step1_data_upload():
    """ステップ1: データアップロード"""
    st.markdown('<div class="step-header"><h2>📁 ステップ1: データファイルをアップロード</h2></div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### 📋 データ要件
    - **ファイル形式**: CSV, Excel (.xlsx, .xls)
    - **必須列**: 時間を表す列（日付・時刻）、予測したい数値列
    - **推奨データ量**: 最低50行以上（多いほど精度向上）
    
    ### 📊 データ例
    | 日付 | 売上 | 来客数 |
    |------|------|--------|
    | 2023-01-01 | 150000 | 45 |
    | 2023-01-02 | 120000 | 38 |
    """)
    
    # ファイルアップロード
    uploaded_file = st.file_uploader(
        "データファイルを選択してください",
        type=['csv', 'xlsx', 'xls'],
        help="CSV形式またはExcel形式のファイルをアップロードしてください"
    )
    
    if uploaded_file is not None:
        try:
            # ファイル読み込み
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ ファイル読み込み成功: {len(df)}行 × {len(df.columns)}列")
            
            # データプレビュー
            st.subheader("📊 データプレビュー")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(df.head(10), use_container_width=True)
            
            with col2:
                st.write("**基本情報**")
                st.write(f"行数: {len(df):,}")
                st.write(f"列数: {len(df.columns)}")
                st.write(f"欠損値: {df.isnull().sum().sum():,}")
                
                st.write("**列一覧**")
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    missing = df[col].isnull().sum()
                    st.write(f"• {col} ({dtype})")
                    if missing > 0:
                        st.write(f"  欠損: {missing}件")
            
            # データ品質チェック
            st.subheader("🔍 データ品質チェック")
            
            quality_issues = []
            
            # 行数チェック
            if len(df) < 20:
                quality_issues.append("⚠️ データ行数が少なすぎます（20行未満）")
            elif len(df) < 50:
                quality_issues.append("⚠️ データ行数が推奨値を下回ります（50行未満）")
            
            # 数値列チェック
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                quality_issues.append("❌ 数値列が見つかりません")
            
            # 時間列チェック
            time_candidates = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        pd.to_datetime(df[col].dropna().iloc[:5])
                        time_candidates.append(col)
                    except:
                        pass
                elif 'date' in col.lower() or 'time' in col.lower():
                    time_candidates.append(col)
            
            if len(time_candidates) == 0:
                quality_issues.append("❌ 時間を表す列が見つかりません")
            
            # 品質判定結果
            if quality_issues:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.write("**⚠️ データ品質の問題**")
                for issue in quality_issues:
                    st.write(issue)
                st.markdown('</div>', unsafe_allow_html=True)
                
                if any("❌" in issue for issue in quality_issues):
                    st.error("重大な問題があります。データを修正してから再アップロードしてください。")
                    return
            else:
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.write("✅ **データ品質は良好です**")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # データ保存と次のステップ
            if st.button("📊 このデータで分析を開始", type="primary", use_container_width=True):
                st.session_state.data = df
                
                # データ分析実行
                analyzer = DataAnalyzer()
                data_analysis = analyzer.analyze_data_structure(df)
                st.session_state.data_analysis = data_analysis
                
                st.session_state.step = 2
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ ファイル読み込みエラー: {e}")
            st.info("ファイル形式やエンコーディングを確認してください")

def step2_business_questions():
    """ステップ2: ビジネス目的の質問"""
    st.markdown('<div class="step-header"><h2>🎯 ステップ2: ビジネス目的の設定</h2></div>', 
                unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.error("データがありません。ステップ1からやり直してください。")
        return
    
    # 対話型質問
    questioner = InteractiveQuestioner()
    
    business_context = questioner.ask_business_questions(st.session_state.data_analysis)
    
    if business_context:
        st.session_state.business_context = business_context
        
        # 設定内容の確認
        st.subheader("✅ 設定内容の確認")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**業界**")
            st.info(business_context.get('industry', '未設定'))
            
            st.write("**予測期間**")
            st.info(business_context.get('prediction_period', '未設定'))
        
        with col2:
            st.write("**ビジネス目的**")
            st.info(business_context.get('purpose', '未設定'))
            
            st.write("**精度要求**")
            st.info(business_context.get('accuracy_requirement', '未設定'))
        
        # 次のステップボタン
        if all(key in business_context for key in ['industry', 'purpose', 'prediction_period', 'accuracy_requirement']):
            if st.button("📈 データ内容の確認へ進む", type="primary", use_container_width=True):
                st.session_state.step = 3
                st.rerun()
        else:
            st.warning("全ての項目を設定してください")

def step3_data_questions():
    """ステップ3: データ内容の質問"""
    st.markdown('<div class="step-header"><h2>📊 ステップ3: データ内容の確認</h2></div>', 
                unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.error("データがありません。ステップ1からやり直してください。")
        return
    
    # データの詳細分析
    analyzer = DataAnalyzer()
    
    # 対話型質問
    questioner = InteractiveQuestioner()
    data_context = questioner.ask_data_questions(st.session_state.data, st.session_state.data_analysis)
    
    if data_context and 'time_column' in data_context and 'target_column' in data_context:
        st.session_state.data_context = data_context
        
        # 時系列パターン分析
        st.subheader("📈 時系列パターン分析")
        
        try:
            patterns = analyzer.detect_time_patterns(
                st.session_state.data, 
                data_context['time_column'], 
                data_context['target_column']
            )
            
            # パターン表示
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**データ期間**")
                date_range = patterns.get('date_range', {})
                st.info(f"{date_range.get('start', 'N/A')[:10]} ～ {date_range.get('end', 'N/A')[:10]}")
                
                st.write("**データ頻度**")
                st.info(patterns.get('frequency', '不明'))
                
                st.write("**トレンド**")
                st.info(patterns.get('trend', '不明'))
            
            with col2:
                st.write("**季節性**")
                seasonality = patterns.get('seasonality', {})
                for key, value in seasonality.items():
                    st.write(f"• {key}: {value}")
                
                st.write("**外れ値**")
                outliers = patterns.get('outliers', {})
                st.info(f"{outliers.get('count', 0)}件 ({outliers.get('percentage', 0):.1f}%)")
            
            # 可視化
            try:
                viz_path = analyzer.create_summary_visualization(
                    st.session_state.data,
                    data_context['time_column'],
                    data_context['target_column']
                )
                
                if os.path.exists(viz_path):
                    st.subheader("📊 データ可視化")
                    st.image(viz_path, caption="データ分析サマリー")
            except Exception as e:
                st.warning(f"可視化エラー: {e}")
            
        except Exception as e:
            st.error(f"時系列分析エラー: {e}")
        
        # 最終確認
        st.subheader("🔍 最終確認")
        
        confirmation = questioner.confirm_analysis_approach(
            st.session_state.business_context,
            data_context
        )
        
        if confirmation.get('confirmed', False):
            if st.button("🚀 予測分析を実行", type="primary", use_container_width=True):
                st.session_state.step = 4
                st.rerun()

def step4_analysis_execution():
    """ステップ4: 分析実行"""
    st.markdown('<div class="step-header"><h2>🚀 ステップ4: 予測分析実行中</h2></div>', 
                unsafe_allow_html=True)
    
    if not all([st.session_state.data is not None, 
                st.session_state.business_context, 
                st.session_state.data_context]):
        st.error("必要な情報が不足しています。前のステップを完了してください。")
        return
    
    # 分析実行
    with st.spinner("🔄 予測モデルを構築中です...（数分かかる場合があります）"):
        try:
            # 予測エンジン初期化
            forecast_engine = ForecastEngine()
            
            # 分析設定
            analysis_config = {
                'aggregation_level': st.session_state.data_context.get('aggregation_level', '現在のままで良い'),
                'missing_strategy': st.session_state.data_context.get('missing_strategy', '欠損値を除外して分析'),
                'prediction_period': st.session_state.business_context.get('prediction_period', '1ヶ月先')
            }
            
            # 分析実行
            results = forecast_engine.run_full_analysis(
                st.session_state.data,
                st.session_state.data_context['time_column'],
                st.session_state.data_context['target_column'],
                analysis_config
            )
            
            st.session_state.analysis_results = results
            
            # 進捗表示
            if 'error' not in results:
                st.success("✅ 予測分析が完了しました！")
                
                # 結果概要
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "使用データ数", 
                        f"{results.get('data_preparation', {}).get('prepared_rows', 0):,}行"
                    )
                
                with col2:
                    best_model = results.get('best_model', {})
                    st.metric(
                        "最良モデル", 
                        best_model.get('name', '未選択')
                    )
                
                with col3:
                    mape = best_model.get('mape', 0)
                    st.metric(
                        "予測精度 (MAPE)", 
                        f"{mape:.1f}%",
                        delta=f"目標: 10%以下" if mape <= 10 else "要改善"
                    )
                
                # 自動進行
                st.info("分析結果を確認してください...")
                
                if st.button("📊 結果を確認", type="primary", use_container_width=True):
                    st.session_state.step = 5
                    st.rerun()
                
            else:
                st.error(f"❌ 分析エラー: {results['error']}")
                
                st.subheader("🔧 トラブルシューティング")
                st.write("""
                **考えられる原因と対処法:**
                - データ数が不足している → より多くのデータを準備
                - 時間列の形式が不正 → 日付形式を確認
                - 数値列に問題がある → 欠損値や文字列混入を確認
                - メモリ不足 → データサイズを縮小
                """)
                
        except Exception as e:
            st.error(f"❌ 予測分析エラー: {e}")
            st.code(traceback.format_exc())

def step5_results_and_download():
    """ステップ5: 結果確認・ダウンロード"""
    st.markdown('<div class="step-header"><h2>📊 ステップ5: 結果確認とダウンロード</h2></div>', 
                unsafe_allow_html=True)
    
    if st.session_state.analysis_results is None:
        st.error("分析結果がありません。ステップ4を実行してください。")
        return
    
    results = st.session_state.analysis_results
    
    if 'error' in results:
        st.error(f"分析エラー: {results['error']}")
        return
    
    # 結果サマリー
    st.subheader("📈 分析結果サマリー")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        data_prep = results.get('data_preparation', {})
        st.metric(
            "使用データ数",
            f"{data_prep.get('prepared_rows', 0):,}行",
            f"元データ: {data_prep.get('original_rows', 0):,}行"
        )
    
    with col2:
        best_model = results.get('best_model', {})
        st.metric(
            "採用モデル",
            best_model.get('name', '未選択')
        )
    
    with col3:
        mape = best_model.get('mape', 0)
        delta_color = "normal" if mape <= 10 else "inverse"
        st.metric(
            "予測精度 (MAPE)",
            f"{mape:.1f}%",
            delta=f"目標: ≤10%",
            delta_color=delta_color
        )
    
    with col4:
        forecast_df = results.get('forecast')
        forecast_count = len(forecast_df) if forecast_df is not None else 0
        st.metric(
            "予測期間",
            f"{forecast_count}日分"
        )
    
    # 可視化表示
    if results.get('visualization_paths'):
        st.subheader("📊 分析結果グラフ")
        for viz_path in results['visualization_paths']:
            if os.path.exists(viz_path):
                st.image(viz_path, caption="予測分析結果", use_column_width=True)
    
    # 予測値テーブル
    forecast_df = results.get('forecast')
    if forecast_df is not None and len(forecast_df) > 0:
        st.subheader("🔢 予測値一覧")
        
        # 表示用に列名を日本語化
        display_df = forecast_df.copy()
        if len(display_df.columns) >= 4:
            display_df.columns = ['日付', '予測値', '下限値', '上限値']
        
        # 統計情報
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(display_df.head(20), use_container_width=True)
            if len(display_df) > 20:
                st.info(f"表示: 上位20件 / 全{len(display_df)}件")
        
        with col2:
            st.write("**予測値統計**")
            forecast_values = forecast_df['forecast']
            st.write(f"平均: {forecast_values.mean():.1f}")
            st.write(f"最大: {forecast_values.max():.1f}")
            st.write(f"最小: {forecast_values.min():.1f}")
            st.write(f"標準偏差: {forecast_values.std():.1f}")
    
    # ダウンロードセクション
    st.subheader("💾 結果ダウンロード")
    
    col1, col2 = st.columns(2)
    
    # CSV ダウンロード
    with col1:
        if forecast_df is not None and len(forecast_df) > 0:
            try:
                report_gen = ReportGenerator()
                csv_filename = report_gen.export_forecast_csv(forecast_df)
                
                with open(csv_filename, 'rb') as f:
                    csv_data = f.read()
                
                st.download_button(
                    label="📊 予測値をCSVダウンロード",
                    data=csv_data,
                    file_name=csv_filename,
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"CSVエクスポートエラー: {e}")
    
    # PDF レポート ダウンロード
    with col2:
        try:
            # LLM による詳細解説生成
            llm_explanation = None
            if st.session_state.llm_client and st.session_state.llm_client.check_connection():
                with st.spinner("AI解説を生成中..."):
                    llm_explanation = st.session_state.llm_client.explain_model(results)
            
            # PDF レポート生成
            report_gen = ReportGenerator()
            pdf_filename = report_gen.generate_forecast_report(
                results,
                st.session_state.business_context,
                st.session_state.data_context,
                llm_explanation
            )
            
            with open(pdf_filename, 'rb') as f:
                pdf_data = f.read()
            
            st.download_button(
                label="📄 詳細レポートをPDFダウンロード",
                data=pdf_data,
                file_name=pdf_filename,
                mime="application/pdf",
                type="secondary",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"PDFレポート生成エラー: {e}")
    
    # ユーザー質問セクション
    st.subheader("❓ 結果についての質問")
    
    questioner = InteractiveQuestioner()
    user_question = questioner.get_user_question()
    
    if user_question:
        if st.session_state.llm_client and st.session_state.llm_client.check_connection():
            with st.spinner("AI が回答を生成中..."):
                try:
                    answer = st.session_state.llm_client.explain_model(results, user_question)
                    
                    st.markdown("### 🤖 AI からの回答")
                    st.markdown(f"**質問**: {user_question}")
                    st.markdown(f"**回答**: {answer}")
                    
                except Exception as e:
                    st.error(f"AI回答生成エラー: {e}")
        else:
            st.warning("LLMサーバーに接続できないため、質問に回答できません。")
    
    # 新しい分析ボタン
    st.markdown("---")
    if st.button("🔄 新しいデータで分析を開始", type="secondary", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

if __name__ == "__main__":
    main()