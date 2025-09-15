from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfutils
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List
import os
from datetime import datetime

class ReportGenerator:
    """予測分析レポート生成クラス"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_japanese_font()
        
    def setup_japanese_font(self):
        """日本語フォント設定（システムフォント使用）"""
        try:
            # macOSの場合
            if os.path.exists('/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc'):
                font_path = '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc'
                pdfmetrics.registerFont(TTFont('Japanese', font_path))
            # その他の環境用のフォールバック
            else:
                # デフォルトフォントを使用（日本語は表示されない可能性あり）
                pass
        except:
            # フォント登録に失敗した場合はデフォルトを使用
            pass
    
    def generate_forecast_report(self, analysis_results: Dict, business_context: Dict, 
                               data_context: Dict, llm_explanation: str = None) -> str:
        """予測分析レポートの生成"""
        
        filename = f"forecast_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        doc = SimpleDocTemplate(filename, pagesize=A4)
        story = []
        
        # タイトル
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # 中央揃え
        )
        
        story.append(Paragraph("売上予測分析レポート", title_style))
        story.append(Spacer(1, 20))
        
        # 1. 実行サマリー
        story.append(Paragraph("1. 実行サマリー", self.styles['Heading2']))
        
        summary_data = self._create_summary_section(analysis_results, business_context, data_context)
        story.extend(summary_data)
        story.append(Spacer(1, 20))
        
        # 2. 分析計画とその理由
        story.append(Paragraph("2. 分析計画", self.styles['Heading2']))
        
        plan_data = self._create_analysis_plan_section(business_context, data_context)
        story.extend(plan_data)
        story.append(Spacer(1, 20))
        
        # 3. 予測結果
        story.append(Paragraph("3. 予測結果", self.styles['Heading2']))
        
        results_data = self._create_results_section(analysis_results)
        story.extend(results_data)
        story.append(Spacer(1, 20))
        
        # 4. モデルの説明
        story.append(Paragraph("4. 予測手法の説明", self.styles['Heading2']))
        
        model_data = self._create_model_explanation_section(analysis_results)
        story.extend(model_data)
        story.append(Spacer(1, 20))
        
        # 5. 信頼性・精度
        story.append(Paragraph("5. 予測の信頼性", self.styles['Heading2']))
        
        reliability_data = self._create_reliability_section(analysis_results)
        story.extend(reliability_data)
        story.append(Spacer(1, 20))
        
        # 6. ビジネス活用方法
        story.append(Paragraph("6. ビジネス活用方法", self.styles['Heading2']))
        
        business_data = self._create_business_application_section(business_context, analysis_results)
        story.extend(business_data)
        story.append(Spacer(1, 20))
        
        # 7. 注意事項・改善案
        story.append(Paragraph("7. 注意事項と改善案", self.styles['Heading2']))
        
        notes_data = self._create_notes_section(analysis_results)
        story.extend(notes_data)
        
        # 8. LLMによる詳細解説（オプション）
        if llm_explanation:
            story.append(Spacer(1, 20))
            story.append(Paragraph("8. AI による詳細解説", self.styles['Heading2']))
            story.append(Paragraph(llm_explanation, self.styles['Normal']))
        
        # グラフの挿入
        if analysis_results.get('visualization_paths'):
            story.append(Spacer(1, 20))
            story.append(Paragraph("9. 分析結果グラフ", self.styles['Heading2']))
            
            for viz_path in analysis_results['visualization_paths']:
                if os.path.exists(viz_path):
                    img = Image(viz_path, width=6*inch, height=4.5*inch)
                    story.append(img)
                    story.append(Spacer(1, 10))
        
        # PDFビルド
        doc.build(story)
        
        return filename
    
    def _create_summary_section(self, analysis_results: Dict, business_context: Dict, 
                              data_context: Dict) -> List:
        """サマリーセクション作成"""
        content = []
        
        # サマリーテーブル
        summary_info = [
            ['項目', '内容'],
            ['業界', business_context.get('industry', '未設定')],
            ['予測対象', data_context.get('column_meaning', data_context.get('target_column', ''))],
            ['データ期間', self._get_data_range(analysis_results)],
            ['使用データ数', f"{analysis_results.get('data_preparation', {}).get('prepared_rows', 0)}行"],
            ['最良モデル', analysis_results.get('best_model', {}).get('name', '未選択')],
            ['予測精度(MAPE)', f"{analysis_results.get('best_model', {}).get('mape', 0):.1f}%"],
            ['予測期間', business_context.get('prediction_period', '未設定')]
        ]
        
        table = Table(summary_info, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        content.append(table)
        content.append(Spacer(1, 15))
        
        # 精度評価
        mape = analysis_results.get('best_model', {}).get('mape', 100)
        if mape <= 10:
            accuracy_text = "優秀な精度です。予測値は信頼できる水準にあります。"
        elif mape <= 20:
            accuracy_text = "良好な精度です。実用的な水準の予測が可能です。"
        else:
            accuracy_text = "精度に改善の余地があります。参考値として活用し、継続的な改善が必要です。"
        
        content.append(Paragraph(f"<b>精度評価:</b> {accuracy_text}", self.styles['Normal']))
        
        return content
    
    def _create_analysis_plan_section(self, business_context: Dict, data_context: Dict) -> List:
        """分析計画セクション作成"""
        content = []
        
        # 分析方針
        content.append(Paragraph("<b>分析方針:</b>", self.styles['Normal']))
        
        plan_points = []
        
        # データ前処理方針
        aggregation = data_context.get('aggregation_level', '現在のままで良い')
        if aggregation != '現在のままで良い':
            plan_points.append(f"• データ集約: {aggregation}")
        
        missing_strategy = data_context.get('missing_strategy', '欠損値を除外して分析')
        plan_points.append(f"• 欠損値処理: {missing_strategy}")
        
        # 外部要因
        external_factors = data_context.get('external_factors', [])
        if external_factors:
            plan_points.append(f"• 考慮する外部要因: {', '.join(external_factors)}")
        
        # 精度目標
        accuracy_req = business_context.get('accuracy_requirement', '')
        if accuracy_req:
            plan_points.append(f"• 精度要求: {accuracy_req}")
        
        for point in plan_points:
            content.append(Paragraph(point, self.styles['Normal']))
        
        content.append(Spacer(1, 10))
        
        # 計画理由
        content.append(Paragraph("<b>この計画を選択した理由:</b>", self.styles['Normal']))
        
        reasons = []
        if data_context.get('missing_reason'):
            reasons.append(f"• 欠損値の理由: {data_context['missing_reason']}")
        
        if aggregation != '現在のままで良い':
            reasons.append(f"• {aggregation}を選択した理由: データの安定性と予測精度の向上")
        
        purpose = business_context.get('purpose', '')
        if purpose:
            reasons.append(f"• ビジネス目的: {purpose}")
        
        for reason in reasons:
            content.append(Paragraph(reason, self.styles['Normal']))
        
        return content
    
    def _create_results_section(self, analysis_results: Dict) -> List:
        """予測結果セクション作成"""
        content = []
        
        if analysis_results.get('forecast') is not None:
            forecast_df = analysis_results['forecast']
            
            if len(forecast_df) > 0:
                # 予測値テーブル（最初の10件）
                content.append(Paragraph("<b>予測値（抜粋）:</b>", self.styles['Normal']))
                
                # テーブルデータ準備
                table_data = [['日付', '予測値', '下限', '上限']]
                
                display_rows = min(10, len(forecast_df))
                for i in range(display_rows):
                    row = forecast_df.iloc[i]
                    table_data.append([
                        str(row.iloc[0])[:10],  # 日付（日付部分のみ）
                        f"{row['forecast']:.1f}",
                        f"{row['lower_bound']:.1f}",
                        f"{row['upper_bound']:.1f}"
                    ])
                
                if len(forecast_df) > 10:
                    table_data.append(['...', '...', '...', '...'])
                
                table = Table(table_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                content.append(table)
                content.append(Spacer(1, 10))
                
                # 統計サマリー
                forecast_values = forecast_df['forecast']
                content.append(Paragraph(f"<b>予測値統計:</b>", self.styles['Normal']))
                content.append(Paragraph(f"• 平均値: {forecast_values.mean():.1f}", self.styles['Normal']))
                content.append(Paragraph(f"• 最大値: {forecast_values.max():.1f}", self.styles['Normal']))
                content.append(Paragraph(f"• 最小値: {forecast_values.min():.1f}", self.styles['Normal']))
                content.append(Paragraph(f"• 標準偏差: {forecast_values.std():.1f}", self.styles['Normal']))
        
        return content
    
    def _create_model_explanation_section(self, analysis_results: Dict) -> List:
        """モデル説明セクション作成"""
        content = []
        
        best_model = analysis_results.get('best_model')
        if best_model:
            model_name = best_model.get('name', '未選択')
            
            # モデル選択理由
            content.append(Paragraph(f"<b>選択されたモデル: {model_name}</b>", self.styles['Normal']))
            content.append(Spacer(1, 5))
            
            # モデル説明
            model_explanations = {
                'ARIMA': 'ARIMA（自己回帰和分移動平均）モデルは、過去の値とその変化に基づいて未来を予測します。トレンドや過去のパターンを学習して予測を行います。',
                'SARIMA': 'SARIMA（季節性ARIMA）モデルは、ARIMAに季節性を追加したモデルです。年間や月間の周期的なパターンを捉えて予測精度を向上させます。',
                'Prophet': 'Prophet（プロフェット）モデルは、Facebookが開発した時系列予測ツールです。トレンド、季節性、祝日効果を自動的に検出して予測を行います。'
            }
            
            explanation = model_explanations.get(model_name, 'カスタムモデル')
            content.append(Paragraph(explanation, self.styles['Normal']))
            content.append(Spacer(1, 10))
            
            # モデル性能比較
            content.append(Paragraph("<b>他のモデルとの性能比較:</b>", self.styles['Normal']))
            
            comparison_data = [['モデル', 'MAPE (%)', '採用理由']]
            
            models = analysis_results.get('models', {})
            for model_type, model_result in models.items():
                if 'evaluation' in model_result:
                    mape = model_result['evaluation'].get('mape', 0)
                    reason = "最高精度" if model_type == model_name else "精度劣る"
                    comparison_data.append([model_type, f"{mape:.1f}", reason])
            
            if len(comparison_data) > 1:
                table = Table(comparison_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                content.append(table)
        
        return content
    
    def _create_reliability_section(self, analysis_results: Dict) -> List:
        """信頼性セクション作成"""
        content = []
        
        mape = analysis_results.get('best_model', {}).get('mape', 0)
        
        # 精度レベルの判定
        if mape <= 5:
            reliability_level = "非常に高い"
            confidence_range = "±5%"
            business_impact = "高い精度で意思決定に活用可能"
        elif mape <= 10:
            reliability_level = "高い"
            confidence_range = "±10%"
            business_impact = "実用的な精度で計画立案に活用可能"
        elif mape <= 20:
            reliability_level = "中程度"
            confidence_range = "±20%"
            business_impact = "参考値として活用、継続監視が必要"
        else:
            reliability_level = "要改善"
            confidence_range = f"±{mape:.0f}%"
            business_impact = "大まかな傾向把握のみ、モデル改善が必要"
        
        content.append(Paragraph(f"<b>信頼性レベル:</b> {reliability_level}", self.styles['Normal']))
        content.append(Paragraph(f"<b>予測精度(MAPE):</b> {mape:.1f}%", self.styles['Normal']))
        content.append(Paragraph(f"<b>実際の値の範囲:</b> 予測値の{confidence_range}程度", self.styles['Normal']))
        content.append(Paragraph(f"<b>ビジネス活用度:</b> {business_impact}", self.styles['Normal']))
        
        content.append(Spacer(1, 10))
        
        # バックテスト結果
        content.append(Paragraph("<b>バックテスト結果:</b>", self.styles['Normal']))
        
        best_model_data = analysis_results.get('best_model', {}).get('model_data', {})
        evaluation = best_model_data.get('evaluation', {})
        
        if evaluation:
            mae = evaluation.get('mae', 0)
            rmse = evaluation.get('rmse', 0)
            
            content.append(Paragraph(f"• 平均絶対誤差(MAE): {mae:.2f}", self.styles['Normal']))
            content.append(Paragraph(f"• 二乗平均平方根誤差(RMSE): {rmse:.2f}", self.styles['Normal']))
            content.append(Paragraph(f"• 過去データでの予測精度: {mape:.1f}%", self.styles['Normal']))
        
        return content
    
    def _create_business_application_section(self, business_context: Dict, analysis_results: Dict) -> List:
        """ビジネス活用方法セクション作成"""
        content = []
        
        industry = business_context.get('industry', '')
        purpose = business_context.get('purpose', '')
        
        content.append(Paragraph("<b>推奨活用方法:</b>", self.styles['Normal']))
        
        # 業界別の活用方法
        industry_applications = {
            '医療・病院': [
                '• 医師・看護師の配置計画',
                '• 医療機器・病床の準備',
                '• 診療材料の発注計画',
                '• 月次・四半期予算計画'
            ],
            '小売・EC': [
                '• 商品の発注・在庫計画',
                '• スタッフのシフト調整',
                '• キャンペーン効果の測定',
                '• 店舗運営費の予算計画'
            ],
            '製造業': [
                '• 生産計画の策定',
                '• 原材料の調達計画',
                '• 設備稼働率の計画',
                '• 出荷・物流計画'
            ]
        }
        
        applications = industry_applications.get(industry, [
            '• 事業計画の策定',
            '• リソース配分の最適化',
            '• 予算計画の精度向上',
            '• リスク管理の強化'
        ])
        
        for app in applications:
            content.append(Paragraph(app, self.styles['Normal']))
        
        content.append(Spacer(1, 10))
        
        # 具体的な活用手順
        content.append(Paragraph("<b>活用手順:</b>", self.styles['Normal']))
        content.append(Paragraph("1. 予測値をCSVファイルでダウンロード", self.styles['Normal']))
        content.append(Paragraph("2. Excelで予測値を加工・集計", self.styles['Normal']))
        content.append(Paragraph("3. 現在の計画値と比較分析", self.styles['Normal']))
        content.append(Paragraph("4. 差分を基に追加対策を検討", self.styles['Normal']))
        content.append(Paragraph("5. 定期的な実績との比較で精度検証", self.styles['Normal']))
        
        return content
    
    def _create_notes_section(self, analysis_results: Dict) -> List:
        """注意事項セクション作成"""
        content = []
        
        mape = analysis_results.get('best_model', {}).get('mape', 0)
        
        # 一般的な注意事項
        content.append(Paragraph("<b>使用時の注意事項:</b>", self.styles['Normal']))
        content.append(Paragraph("• 予測は過去のパターンに基づくため、急激な環境変化には対応できません", self.styles['Normal']))
        content.append(Paragraph("• 定期的に実績と比較し、モデルの再構築を検討してください", self.styles['Normal']))
        content.append(Paragraph("• 予測値は参考値として、専門知識と組み合わせて判断してください", self.styles['Normal']))
        
        content.append(Spacer(1, 10))
        
        # 精度に基づく改善提案
        content.append(Paragraph("<b>精度改善の提案:</b>", self.styles['Normal']))
        
        if mape > 20:
            content.append(Paragraph("• データ期間を延長（より多くの履歴データの収集）", self.styles['Normal']))
            content.append(Paragraph("• 外部要因（祝日、イベント、天候等）の追加", self.styles['Normal']))
            content.append(Paragraph("• データの集約方法の見直し", self.styles['Normal']))
        elif mape > 10:
            content.append(Paragraph("• 季節性の考慮強化", self.styles['Normal']))
            content.append(Paragraph("• 外部変数の追加検討", self.styles['Normal']))
        else:
            content.append(Paragraph("• 現在の精度は十分です", self.styles['Normal']))
            content.append(Paragraph("• 定期的な再学習で精度維持を推奨", self.styles['Normal']))
        
        return content
    
    def _get_data_range(self, analysis_results: Dict) -> str:
        """データ範囲の取得"""
        data_prep = analysis_results.get('data_preparation', {})
        date_range = data_prep.get('date_range', {})
        
        if date_range:
            start = date_range.get('start', '')[:10]  # YYYY-MM-DD
            end = date_range.get('end', '')[:10]
            return f"{start} ～ {end}"
        
        return "未設定"
    
    def export_forecast_csv(self, forecast_df: pd.DataFrame, filename: str = None) -> str:
        """予測結果のCSV出力"""
        if filename is None:
            filename = f"forecast_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # 日本語列名に変更
        export_df = forecast_df.copy()
        if len(export_df.columns) >= 4:
            export_df.columns = ['日付', '予測値', '下限値', '上限値']
        
        export_df.to_csv(filename, index=False, encoding='utf-8-sig')
        
        return filename