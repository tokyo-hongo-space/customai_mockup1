import streamlit as st
from typing import Dict, List, Optional, Tuple
import pandas as pd

class InteractiveQuestioner:
    """ユーザーとの対話を管理するクラス"""
    
    def __init__(self):
        self.business_context = {}
        self.data_context = {}
        
    def ask_business_questions(self, data_info: Dict) -> Dict:
        """ビジネス目的に関する質問（重要①）"""
        st.subheader("📊 ビジネス目的の確認")
        
        # 1. 業界の確認
        industry = st.selectbox(
            "ユーザー様の業界は何になりますか？",
            [
                "選択してください",
                "医療・病院",
                "小売・EC",
                "製造業",
                "飲食業",
                "サービス業",
                "金融業",
                "教育",
                "不動産",
                "その他"
            ],
            key="industry"
        )
        
        if industry != "選択してください":
            self.business_context['industry'] = industry
            
            # 2. 予測対象の確認
            target_column = data_info.get('target_column', '不明')
            
            # 業界に基づいた目的の推定
            purpose_suggestions = self._get_purpose_suggestions(industry, target_column)
            
            st.write(f"**データから読み取れる予測対象**: {target_column}")
            
            purpose_confirmed = st.radio(
                f"ビジネス目的として、{purpose_suggestions['main']}ということでしょうか？",
                ["はい", "いいえ", "まだ決めていない"],
                key="purpose_main"
            )
            
            if purpose_confirmed == "はい":
                self.business_context['purpose'] = purpose_suggestions['main']
                self.business_context['use_case'] = purpose_suggestions['use_case']
                
            elif purpose_confirmed == "いいえ":
                # 代替目的の確認
                st.write("では、予測した値をどのように使う予定ですか？")
                
                alternative_purposes = st.multiselect(
                    "該当するものを選択してください（複数選択可）",
                    purpose_suggestions['alternatives'],
                    key="purpose_alternatives"
                )
                
                if alternative_purposes:
                    self.business_context['purpose'] = "; ".join(alternative_purposes)
                
                custom_purpose = st.text_area(
                    "その他、具体的な用途があれば教えてください",
                    key="custom_purpose"
                )
                
                if custom_purpose:
                    self.business_context['purpose'] = custom_purpose
            
            # 3. 予測期間の確認
            prediction_period = st.selectbox(
                "どの程度先まで予測したいですか？",
                [
                    "選択してください",
                    "1週間先",
                    "1ヶ月先", 
                    "3ヶ月先",
                    "6ヶ月先",
                    "1年先"
                ],
                key="prediction_period"
            )
            
            if prediction_period != "選択してください":
                self.business_context['prediction_period'] = prediction_period
            
            # 4. 精度要求の確認
            accuracy_requirement = st.radio(
                "予測精度について、どの程度の正確さが必要ですか？",
                [
                    "概算で十分（±20%程度）",
                    "ある程度正確（±10%程度）", 
                    "高精度が必要（±5%程度）"
                ],
                key="accuracy_requirement"
            )
            
            self.business_context['accuracy_requirement'] = accuracy_requirement
        
        return self.business_context
    
    def ask_data_questions(self, df: pd.DataFrame, data_analysis: Dict) -> Dict:
        """データに関する質問（重要②）"""
        st.subheader("📈 データ内容の確認")
        
        # 1. 時間列の確認
        time_candidates = data_analysis.get('time_column_candidates', [])
        
        if len(time_candidates) > 1:
            time_column = st.selectbox(
                "時間を表す列はどれですか？",
                time_candidates,
                key="time_column_select"
            )
        elif len(time_candidates) == 1:
            time_column = time_candidates[0]
            st.success(f"時間列を自動検出しました: **{time_column}**")
        else:
            st.error("時間を表す列が見つかりません。データを確認してください。")
            return {}
        
        self.data_context['time_column'] = time_column
        
        # 2. 予測対象列の確認
        numeric_columns = data_analysis.get('numeric_columns', [])
        
        if len(numeric_columns) > 1:
            target_column = st.selectbox(
                "予測したい数値の列はどれですか？",
                numeric_columns,
                key="target_column_select"
            )
        elif len(numeric_columns) == 1:
            target_column = numeric_columns[0]
            st.success(f"予測対象列を自動検出しました: **{target_column}**")
        else:
            st.error("数値列が見つかりません。データを確認してください。")
            return {}
        
        self.data_context['target_column'] = target_column
        
        # 3. データの内容確認
        column_meaning = st.text_input(
            f"「{target_column}」は具体的に何の数字ですか？",
            placeholder="例: 1日あたりの来院患者数",
            key="column_meaning"
        )
        
        if column_meaning:
            self.data_context['column_meaning'] = column_meaning
        
        # 4. 欠損値の理由確認
        missing_percentage = data_analysis.get('missing_percentage', {}).get(target_column, 0)
        
        if missing_percentage > 5:  # 5%以上の欠損値がある場合
            st.warning(f"⚠️ {target_column}列に{missing_percentage:.1f}%の欠損値があります")
            
            missing_reason = st.radio(
                "欠損値の理由として考えられるものはありますか？",
                [
                    "土日祝日は営業していない",
                    "年末年始・GWは休業",
                    "深夜時間帯はデータなし",
                    "システムの不具合",
                    "その他・不明"
                ],
                key="missing_reason"
            )
            
            self.data_context['missing_reason'] = missing_reason
            
            # 欠損値への対応方針
            missing_strategy = st.radio(
                "欠損値はどのように扱いますか？",
                [
                    "欠損値を除外して分析",
                    "0で埋める",
                    "前後の値で補間",
                    "平均値で補間"
                ],
                key="missing_strategy"
            )
            
            self.data_context['missing_strategy'] = missing_strategy
        
        # 5. データ集約レベルの確認
        frequency = data_analysis.get('frequency', '不明')
        st.info(f"現在のデータ頻度: **{frequency}**")
        
        aggregation_level = st.radio(
            "予測に使用するデータの集約レベルはどうしますか？",
            [
                "現在のままで良い",
                "週単位で集約したい",
                "月単位で集約したい",
                "日単位で集約したい"
            ],
            key="aggregation_level"
        )
        
        self.data_context['aggregation_level'] = aggregation_level
        
        # 6. 外部要因の確認
        st.subheader("🌟 外部要因の影響")
        
        external_factors = st.multiselect(
            "予測対象に影響すると思われる要因はありますか？",
            [
                "季節・天候",
                "祝日・連休",
                "キャンペーン・イベント",
                "経済状況",
                "競合他社の動向",
                "法規制の変更",
                "その他"
            ],
            key="external_factors"
        )
        
        if external_factors:
            self.data_context['external_factors'] = external_factors
        
        return self.data_context
    
    def confirm_analysis_approach(self, business_context: Dict, data_context: Dict) -> Dict:
        """分析アプローチの最終確認"""
        st.subheader("✅ 分析方針の最終確認")
        
        # サマリー表示
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**ビジネス要件**")
            st.write(f"・業界: {business_context.get('industry', '未設定')}")
            st.write(f"・目的: {business_context.get('purpose', '未設定')}")
            st.write(f"・予測期間: {business_context.get('prediction_period', '未設定')}")
            st.write(f"・精度要求: {business_context.get('accuracy_requirement', '未設定')}")
        
        with col2:
            st.write("**データ要件**")
            st.write(f"・時間列: {data_context.get('time_column', '未設定')}")
            st.write(f"・予測対象: {data_context.get('target_column', '未設定')}")
            st.write(f"・集約レベル: {data_context.get('aggregation_level', '未設定')}")
            
            if 'external_factors' in data_context:
                st.write(f"・外部要因: {', '.join(data_context['external_factors'])}")
        
        # 確認
        confirmation = st.radio(
            "この内容で分析を開始してよろしいですか？",
            ["はい、開始する", "設定を見直したい"],
            key="final_confirmation"
        )
        
        return {
            'confirmed': confirmation == "はい、開始する",
            'business_context': business_context,
            'data_context': data_context
        }
    
    def _get_purpose_suggestions(self, industry: str, target_column: str) -> Dict:
        """業界と列名から目的を推定"""
        suggestions = {
            "医療・病院": {
                "main": f"将来の{target_column}を予測して、医師・看護師の配置計画や設備準備に活用したい",
                "use_case": "スタッフィング最適化・リソース計画",
                "alternatives": [
                    "月次予算計画の策定",
                    "設備投資の判断材料",
                    "患者受入れキャパシティの検討",
                    "診療報酬の予測"
                ]
            },
            "小売・EC": {
                "main": f"将来の{target_column}を予測して、在庫管理や販売促進の計画に活用したい",
                "use_case": "在庫最適化・マーケティング計画",
                "alternatives": [
                    "仕入れ計画の策定",
                    "キャンペーン効果の測定",
                    "店舗配置の最適化",
                    "売上目標の設定"
                ]
            },
            "製造業": {
                "main": f"将来の{target_column}を予測して、生産計画や原材料調達に活用したい",
                "use_case": "生産計画・サプライチェーン最適化",
                "alternatives": [
                    "設備稼働率の計画",
                    "人員配置の最適化",
                    "品質管理の向上",
                    "コスト削減の検討"
                ]
            },
            "飲食業": {
                "main": f"将来の{target_column}を予測して、食材調達やスタッフ配置に活用したい",
                "use_case": "オペレーション最適化・食材ロス削減",
                "alternatives": [
                    "メニュー構成の検討",
                    "営業時間の最適化",
                    "季節限定商品の企画",
                    "売上目標の設定"
                ]
            }
        }
        
        # デフォルト（その他業界）
        default = {
            "main": f"将来の{target_column}を予測して、事業計画や意思決定に活用したい",
            "use_case": "事業計画・意思決定支援",
            "alternatives": [
                "予算計画の策定",
                "リソース配分の最適化",
                "目標設定と評価",
                "リスク管理"
            ]
        }
        
        return suggestions.get(industry, default)
    
    def get_user_question(self) -> Optional[str]:
        """ユーザーからの質問を受け付け"""
        st.subheader("❓ 予測結果について質問がありますか？")
        
        question_type = st.selectbox(
            "質問の種類を選択してください",
            [
                "質問なし",
                "予測値の使い方について",
                "予測の信頼性について", 
                "予測の根拠について",
                "その他の質問"
            ],
            key="question_type"
        )
        
        if question_type == "質問なし":
            return None
        
        if question_type == "予測値の使い方について":
            specific_questions = [
                "この予測値を具体的にどう活用すれば良いですか？",
                "予測値をExcelでどう加工すれば良いですか？",
                "予測値から何を判断できますか？"
            ]
        elif question_type == "予測の信頼性について":
            specific_questions = [
                "この予測はどの程度信用できますか？",
                "予測精度を向上させるには何が必要ですか？",
                "予測が外れる可能性はありますか？"
            ]
        elif question_type == "予測の根拠について":
            specific_questions = [
                "なぜこの予測値になったのですか？",
                "どのような要因が影響していますか？",
                "上司に説明するにはどう伝えれば良いですか？"
            ]
        else:
            specific_questions = []
        
        if specific_questions:
            selected_question = st.selectbox(
                "具体的な質問を選択してください",
                ["選択してください"] + specific_questions,
                key="selected_question"
            )
            
            if selected_question != "選択してください":
                return selected_question
        
        # カスタム質問
        custom_question = st.text_area(
            "その他、何でも気になることを入力してください",
            placeholder="例: 来月のイベントの影響はどう考慮されていますか？",
            key="custom_question"
        )
        
        return custom_question if custom_question else None