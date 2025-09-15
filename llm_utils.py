import requests
import json
import logging
from typing import Dict, List, Optional

class LocalLLMClient:
    """ローカルLLM（Ollama）との接続クライアント"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2:7b-chat"):
        self.base_url = base_url
        self.model = model
        self.logger = logging.getLogger(__name__)
    
    def generate_response(self, prompt: str, system_prompt: str = None) -> str:
        """LLMから応答を生成"""
        try:
            url = f"{self.base_url}/api/generate"
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "top_k": 40
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"LLM API エラー: {e}")
            return "申し訳ございませんが、一時的に応答できません。"
        except Exception as e:
            self.logger.error(f"予期しないエラー: {e}")
            return "エラーが発生しました。"
    
    def generate_analysis_plan(self, business_context: Dict, data_features: Dict) -> str:
        """分析計画を自動生成"""
        system_prompt = """
        あなたは経験豊富なビジネスアナリストです。
        ビジネス目的とデータ特徴から最適な分析計画を策定してください。
        回答は日本語で、簡潔かつ要点を落とさないように記述してください。
        """
        
        prompt = f"""
        以下のビジネスコンテキストとデータ特徴から分析計画を作成してください：

        【ビジネスコンテキスト】
        業界: {business_context.get('industry', '不明')}
        目的: {business_context.get('purpose', '不明')}
        
        【データ特徴】
        時間軸: {data_features.get('time_column', '不明')}
        予測対象: {data_features.get('target_column', '不明')}
        データ期間: {data_features.get('date_range', '不明')}
        データ数: {data_features.get('data_count', '不明')}行
        欠損パターン: {data_features.get('missing_pattern', 'なし')}
        トレンド: {data_features.get('trend', '不明')}
        季節性: {data_features.get('seasonality', '不明')}
        
        分析計画として以下を含めてください：
        1. データ前処理方針
        2. 予測手法の選択理由
        3. 予測精度の目標
        4. ビジネス活用方法
        """
        
        return self.generate_response(prompt, system_prompt)
    
    def generate_model_code(self, analysis_plan: str, data_info: Dict) -> str:
        """予測モデルのPythonコードを自動生成"""
        system_prompt = """
        あなたは時系列予測の専門家です。
        分析計画に基づいて、Pythonで実行可能な予測モデルのコードを生成してください。
        pandas, numpy, statsmodels, prophetライブラリを使用してください。
        コードは実行可能で、エラーハンドリングも含めてください。
        """
        
        prompt = f"""
        以下の分析計画とデータ情報に基づいて、時系列予測モデルのPythonコードを生成してください：

        【分析計画】
        {analysis_plan}
        
        【データ情報】
        時間列: {data_info.get('time_column')}
        予測対象列: {data_info.get('target_column')}
        データファイル: {data_info.get('file_path')}
        
        以下の機能を含むコードを作成してください：
        1. データ読み込み・前処理
        2. 訓練・テストデータ分割
        3. 複数の予測手法（ARIMA, SARIMA, Prophet）
        4. 精度評価（MAPE計算）
        5. 予測結果の可視化
        6. 予測値のCSV出力
        
        関数形式で実装し、main()関数から呼び出せるようにしてください。
        """
        
        return self.generate_response(prompt, system_prompt)
    
    def explain_model(self, model_results: Dict, user_question: str = None) -> str:
        """モデル結果の説明を生成"""
        system_prompt = """
        あなたはビジネスアナリスト向けにデータ分析結果を分かりやすく説明する専門家です。
        統計的な用語は避け、ビジネスに影響のある内容を重点的に説明してください。
        """
        
        if user_question:
            prompt = f"""
            以下のモデル結果について、ユーザーの質問に答えてください：
            
            【モデル結果】
            {json.dumps(model_results, ensure_ascii=False, indent=2)}
            
            【ユーザーの質問】
            {user_question}
            
            ビジネスアナリストが理解しやすいように、具体的で実用的な回答をしてください。
            """
        else:
            prompt = f"""
            以下のモデル結果を分かりやすく説明してください：
            
            【モデル結果】
            {json.dumps(model_results, ensure_ascii=False, indent=2)}
            
            以下の観点から説明してください：
            1. 予測の信頼性（精度）
            2. 予測の根拠（何に基づいて予測したか）
            3. ビジネスでの活用方法
            4. 注意すべき点
            """
        
        return self.generate_response(prompt, system_prompt)
    
    def check_connection(self) -> bool:
        """LLMサーバーとの接続確認"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False