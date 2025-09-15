#!/bin/bash

# ローカルLLM売上予測ツール 実行スクリプト

echo "=== ローカルLLM売上予測ツール 起動 ==="

# 1. Ollama サーバー起動確認
echo "🔍 Ollama サーバー状態確認中..."
if ! pgrep -x "ollama" > /dev/null; then
    echo "📡 Ollama サーバーを起動中..."
    nohup ollama serve > ollama.log 2>&1 &
    sleep 3
    echo "✅ Ollama サーバー起動完了"
else
    echo "✅ Ollama サーバーは既に起動中です"
fi

# 2. LLMモデル確認
echo "🤖 LLMモデル確認中..."
if ! ollama list | grep -q "llama2:7b-chat"; then
    echo "📥 必要なLLMモデルをダウンロード中..."
    ollama pull llama2:7b-chat
fi

# 3. Python環境確認
echo "🐍 Python環境確認中..."
if ! conda info --envs | grep -q "sales_forecast"; then
    echo "❌ Python環境 'sales_forecast' が見つかりません"
    echo "setup.sh を実行して環境を構築してください"
    exit 1
fi

# 4. 仮想環境有効化
echo "🔧 Python仮想環境を有効化中..."
eval "$(conda shell.bash hook)"
conda activate sales_forecast

# 5. 必要なパッケージ確認
echo "📦 必要なパッケージ確認中..."
python -c "
import streamlit
import pandas
import numpy
import statsmodels
import prophet
import requests
import reportlab
print('✅ 全てのパッケージが利用可能です')
" 2>/dev/null || {
    echo "📦 不足パッケージをインストール中..."
    pip install -r requirements.txt
}

# 6. ディレクトリ構造確認
echo "📁 ディレクトリ構造確認中..."
if [ ! -d "utils" ]; then
    echo "❌ utilsディレクトリが見つかりません"
    echo "必要なファイルが配置されているか確認してください"
    exit 1
fi

# 7. Streamlit アプリケーション起動
echo ""
echo "🚀 Streamlit アプリケーションを起動します..."
echo "ブラウザで http://localhost:8501 にアクセスしてください"
echo ""
echo "停止するには Ctrl+C を押してください"
echo ""

# Streamlit設定
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=localhost

# アプリケーション起動
streamlit run app.py

echo ""
echo "=== アプリケーションを終了しました ==="