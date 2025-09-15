#!/bin/bash

# ローカルLLM売上予測ツール 環境構築スクリプト
# MacBook Air M3 2024 対応

echo "=== ローカルLLM売上予測ツール 環境構築開始 ==="

# 1. Homebrew インストール確認
if ! command -v brew &> /dev/null; then
    echo "Homebrewをインストールしています..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# 2. Python環境（conda）のインストール
if ! command -v conda &> /dev/null; then
    echo "Minicondaをインストールしています..."
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
    bash Miniconda3-latest-MacOSX-arm64.sh -b -p $HOME/miniconda
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.zshrc
    source ~/.zshrc
    rm Miniconda3-latest-MacOSX-arm64.sh
fi

# 3. Ollamaのインストール（ローカルLLM実行環境）
if ! command -v ollama &> /dev/null; then
    echo "Ollamaをインストールしています..."
    brew install ollama
fi

# 4. conda環境作成
echo "Python仮想環境を作成しています..."
conda create -n sales_forecast python=3.10 -y
conda activate sales_forecast

# 5. 必要なPythonパッケージのインストール
echo "必要なパッケージをインストールしています..."
pip install streamlit pandas numpy scikit-learn matplotlib seaborn plotly
pip install statsmodels prophet requests ollama-python reportlab

# 6. ローカルLLMモデルのダウンロード
echo "ローカルLLMモデル（Llama2 7B）をダウンロードしています..."
ollama pull llama2:7b-chat

echo "推奨LLMモデル（CodeLlama）もダウンロードしています..."
ollama pull codellama:7b-instruct

# 7. Ollamaサーバー起動
echo "Ollamaサーバーを起動しています..."
nohup ollama serve > ollama.log 2>&1 &

# 8. プロジェクトディレクトリ構造作成
mkdir -p sales_forecast_tool/{data,models,reports,utils}
cd sales_forecast_tool

# 9. 設定完了メッセージ
echo ""
echo "=== 環境構築完了 ==="
echo "次のステップ:"
echo "1. cd sales_forecast_tool"
echo "2. conda activate sales_forecast"
echo "3. streamlit run app.py"
echo ""
echo "使用可能なLLMモデル:"
echo "- llama2:7b-chat (汎用対話)"
echo "- codellama:7b-instruct (コード生成)"
echo ""
echo "Ollamaサーバーはバックグラウンドで実行中です"