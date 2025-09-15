# ローカルLLM売上予測ツール

ビジネス・アナリスト向けの売上予測ツールです。ローカルLLMを活用し、ユーザーとの対話を通じて自動的に予測モデルを構築します。

## 🚀 主な機能

- **対話型分析**: ツール側からの問いかけで分析計画を自動策定
- **複数モデル対応**: ARIMA、SARIMA、Prophetモデルを自動選択
- **ローカル処理**: データは外部に送信されず、機密保持が可能
- **ビジネス用語対応**: 統計知識不要で操作可能
- **詳細レポート**: PDF形式での分析結果出力

## 💻 対応環境

- **OS**: macOS 15.6.1以降
- **ハードウェア**: Apple MacBook Air M3 2024（推奨）
- **メモリ**: 16GB以上（推奨）
- **Python**: 3.10以降

## 📦 インストール手順

### 1. 環境構築スクリプトの実行

```bash
# リポジトリをクローンまたはダウンロード
git clone <repository_url>
cd sales_forecast_tool

# 環境構築スクリプトを実行（初回のみ）
chmod +x setup.sh
./setup.sh
```

### 2. 手動インストール（スクリプトが失敗した場合）

```bash
# Minicondaのインストール
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh

# Python仮想環境の作成
conda create -n sales_forecast python=3.10 -y
conda activate sales_forecast

# 必要パッケージのインストール
pip install -r requirements.txt

# Ollamaのインストール（ローカルLLM）
brew install ollama

# LLMモデルのダウンロード
ollama pull llama2:7b-chat
ollama pull codellama:7b-instruct
```

## 🏃‍♂️ 使用方法

### 1. システム起動

```bash
# 1. ローカルLLMサーバー起動（別ターミナル）
ollama serve

# 2. 仮想環境の有効化
conda activate sales_forecast

# 3. アプリケーション起動
cd sales_forecast_tool
streamlit run app.py
```

### 2. ブラウザでアクセス

アプリケーション起動後、ブラウザで `http://localhost:8501` にアクセス

### 3. 分析手順

1. **データアップロード**: CSV/Excelファイルをアップロード
2. **ビジネス目的設定**: 業界・目的・精度要求を設定
3. **データ内容確認**: 時間列・予測対象列・集約方法を設定
4. **分析実行**: 自動で複数モデルを構築・評価
5. **結果確認**: 予測値ダウンロード・詳細レポート出力

## 📊 対応データ形式

### 必須要件
- **ファイル形式**: CSV, Excel (.xlsx, .xls)
- **時間列**: 日付・時刻を表す列（例: 2023-01-01, 2023/1/1）
- **数値列**: 予測対象となる数値データ列
- **最小行数**: 20行以上（50行以上推奨）

### データ例

```csv
日付,売上,来客数
2023-01-01,150000,45
2023-01-02,120000,38
2023-01-03,180000,52
```

## 🔧 トラブルシューティング

### よくある問題と解決方法

#### 1. LLM接続エラー
```bash
# Ollamaサーバーの状態確認
ollama list

# サーバー再起動
pkill ollama
ollama serve
```

#### 2. メモリ不足エラー
- データサイズを縮小（行数を減らす）
- より軽量なモデルを使用: `ollama pull llama2:7b-chat`

#### 3. 予測精度が低い（MAPE > 20%）
- データ期間を延長
- 欠損値処理方法を変更
- データ集約レベルを調整

#### 4. 時間列が認識されない
- 日付形式を統一（YYYY-MM-DD推奨）
- 列名に「date」「time」を含める
- 日付列のデータ型を確認

## 📈 性能指標

### 予測精度（MAPE）の目安
- **優秀**: ≤ 10%（高精度での意思決定可能）
- **良好**: 10-20%（実用的な精度）
- **要改善**: > 20%（参考値として活用）

### 推奨データ量
- **最小**: 20行（基本的な予測）
- **推奨**: 50-200行（安定した予測）
- **理想**: 200行以上（高精度予測）

## 📋 対応業界・用途

### 主要対応業界
- **医療・病院**: 患者数予測、リソース計画
- **小売・EC**: 売上予測、在庫管理
- **製造業**: 生産計画、需要予測  
- **飲食業**: 来客数予測、食材調達
- **サービス業**: 利用者数予測、人員配置

### 活用シーン
- 月次・四半期予算計画
- リソース配分の最適化
- 在庫・調達計画
- 人員配置計画
- リスク管理・対策立案

## 🔒 セキュリティ・プライバシー

- **ローカル処理**: データは外部サーバーに送信されません
- **機密保持**: すべての処理がローカル環境で完結
- **データ保護**: アップロードファイルはセッション終了時に削除

## 🛠️ 技術仕様

### 使用技術
- **フロントエンド**: Streamlit
- **機械学習**: scikit-learn, statsmodels, Prophet
- **LLM**: Ollama (Llama2, CodeLlama)
- **データ処理**: pandas, numpy
- **可視化**: matplotlib, seaborn, plotly
- **レポート**: ReportLab

### アーキテクチャ
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   Data Analysis │    │   Forecast      │
│   (Frontend)    │───▶│   (pandas/numpy)│───▶│   (ML Models)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Local LLM     │    │   Interactive   │    │   Report Gen    │
│   (Ollama)      │    │   Questioner    │    │   (PDF/CSV)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📞 サポート・連絡先

### 技術的な問題
1. GitHub Issuesで問題を報告
2. ログファイルの確認：`ollama.log`
3. エラーメッセージのスクリーンショットを添付

### 機能要望・改善提案
- GitHub Discussionsで議論
- 具体的な用途・業界での活用例を共有

## 📚 参考資料

### 時系列予測について
- [ARIMA モデルの基礎](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average)
- [Prophet: 時系列予測ツール](https://facebook.github.io/prophet/)
- [時系列分析入門](https://www.statsmodels.org/stable/tsa.html)

### ビジネス活用
- 需要予測の実践
- データドリブン経営
- BI（ビジネスインテリジェンス）活用

## 🔄 更新履歴

### v1.0.0 (2024-09-14)
- 初期リリース
- 基本的な時系列予測機能
- 対話型UI実装
- PDF/CSVレポート出力
- ローカルLLM統合

## 📄 ライセンス

MIT License - 詳細は`LICENSE`ファイルを参照

## 🙏 謝辞

- [Streamlit](https://streamlit.io/) - Webアプリケーションフレームワーク
- [Ollama](https://ollama.ai/) - ローカルLLM実行環境
- [Prophet](https://facebook.github.io/prophet/) - 時系列予測ライブラリ
- [pandas](https://pandas.pydata.org/) - データ処理ライブラリ