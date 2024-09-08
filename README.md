# Gradio を使った LLM チャットボット

## 機能概要

管理栄養士としての役割を果たす AI チャットボットが、チャットの内容（お悩み）に基づいて、最適なレシピを考案してくれるチャットルームです。利用者が食材や健康状態、食事の目的などを入力すると、AI がそれに合わせたレシピを提案します。

## 使用技術

- **言語**: Python
- **Web インターフェース**: Gradio
- **AI モデル**: OpenAI GPT-3.5

## セットアップ手順

1. **Python のセットアップ**

   - Python 3.7 以上をインストールしてください。公式サイト: [Python.org](https://www.python.org/)

2. **リポジトリをクローン**

   ```bash
   git clone git@github.com:Nekomusume346/llm_app.git
   cd gradio-llm-chatbot
   ```

3. **必要なライブラリをインストール**

   - 必要な Python ライブラリをインストールします。

   ```bash
   pip install -r requirements.txt
   ```

4. **API キーの設定**

   - プロジェクトのルートディレクトリに `.env` ファイルを作成し、OpenAI の API キーを設定します。

   ```plaintext
   OPENAI_API_KEY=sk-XXXXXXXXX  # あなたのAPIキーをここに記入してください
   GRADIO_USER=user #チャットルームに入るためのユーザー名を指定
   GRADIO_PASSWORD=password  #チャットルームに入るためのパスワードを設定
   ```

5. **アプリケーションの起動**
   ```bash
   python app.py
   ```
   - 上記のコマンドでアプリケーションが起動します。ローカルホスト（例: `http://127.0.0.1:7860/`）にアクセスして、チャットボットを利用できます。

## 使い方

1. ブラウザで表示された Gradio インターフェースにアクセスします。
2. テキストボックスにお悩み（例: 「朝食に適した高タンパクで低カロリーのレシピを教えてください」）を入力します。
3. 送信ボタンを押して、AI 管理栄養士のレシピ提案を受け取ります。

## サンプルコード

以下は、Gradio を使用して AI チャットボットを構築するためのサンプルコードです。

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from langchain.prompts import PromptTemplate
import gradio as gr
import os
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

# 環境変数からAPIキーを取得
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(temperature=1.0, model='gpt-3.5-turbo')

# 管理栄養士としてのプロンプトテンプレートを定義
template = """
あなたは管理栄養士です。以下の情報を元に、健康的でおいしいレシピを提案してください。
ユーザーの要望: {user_input}
"""

prompt = PromptTemplate(
    input_variables=["user_input"],
    template=template
)

def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))

    # ユーザーの入力に基づいてプロンプトを生成
    formatted_prompt = prompt.format(user_input=message)
    history_langchain_format.append(HumanMessage(content=formatted_prompt))

    gpt_response = llm(history_langchain_format)
    return gpt_response.content

gr.ChatInterface(predict).launch()
```
