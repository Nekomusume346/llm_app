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
gradio_user = os.getenv("GRADIO_USER")
gradio_password = os.getenv("GRADIO_PASSWORD")

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

# ログイン情報の検証
def authenticate(username, password):
    # 簡易的な認証処理（ここではユーザー名とパスワードをハードコーディング）
    if username == gradio_user and password == gradio_password:
        return True
    else:
        return False

# チャットボットの応答を生成する関数
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

# Gradioインターフェースを定義
with gr.Blocks() as demo:
    # ログイン画面
    gr.Markdown("## 管理栄養士チャットボット - ログイン")
    username = gr.Textbox(label="ユーザー名")
    password = gr.Textbox(label="パスワード", type="password")
    login_button = gr.Button("ログイン")
    login_status = gr.Textbox(label="ログインステータス", interactive=False)
    
    # チャット画面 (初期的には非表示)
    with gr.Column(visible=False) as chat_interface:
        gr.Markdown("## チャットルーム")
        chatbot = gr.ChatInterface(predict)

    # ログインボタンがクリックされたときの処理
    def handle_login(username, password):
        if authenticate(username, password):
            # ログイン成功時
            return "ログイン成功！", gr.update(visible=True)
        else:
            # ログイン失敗時
            return "ユーザー名またはパスワードが間違っています。", gr.update(visible=False)

    # ボタンをクリックしたときにhandle_loginを呼び出す
    login_button.click(
        handle_login, 
        inputs=[username, password], 
        outputs=[login_status, chat_interface]
    )

# アプリケーションを起動
demo.launch()
