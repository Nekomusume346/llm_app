from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import gradio as gr
import os
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
# load_dotenv()

# 明示的に.envファイルのパスを指定して読み込む
load_dotenv(dotenv_path=".env")

# 環境変数からAPIキーとログイン情報を取得
api_key = os.getenv("OPENAI_API_KEY")
gradio_user = os.getenv("GRADIO_USER")
gradio_password = os.getenv("GRADIO_PASSWORD")

# デバッグ用出力
print("Gradio User:", gradio_user)
print("Gradio Password:", gradio_password)

llm = ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo')


# プライベートドキュメントのサンプルを準備
documents = [
    {"content": "会社の新しい休暇ポリシーは、年間20日の有給休暇を提供します。"},
    {"content": "次の四半期の目標は、売上を10%増加させることです。"},
    {"content": "社内のITサポートチームへの連絡先は内線1234です。"}
]

# 文書をベクトル化してFAISSでインデックスを作成
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts([doc["content"] for doc in documents], embeddings)

# VectorStoreRetrieverを作成
retriever = vectorstore.as_retriever()

# プライベートデータを使った質問応答システムを構築
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# ログイン情報の検証
def authenticate(username, password):
    if username == gradio_user and password == gradio_password:
        return True
    else:
        return False

# チャットボットの応答を生成する関数
def predict(message, history):
    # プライベートドキュメントから関連情報を検索し、回答を生成
    result = qa_chain.run(message)
    return result

# Gradioインターフェースを定義
def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## 社内マニュアル応答チャットボット - ログイン")
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
                return "ログイン成功！", gr.update(visible=True)
            else:
                return "ユーザー名またはパスワードが間違っています。", gr.update(visible=False)

        # ボタンをクリックしたときにhandle_loginを呼び出す
        login_button.click(
            handle_login, 
            inputs=[username, password], 
            outputs=[login_status, chat_interface]
        )

    return demo

# アプリケーションを起動
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
