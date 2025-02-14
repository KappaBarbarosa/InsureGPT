import streamlit as st
import google.generativeai as genai
import chromadb

# 設定 Google Gemini API
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

# 建立 Gemini 2.0 Flash 模型
model = genai.GenerativeModel("gemini-2.0-flash")

from FlagEmbedding import FlagModel
embedding_model = FlagModel(
    'chuxin-llm/Chuxin-Embedding',
    query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
    use_fp16=True  # 使用半精度計算，加速推理
)

# 初始化 ChromaDB
chroma_client = chromadb.EphemeralClient()
collection = chroma_client.get_or_create_collection(name="insurance_rag_test", embedding_function=None)

# 初始化 Streamlit 頁面
st.title("💬 AI 保險顧問")
st.write("🔍 與 AI 互動，詢問台灣保險相關問題。")

# 建立 Chat 歷史（包含 System Prompt）
if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[
        {
            "role": "model",
            "parts": [(
                "背景設定: 你是一位精通台灣保險法規與保險商品的專業顧問。\n"
                "請根據使用者提供的檢索內容，僅在內容範圍內回答問題。\n"
                "若資訊不足或無法從中找到答案，請直接說明無法回答。\n"
                "請使用條列式或表格回答問題，以確保清楚易讀。\n"
                "請勿捏造資訊，若無法回答，請提供合理建議。\n"
            )]
        }
    ])

# 用戶輸入區
user_input = st.text_input("💡 你的問題（例如：什麼是強制險？）", "")



# 定義一個函數來生成向量
def get_chuxin_embedding(text):
    return embedding_model.encode([text])[0].tolist() 

if st.button("送出"):
    if user_input:
        # 進行 RAG 查詢
        expanded_query = f"台灣保險相關問題: {user_input}"
        query_embedding = get_chuxin_embedding(expanded_query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=10  
        )
        
        retrieved_docs = "\n\n---\n\n".join([doc for doc in results["documents"][0]])
        print(retrieved_docs)
        # 新的用戶問題
        user_message = (
            f"以下是可能與『{user_input}』相關的資訊 (來自知識庫檢索結果)：\n\n"
            f"{retrieved_docs}\n\n"
            "請根據以上內容回答下列問題，並在需要時提供必要的保險或法規解釋。\n"
            "回答時要將知識庫檢索結果當成你本來就知道的內容，並將問題視為無關之第三人所問的問題。\n\n"
            f"問題: {user_input}"
        )
        print(user_message)
        # 發送訊息到 Chat（保留對話上下文）
        chat = st.session_state.chat
        response = chat.send_message(user_message)

        # 顯示回應
        st.markdown(f"### 🤖 AI 回應：\n{response.text}")

        # 更新對話紀錄
        st.session_state.chat = chat
    else:
        st.warning("請輸入問題後再點擊送出！")

st.subheader("💬 聊天記錄")

for msg in st.session_state.chat.history:
    role = "user" if msg.role == "user" else "assistant"
    if role == "assistant" and "背景設定" in msg.parts[0].text:
        continue
    if role == "user":
        question_start = msg.parts[0].text.find("問題: ")
        if question_start != -1:
            msg_text = msg.parts[0].text[question_start:]
        else:
            msg_text = msg.parts[0].text
    else:
        msg_text = msg.parts[0].text
    with st.chat_message(role):
        st.markdown(msg_text)
    

