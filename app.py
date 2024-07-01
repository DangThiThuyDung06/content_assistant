import streamlit as st
from _chain_ import ContentAssistant_run
from _prompts import Content_Assistant
from qdrant_client import QdrantClient
from create_datavector import retrieve_knowledge, text_load, get_chunk, vector_data, create_collection
from qdrant_client.models import PointStruct, VectorParams

collection_name = "new_collection_06"

client = QdrantClient(url='http://localhost:6333')

files = ['quy_trinh_viet_content.txt']  # Thay thế bằng đường dẫn thực tế của bạn
documents = text_load(files)
text_chunks = get_chunk(documents)

# Create collection
create_collection(client, collection_name)

# Upsert data into the collection
vector_store = vector_data(text_chunks, collection_name)

st.title("Content Assistant Bot")
st.write("Chào mừng bạn đến với Content Assistant Bot")

bot = ContentAssistant_run()

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

user_input = st.text_input("Bạn: ", key="user_input")

if st.button("Gửi"):
    query = user_input

    file_txt_content = retrieve_knowledge(query, collection_name)

    # Lấy lịch sử trò chuyện từ session state
    chat_history = st.session_state.get('messages', [])

    bot_response = bot.collect_messages(
        user_input=user_input,
        chat_history=chat_history,
        file_txt=file_txt_content
    )

    # Cập nhật lịch sử trò chuyện vào session state
    st.session_state['messages'].append({"role": "user", "content": user_input})
    st.session_state['messages'].append({"role": "assistant", "content": bot_response})

    for message in st.session_state['messages']:
        if message['role'] == 'user':
            st.write(f"**Bạn:** {message['content']}")
        else:
            st.write(f"**Assistant:** {message['content']}")
