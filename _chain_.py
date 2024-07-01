import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
# from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import TextLoader
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
import os
import json
from _prompts import Content_Assistant

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OPENAI_API_KEY"] = ''

embeddings = OpenAIEmbeddings()

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get('OPENAI_API_KEY'),
)


class ContentAssistant_run:
    def __init__(self):
        self.history = []

    def save_history(self):
        with open('chat_history.json', 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=4)

    def load_history(self):
        if os.path.exists('chat_history.json'):
            with open('chat_history.json', 'r', encoding='utf-8') as f:
                self.history = json.load(f)

    def intent_detection(self, chat_history, user_input):
        messages = [
            {'role': 'user', 'content': Content_Assistant.ROUTER_PROMPT.format(chat_history=chat_history, user_input=user_input)}
        ]
        response = client.chat.completions.create(
            model='gpt-4o',
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content.strip(" \n")

    def tech_prompt(self, chat_history, user_input):
        messages = [
            {'role': 'user', 'content': Content_Assistant.TECH_PROMPT.format(chat_history=chat_history, user_input=user_input)}
        ]
        response = client.chat.completions.create(
            model='gpt-4o',
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content.strip(" \n")

    def method_prompt(self, chat_history, user_input):
        messages = [
            {'role': 'user', 'content': Content_Assistant.METHOD_PROMPT.format(chat_history=chat_history, user_input=user_input)}
        ]
        response = client.chat.completions.create(
            model='gpt-4o',
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content.strip(" \n")

    def context_prompt(self, chat_history, user_input):
        messages = [
            {'role': 'user', 'content': Content_Assistant.CONTEXT_PROMPT.format(chat_history=chat_history, user_input=user_input)}
        ]
        response = client.chat.completions.create(
            model='gpt-4o',
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content.strip(" \n")

    def collect_messages(self, user_input, chat_history, file_txt):
        self.load_history()
        messages = chat_history.copy()
        messages.append(
            {'role': 'user', 'content': Content_Assistant.CALL_SUPPORT_PROMPT.format(file_txt=file_txt, chat_history=chat_history, user_input=user_input)}
        )
        response = client.chat.completions.create(
            model='gpt-4',
            messages=messages,
            temperature=0
        )
        agent_response = response.choices[0].message['content']
        self.history.append({'role': 'assistant', 'content': agent_response})
        self.save_history()

        if 'END_OF_CONVERSATION' in agent_response:
            messages.append(
                {'role': 'assistant', 'content': 'Cảm ơn bạn đã kết nối với chúng tôi, chúc bạn một ngày tốt lành'}
            )
            return "Assistant: Cảm ơn bạn đã kết nối với chúng tôi, chúc bạn một ngày tốt lành"

        intent = self.intent_detection(user_input, chat_history)

        if 'OUT_OF_CONTEXT' in intent:
            messages.append(
                {'role': 'user', 'content': Content_Assistant.CALL_SUPPORT_PROMPT.format(file_txt=file_txt, chat_history=chat_history, user_input=user_input)}
            )
            agent_response = 'Xin lỗi khách hàng và nói về nhiệm vụ của bạn'
        elif 'GREETING' in intent:
            agent_response = 'Xin chào! Tôi là chatbot của Mekong AI, sẵn sàng hỗ trợ bạn về những vấn đề liên quan đến content như kỹ thuật viết content, phương pháp viết content, hướng dẫn bạn viết bài content. Bạn cần giúp đỡ về vấn đề gì?'
        elif 'TECH_ENQUIRY' in intent:
            tech_details = self.tech_prompt(user_input, chat_history)
            agent_response = f'Cung cấp thông tin chi tiết về kỹ thuật content: {tech_details}'
        elif 'METHOD_ENQUIRY' in intent:
            method_details = self.method_prompt(user_input, chat_history)
            agent_response = f'Cung cấp thông tin chi tiết về phương pháp viết content: {method_details}'
        elif 'CONTEXT_ENQUIRY' in intent:
            context_details = self.context_prompt(user_input, chat_history)
            agent_response = f'Viết bài content theo yêu cầu: {context_details}'
        else:
            agent_response = "Xin lỗi, tôi không thể hiểu được yêu cầu của bạn."

        self.history.append({'role': 'assistant', 'content': agent_response})
        self.save_history()

        return agent_response
