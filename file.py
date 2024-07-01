from create_datavector import retrieve_knowledge, text_load, get_chunk, vector_data
files = ['quy_trinh_viet_content.txt']
documents = text_load(files)
text_chunks = get_chunk(documents)
vector_store = vector_data(text_chunks)
file_txt_content = ' '.join([doc.page_content for doc in documents])
print("Vector store created and documents loaded.")
collection_name = 'documents'
query = 'Xác định 5W - 1H '
file_txt_content = retrieve_knowledge(query, collection_name)
print("File: ", file_txt_content)

