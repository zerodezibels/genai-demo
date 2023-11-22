from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
import time

import openai
import os
import langchain
import streamlit as st

from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings


os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

openai.api_key = os.environ['OPENAI_API_KEY']

# account for deprecation of LLM model
import datetime
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"

st.set_page_config(page_title="GenAI Demo", page_icon=":robot:", layout = "wide")

file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)

# With magic:
#st.session_state


@st.cache_resource
def load_db(param=1):
    embeddings = OpenAIEmbeddings()
    docs = loader.load()

    db = DocArrayInMemorySearch.from_documents(
        docs,
        embeddings
    )

    retriever = db.as_retriever()

    llm = ChatOpenAI(temperature = 0.0, model=llm_model)

    qa_stuff = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        verbose=True
    )

    return qa_stuff


st.header("GenAI Product Search Demo")

#option_search_type = st.radio('Search Type',['Text', 'Image'])

if 'qry' not in st.session_state:
    st.session_state['qry'] = 'dummy'


input_text = st.text_area(label="Search Query...", label_visibility='collapsed', placeholder="Your Query...", key="query_input")
format_option = st.radio('Output Format', ['Descriptive', 'Short'])
button_clicked = st.button('Search Products', type='primary')



if button_clicked:
    if input_text != st.session_state['qry']:
        st.session_state['qry'] = input_text
        query = input_text
        if format_option == 'Descriptive':
            query_append = query + '. Return response in markdown table format and summarize each one along with product code. Add addtitional column on why this product is recommended'
        else:
            query_append = query + '. Return response in markdown table format and provide short summary for each one along with product code'

        st.write("Actual Query submitted to ChatGPT")
        st.write(query_append)

        with st.spinner('Getting Response from ChatGPT'):
            qa_stuff = load_db(1)
            response = qa_stuff.run(query_append)

        st.success('Here are the results')

        st.markdown(response)

    else:
        st.write("No New Query Submitted")
