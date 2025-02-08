import numpy as np 
import pandas as pd 

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

import gradio as gr

books = pd.read_csv('data/books_cleaned_with_emotions.csv')
books['largest_thumbnail'] = books['thumbnail'] + '&fife=w800'
books['largest_thumbnail'] = np.where(
    books['largest_thumbnail'].isna(),
    'data/cover-not-found.jpg',
    books['largest_thumbnail']
)

row_documents = TextLoader('data/tagged_description.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator='\n')
documents = text_splitter.split_documents(row_documents)
db_books = Chroma.from_documents(
    documents,
    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)

