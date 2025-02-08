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


def retrive_semantic_recommendations(
        query: str, 
        category: str = None,
        tone: str = None,
        initial_top_k:int = 30,
        final_top_k:int = 16
) -> pd.DataFrame:
    
    records = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(record.page_content.strip('"').split()[0]) for record in records]
    book_recs = books[books['isbn13'].isin(books_list)].head(final_top_k)

    if category != 'All':
        book_recs = book_recs[book_recs['sample_categories'] == category].head(final_top_k)
    else:
        book_recs = book_recs[:final_top_k]
    
    tone_map = {
        'Happy': 'joy',
        'Surprising': 'surprise',
        'Angry': 'anger',
        'Suspenseful': 'fear',
        'Sad': 'sadness'
    }

    if tone != 'All':
        book_recs.sort_values(by=tone_map[tone], ascending=False, inplace=True)

    return book_recs


def recommend_book(query: str, category: str, tone: str):
    
    recommendations = retrive_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row['description']
        turncated_desc = description.split()
        turncated_description = ' '.join(turncated_desc[:30]) + '...'

        authors = row['authors'].split(';')
        if len(authors) > 2:
            authors = f"{', '.join(authors[:-1])} and {authors[-1]}"
        elif len(authors) == 2:
            authors = f'{authors[0]} and {authors[1]}'
        else:
            authors = authors[0]
        
        caption = f'{row['title']} by {authors}: {turncated_description}'
        results.append((row['largest_thumbnail'], caption))
    
    return results

categories = ['All'] + books['simple_categories'].unique().tolist()
tones = ['All', 'Happy', 'Surprising', 'Angry', 'Suspenseful', 'Sad']

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown('## Semantic Book Recommender')

    with gr.Row():
        query = gr.Textbox(label='Enter a book description:', placeholder='e.g. A story about forgiveness.', lines=3)
        category_dropdown = gr.Dropdown(label='Select a category:', choices=categories, value='All')
        tone_dropdown = gr.Dropdown(label='Select an emotional tone:', choices=tones, value='All')
        submit_button = gr.Button('Get Recommendations')
    
    gr.Markdown('### Recommendations')
    output = gr.Gallery(label='Recommended Books', columns=8, rows=2)

    submit_button.click(
        fn=recommend_book,
        inputs=[query, category_dropdown, tone_dropdown],
        outputs=output
    )

if __name__ == '__main__':
    dashboard.launch()

