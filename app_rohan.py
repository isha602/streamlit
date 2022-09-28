# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:52:47 2022

@author: bhati
"""

import haystack
import PyPDF2
#from haystack.preprocessor.preprocessor import PreProcessor
import logging
logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)
from haystack.nodes import PreProcessor
from haystack.utils import clean_wiki_text, convert_files_to_docs, fetch_archive_from_http, print_answers
from haystack.nodes import FARMReader, TransformersReader
import os
#launching elasticsearch
from haystack.utils import launch_es
launch_es()

from haystack.document_stores import ElasticsearchDocumentStore

from haystack.nodes import DensePassageRetriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline

import PyPDF2
import re
import glob
import pandas as pd
from pprint import pprint
import requests

preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=False,
    split_by="word",
    split_length=100,
    split_respect_sentence_boundary=True)

class driver:
    
    def __init__(self):
        pass
    
    def print_answer(answers):
        fields = ["answer", "score"]  # "context",
        answers = answers["answers"]
        filtered_answers = []
        
        for ans in answers:
            filtered_ans = {
                field: getattr(ans, field)
                for field in fields
                if getattr(ans, field) is not None
            }
            filtered_answers.append(filtered_ans)
        return filtered_answers
     
    def driver_function(self,file,question):
        self.file=file
        self.question=question
        df = pd.read_excel(file)
        df = df.iloc[:50]
        # df = df[['restaurant_id','reviewText']]
        df['restaurant_id'] = df['restaurant_id'].astype(str)
        df['reviewText']= df['reviewText'].fillna('')
        df.columns =['name', 'content']
        df2=df.to_dict('records')
        docs = preprocessor.process(df2)
        document_store_es = ElasticsearchDocumentStore(host="localhost", index="test",similarity="dot_product")
        document_store_es.delete_documents()
        document_store_es.write_documents(docs)
        retriever_es = DensePassageRetriever(document_store_es)
        # print('Updating.')
        document_store_es.update_embeddings(retriever=retriever_es)
        reader = FARMReader(model_name_or_path="distilbert-base-uncased-distilled-squad", use_gpu=True, num_processes=0)
        pipe = ExtractiveQAPipeline(reader, retriever_es)        
        prediction = pipe.run(query=question, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 1}})
        answers = driver.print_answer(prediction)
        return answers
    
    

#driver_ins = driver().driver_function("C:/Users/bhati/Downloads/Reviews & Rest IDs.xlsx",'Which is the best pizza?') 


import streamlit as st
def main_page():
    st.write("# Welcome to Docparser👋")
    # st.sidebar.markdown("# Upload a document")
    pdf_file=st.file_uploader('Upload the .pdf file',type=['csv','xlsx'])
    st.write("# Ask a Question")
    question=st.text_input("Ask a Question:")
    # text_obj=pdf_qa().pdf_text(pdf_file)

    # st.button("Click to get answer")
    if st.button("Click to get answer"):
        answer=driver().driver_function(pdf_file, question)
        # st.header()
        st.write(answer)
      
page_names_to_funcs = {"Upload and Query": main_page}
selected_page = st.sidebar.selectbox("", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
