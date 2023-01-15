import csv
from flask import Flask, request, jsonify
import sys
from collections import Counter, OrderedDict
import itertools
from itertools import islice, count, groupby, chain
import pandas as pd
import os
import re
from operator import itemgetter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from time import time
from timeit import timeit
from pathlib import Path
import pickle
import pandas as pd
import time
import numpy as np
# from google.cloud import storage
import math

import hashlib
def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()

nltk.download('stopwords')

from inverted_index_gcp import *

read_index_body = InvertedIndex.read_index("/home/MTseytlin345/Indices/body_index","body_index")
read_index_title = InvertedIndex.read_index("/home/MTseytlin345/Indices/title_index","title_index")
read_index_anchor = InvertedIndex.read_index("/home/MTseytlin345/Indices/anchor_index","anchor_index")

id_title = pd.read_pickle('/home/MTseytlin345/DocID_Title/dict_doc_id.pickle')

body_bins_path = '/home/MTseytlin345/bins/body/'
anchor_bins_path = '/home/MTseytlin345/bins/anchor/'
title_bins_path= '/home/MTseytlin345/bins/title/'

TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1
def read_posting_list(inverted, w,file_name):
  with closing(MultiFileReader()) as reader:
    locs = inverted.posting_locs[w]
    locs = [(file_name + lo[0], lo[1]) for lo in locs]
    b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)
    posting_list = []
    for i in range(inverted.df[w]):
      doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
      tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
      posting_list.append((doc_id, tf))
    return sorted(posting_list, key = lambda x : x[1], reverse=True)

'''For Requested Answer Form : [(doc_id),(doc_title),...] '''
def titled_answer(result):
    OurResult = []
    for id,score in result:
        if(id not in id_title.keys()):
            continue
        OurResult.append((id,id_title[id]))
    return OurResult

PageViews_Dict = pd.read_pickle('/home/MTseytlin345/PageViews/PageViews.pkl')

''' reading PageRank CSV File, turning into Dictionary'''
with open('/home/MTseytlin345/PageRank/PageRank.csv','r') as f:
  Csv_Reader=csv.reader(f)
  data=list(Csv_Reader)
  PageRank_Dict=dict([(int(i),float(j)) for i,j in data])

'''Retrieving PageViews values for each Doc_ID'''
def PageViews(list_of_page_ID):
  returnlist = []
  for page_ID in list_of_page_ID:
    returnlist.append(PageViews_Dict[page_ID])
  return returnlist

'''Retrieving PageRank values for each Doc_ID'''
def PageRanks(list_of_page_ID):
  returnlist = []
  for page_ID in list_of_page_ID:
    if page_ID in PageRank_Dict:
      returnlist.append(PageRank_Dict[page_ID])
    else:
      returnlist.append(0)
  return returnlist



'''Cosine Similarity
    No Get Candidates Function, all computed inside
    returns dict : {doc_id:Cosine Similarity Score, ...}
'''
def cosine_similarity(search_query, index, path):
  cosine_similarity_dictionary = {}
  search_query = tokenize(search_query)
  for term in np.unique(search_query):
     if term in index.term_total.keys():
       postings_term = read_posting_list(index, term ,path)
       for doc_id, freq in postings_term:
         normalized_tf = freq / index.DL[doc_id]
         idf = math.log2(len(index.DL)/len(postings_term))
         if doc_id not in cosine_similarity_dictionary:
           cosine_similarity_dictionary[doc_id] = normalized_tf*idf
         else:
           cosine_similarity_dictionary[doc_id] += normalized_tf*idf
  for doc in cosine_similarity_dictionary:
     cosine_similarity_dictionary[doc] = cosine_similarity_dictionary[doc]*(1/len(search_query))*(1/index.DL[doc])
  return cosine_similarity_dictionary

def get_topN_score_for_query(sim_dict, N=100):
        result = sorted([(doc_id, score) for doc_id, score in sim_dict.items()],
                        key=lambda x: x[1],
                        reverse=True)
        result = result[:N]
        return result

def get_top_n(sim_dict,N=3):
    return sorted([(doc_id,round(score,5)) for doc_id, score in sim_dict.items()], key = lambda x: x[1],reverse=True)[:N]

import math
from itertools import chain


# When preprocessing the data have a dictionary of document length for each document saved in a variable called `DL`.
class BM25_from_index:
    def __init__(self, index, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.N = len(read_index_body.DL)
        self.AVGDL = sum(read_index_body.DL.values()) / self.N

    def calc_idf(self, list_of_tokens):
        idf = {}
        for term in list_of_tokens:
            if term in self.index.df.keys():
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                idf[term] = 0
        return idf

    def search(self, query, index, path, N=100):
        query = tokenize(query)
        Docs_To_Be_Sent_To_Score = []
        Frequency_Dict = {}
        Scores_Dict = {}
        for term in np.unique(query):
            if term in index.term_total.keys():
                postings_term = read_posting_list(index, term, path)
                for tup in postings_term:
                    Docs_To_Be_Sent_To_Score.append(tup[0])
                    Frequency_Dict[(tup[0], term)] = tup[1]
        Docs_To_Be_Sent_To_Score = set(Docs_To_Be_Sent_To_Score)
        for doc_id in Docs_To_Be_Sent_To_Score:
            Scores_Dict[doc_id] = self._score(query, doc_id, Frequency_Dict)
        result = get_top_n(Scores_Dict, N)
        # result = [(id,id_title[id]) for id, score in result]
        return result

    def _score(self, query, doc_id, Frequency_Dict):
        score = 0.0
        doc_len = self.index.DL[doc_id]
        self.idf = self.calc_idf(query)
        for term in query:
            if (doc_id, term) in Frequency_Dict:
                freq = Frequency_Dict[(doc_id, term)]
                numerator = self.idf[term] * freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
                score += (numerator / denominator)
        return score


RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
stopwords_frozen = frozenset(stopwords.words('english'))

stemmer = PorterStemmer()
def tokenize(text,stem=False):
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if
                      token.group() not in stopwords_frozen]
    if stem:
        stemmed_tokens = []
        for token in list_of_tokens:
            stemmed_tokens.append(stemmer.stem(token))
        list_of_tokens.extend(stemmed_tokens)
    print(np.unique(list_of_tokens))
    return list_of_tokens

'''Binary_doc Search :
    returns doc_ids where query appeared the most in them '''
def Query_Index_Match(search_query, index, path):
  returnlist = {}
  search_query = tokenize(search_query)
  for term in np.unique(search_query):
    if term in index.term_total.keys():
      postings_term = read_posting_list(index, term ,path)
      for tup in postings_term:
          if(tup[0] in returnlist):
            returnlist[tup[0]] = returnlist[tup[0]] +1
          else:
            returnlist[tup[0]] = 1
  returnlist = sorted(returnlist.items(), key=lambda x: x[1], reverse=True)
  return returnlist

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

def MergeScores(title_scores, body_scores, title_weight=0.5, text_weight=0.5, N=100):
        query_dict = {}
        for tup in body_scores:
            query_dict[tup[0]] = tup[1] * text_weight
        for tup in title_scores:
            if query_dict.get(tup[0]) is not None:
                query_dict[tup[0]] = query_dict[tup[0]] + tup[1] * title_weight
            else:
                query_dict[tup[0]] = tup[1] * title_weight
        new_dict = sorted([(doc_id, score) for doc_id, score in query_dict.items()], key=lambda x: x[1], reverse=True)[:N]
        return new_dict
@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    bm25_body = BM25_from_index(read_index_body)
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    if len(tokenize(query)) < 3:
        res = titled_answer(Query_Index_Match(query, read_index_title, title_bins_path))[:100]
    else:
        title_scores = Query_Index_Match(query, read_index_title, title_bins_path)
        body_scores = bm25_body.search(query, read_index_body, body_bins_path)
        res = titled_answer(MergeScores(title_scores, body_scores, 0.5, 0.5))
    return jsonify(res)
    # END SOLUTION

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = titled_answer(get_topN_score_for_query(cosine_similarity(query,read_index_body,body_bins_path), N=100))
    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title. For example, a document with a 
        title that matches two of the query words will be ranked before a 
        document with a title that matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = titled_answer(Query_Index_Match(query,read_index_title,title_bins_path))
    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = titled_answer(Query_Index_Match(query,read_index_anchor,anchor_bins_path))
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = PageRanks(wiki_ids)
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = PageViews(wiki_ids)
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=False)
