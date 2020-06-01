import collections
import math
import os
from pathlib import Path
import pickle
import re
import tqdm
from urllib.parse import urlparse


from bs4 import BeautifulSoup
from nltk import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from pyserini.search import pysearch
from pyserini.index import pyutils
from sklearn import preprocessing


CLUEWEB_INDEX='/ir/index/lucene-index.cw12b13.pos+docvectors+transformed'
data_folder = Path('~/www3/datasets').expanduser()
C = 52343021  # number of docs
avg_doclen = 651.678


def get_baseline_mapping():
    if not os.path.exists(data_folder / 'baseline.pickle'):
        baseline_documents = {}
        for www in ('www1', 'www2', 'www3'):
            temp = {}
            with open(data_folder / www / 'baselineEng.txt', 'r') as f:
                for line in f:
                    qid, _, did, _, _, _ = line.split()
                    temp.setdefault(qid, [])
                    temp[qid].append(did)
            baseline_documents[www] = temp

        with open(data_folder / 'baseline.pickle', 'wb') as f:
            pickle.dump(baseline_documents, f)
    else:
        with open(data_folder / 'baseline.pickle', 'rb') as f:
            baseline_documents = pickle.load(f)
    return baseline_documents


def get_stem_result(text):
    words = word_tokenize(text.lower())
    stopwordset = stopwords.words('english')
    puncts = [',', '.', '!', '?', '&', ';']
    clean_list = [token for token in words if token not in stopwordset and token not in puncts]
    
    porter = PorterStemmer()
    result = [porter.stem(w) for w in clean_list]
    return result


def get_queries(www):
    # Get the topics for WWW.
    query_map = {}
    with open(data_folder / www / 'topics', 'r') as f:
        soup = BeautifulSoup(f, 'html5lib')
        raw_queries = soup.find_all('query')
        for q in raw_queries:
            qid = int(q.find('qid').text)
            query = q.find('content').text
            query_map[qid] = query
    return query_map


def get_document_frequencies(queries):
    # Find the number of occurances for every term in queries.
    index_utils = pyutils.IndexReaderUtils(CLUEWEB_INDEX)
    term_doc_freq = {}
    for query in queries:
        terms = get_stem_result(query)
        for term in terms:
            # Default analyzer in anserini is porter.
            if term not in term_doc_freq:
                try:
                    df = index_utils.get_term_counts(term, analyzer=None)[0]
                    term_doc_freq[term] = df
                except Exception as e:
                    print('term', term, e)
                    continue
    return term_doc_freq


def get_relevance(www):
    rels = {}
    with open(data_folder / www /'qrels', 'r') as f:
        for line in f:
           qid, did, rel = line.split()
           rels.setdefault(int(qid), {})
           rels[int(qid)][did] = rel
    return rels


def get_urls():
    urls = {}
    with open(data_folder / 'urls.txt', 'r') as f:
        for line in f:
            did, url = line.split('\t')
            urls[did] = url
    return urls


def LMIR(mode, tf, idf, dl, du, Pw, c):
    lamda, mu, delta = 0.5, 50, 0.5
    N = len(dl)
    N_q_terms = len(idf)
    P_LMIR = np.zeros(N)
    if mode=='JM':
        for i in range(N):
            alpha = lamda
            P = 0.0
            for j in range(N_q_terms):
                if c[i,j]>0:
                    Ps = (1-lamda)*tf[i,j] + lamda*Pw[j]
                    P += np.log(Ps/alpha/Pw[j])
            P += (N_q_terms*np.log(alpha) + np.sum(np.log(Pw)))
            P_LMIR[i] = P
    elif mode=='DIR':
        for i in range(N):
            alpha = mu*1.0/(dl[i] + mu)   #float(np.sum(c[i,:])
            P = 0.0
            for j in range(N_q_terms):
                if c[i,j]>0:
                    Ps = (c[i,j]+mu*Pw[j]) / (dl[i] + mu)
                    P += np.log(Ps/alpha/Pw[j])
            P += (N_q_terms*np.log(alpha) + np.sum(np.log(Pw)))
            P_LMIR[i] = P
    elif mode=='ABS':
        for i in range(N):
            alpha = delta*du[i]/dl[i]
            # alpha = mu*1.0/(float(np.sum(c[i,:])) + mu)
            P = 0.0
            for j in range(N_q_terms):
                if c[i,j]>0:
                    Ps = max(c[i,j]-delta,0)/dl[i] + alpha*Pw[j]
                    P += np.log(Ps/alpha/Pw[j])
            P += (N_q_terms*np.log(alpha) + np.sum(np.log(Pw)))
            P_LMIR[i] = P
    return P_LMIR


def add_score(d, did, rel=0, features=None):
    d['rel'].append(rel)
    d['docs'].append(did)
    if not features:
        features = [None] * 32
    for i, v in enumerate(features):
        d[i].append(v)


def replace_whitespace(text):
    return text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ').strip()


def compute_features(query, text, term_doc_freq, features, idx, mode=None):
    text = get_stem_result(text)
    words = collections.Counter(text)
    query_dict = collections.Counter(query)
    dl = len(text)  # doclen
    dl_set = len(set(text))
    features[idx+3*4] = dl

    tfs = []
    idfs = []
    query_tfs = []
    for term in query:
        tf = words[term]
        df = term_doc_freq[term]
        idf = math.log10((C - df + 0.5) / (df + 0.5))
        tfs.append(tf)
        idfs.append(idf)
        query_tfs.append(query_dict[term])

        features[idx] += tf
        features[idx+4] += idf
        features[idx+2*4] += tf * idf
        features[idx+4*4] += (idf * tf * 3.5) / (tf + 2.5 * (1 - 0.8 + 0.8 * dl / avg_doclen))

    try:
        count = np.zeros([1, dl], dtype=np.float32)
        for j in range(len(query)):
            if query[j] not in words:
                continue
            count[0][j] = words[query[j]]
        Pw = (np.sum(count, axis=0) + 0.1) / dl

        lmir_jm = LMIR('JM', np.array([query_tfs], dtype=np.float32), np.array([idfs], dtype=np.float32), [dl], [dl_set], Pw, count)[0]
        lmir_dir = LMIR('DIR', np.array([query_tfs], dtype=np.float32), np.array([idfs], dtype=np.float32), [dl], [dl_set], Pw, count)[0]
        lmir_abs = LMIR('ABS', np.array([query_tfs], dtype=np.float32), np.array([idfs], dtype=np.float32), [dl], [dl_set], Pw, count)[0]
    except Exception as e:
        lmir_jm = lmir_dir = lmir_abs = None
    features[idx+5*4] = lmir_abs
    features[idx+6*4] = lmir_dir
    features[idx+7*4] = lmir_jm


if __name__ == '__main__':
    filename='features.txt'
    searcher = pysearch.SimpleSearcher(CLUEWEB_INDEX)
    baseline_docs = get_baseline_mapping()
    urls = get_urls()
    for www, baseline in baseline_docs.items():
        print('Processing {}'.format(www))
        queries = get_queries(www)
        term_doc_freq = get_document_frequencies(queries.values())
        if www in ('www1', 'www2'):
            rels = get_relevance(www)
        else:
            rels = None
        with open(data_folder / www / filename, 'w') as f:
            for qid, dids in tqdm.tqdm(baseline.items()):
                # For query-level normalization
                scores = {i: [] for i in range(32)}
                scores['docs'] = []
                scores['rel'] = []
                for did in dids:
                    features = [0] * 32  # 32 features for each query-doc pair
                    qid = int(qid)
                    query = queries[qid]
                    query = get_stem_result(query)
                    try:
                        raw_doc = searcher.doc(did).raw()
                    except Exception as e:
                        print('Cannot get doc', did, e)
                        add_score(scores, did)
                        continue
                    try:
                        soup = BeautifulSoup(raw_doc, 'html.parser')
                    except Exception as e:
                        soup = BeautifulSoup(raw_doc, 'html5lib')
                    [script.extract() for script in soup.find_all('script')]
                    [style.extract() for style in soup.find_all('style')]

                    
                    # whole document
                    try:
                        content = soup.get_text()
                        content = replace_whitespace(content)
                        compute_features(query, content, term_doc_freq, features, 3, 'content')
                    except Exception as e:
                        print('Cannot get content of', did, e)
                        for i in range(8):
                            features[3+i*4] = None

                    # title
                    try:
                        title = ''
                        if soup.title and soup.title.string:
                            title = soup.title.string
                        else:
                            # Try using html5lib
                            soup2 = BeautifulSoup(raw_doc, 'html5lib')
                            if soup2.title and soup2.title.string:
                                title = soup2.title.string
                        title = replace_whitespace(title)
                        title = ' '.join(re.split('[ ]+', title))
                        compute_features(query, title, term_doc_freq, features, 1, 'title')
                    except Exception as e:
                        print('Cannot get title of', did, e)
                        for i in range(8):
                            features[1+i*4] = None

                    # anchor text
                    try:
                        anchors = soup.select('a')
                        anchor_text = [anchor.text.lower().strip() for anchor in anchors]
                        anchor_text = ' '.join(anchor_text)
                        anchor_text = replace_whitespace(anchor_text)
                        compute_features(query, anchor_text, term_doc_freq, features, 0, 'anchor')
                    except Exception as e:
                        print('Cannot get anchor text of', did, e)
                        for i in range(8):
                            features[i*4] = None

                    # url
                    try:
                        url = urls[did]
                        o = urlparse(url)
                        qargs = [j for i in o.query.split('&') for j in i.split('=')]
                        url = ' '.join(o.netloc.split('.') + o.path.split('/') + qargs)
                        compute_features(query, url, term_doc_freq, features, 2, 'url')
                    except Exception as e:
                        print('Cannot get url of', did, e)
                        for i in range(8):
                            features[2+i*4] = None

                    if rels:
                        add_score(scores, did, rels[qid].get(did, 0), features)
                    else:
                        add_score(scores, did, 0, features)
                
                
                for i in range(32):
                    # query-level normalization
                    if any(j is None for j in scores[i]):
                        # Replace None with minimum values
                        try:
                            m = min([j for j in scores[i] if j is not None])
                        except ValueError:
                            m = 0
                        for j in range(len(scores[i])):
                            if scores[i][j] is None:
                                scores[i][j] = m
                    scores[i] = preprocessing.minmax_scale(scores[i]).tolist()

                for i, did in enumerate(scores['docs']):
                    f.write(' '.join([str(scores['rel'][i])] + ['qid:' + str(qid)] + ['{}:{:.6f}'.format(idx + 1, scores[idx][i]) for idx in range(32)]) + ' #docid={}\n'.format(did))

