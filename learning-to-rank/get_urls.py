from pathlib import Path
import pickle


data_folder = Path('~/www3/datasets').expanduser()
url_path = Path('~/rpa/ClueWeb12_B13_DocID_To_URL.txt').expanduser()


with open(data_folder / 'baseline.pickle', 'rb') as f:
    baseline_documents = pickle.load(f)

all_dids = set()
for www in baseline_documents.values():
    for dids in www.values():
        for did in dids:
            all_dids.add(did)

with open(url_path, 'r') as f, open(data_folder / 'urls.txt', 'w') as g:
    for line in f:
        did, url = line.split(', ')
        if did in all_dids:
            g.write(did + '\t' + url.strip() + '\n')
                
