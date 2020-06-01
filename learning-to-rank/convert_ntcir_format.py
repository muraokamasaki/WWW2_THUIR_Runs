def idx_to_did(feature_path):
    # Get document ids as ordered in the features file.
    query_to_docs = {}
    with open(feature_path, 'r') as f:
        for line in f:
            features, comment = line.split('#')
            qid = features.split()[1]
            qid = int(qid.split(':')[1])
            query_to_docs.setdefault(qid, [])
            did = comment.split('=')[1].strip()
            query_to_docs[qid].append(did)
    return query_to_docs


def read_ranklib_output(output_path, did_map):
    # Sort the scored output of ranklib in descending order.
    query_to_scores = {}
    with open(output_path, 'r') as f:
        for line in f:
            qid, idx, score = line.split()
            qid = int(qid)
            query_to_scores.setdefault(qid, [])
            did = did_map[qid][int(idx)]
            query_to_scores[qid].append((float(score), did))
    for qid, doc_score in query_to_scores.items():
        doc_score.sort(reverse=True)
    return query_to_scores


def generate_ntcir_format(output_path, query_to_scores, run_name):
    # Write the scores in the NTCIR format to a file.
    sorted_topics = sorted(query_to_scores.keys())
    with open(output_path, 'w') as f:
        f.write('<SYSDESC><SYSDESC>\n')
        for qid in sorted_topics:
            for idx, pair in enumerate(query_to_scores[qid], 1):
                score, did = pair
                f.write('{:04d} 0 {} {} {} {}\n'.format(qid, did, idx, score, run_name))


def convert_anserini_to_ntcir(input_file, output_file, run_name):
    with open(input_file, 'r') as f, open(output_file, 'w') as g:
        g.write('<SYSDESC><SYSDESC>\n')
        for line in f:
            qid, _, did, rank, score, _ = line.split()
            g.write('{:04d} 0 {} {} {} {}\n'.format(int(qid), did, rank, score, run_name))


if __name__ == '__main__':
    query_to_docs = idx_to_did('datasets/www3/features4.txt')
    for idx, model_num in enumerate(('2', '3', '4', '6'), 1):
        query_to_scores = read_ranklib_output('results/result_{}.txt'.format(model_num), query_to_docs)
        run_name = 'SLWWW-E-CO-REP-{}'.format(idx)
        generate_ntcir_format(run_name + '.txt', query_to_scores, run_name)
    convert_anserini_to_ntcir('results/bm25tuned.txt', 'SLWWW-E-CD-NEW-5.txt', 'SLWWW-E-CD-NEW-5')

