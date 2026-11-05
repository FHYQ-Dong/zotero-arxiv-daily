import numpy as np
from sentence_transformers import SentenceTransformer
from paper import ArxivPaper
from datetime import datetime
from loguru import logger

def rerank_paper(candidate:list[ArxivPaper],corpus:list[dict],model:str='avsolatorio/GIST-small-Embedding-v0') -> list[ArxivPaper]:
    encoder = SentenceTransformer(model)
    candidate_feature = encoder.encode([paper.summary for paper in candidate])
    # Each Zotero collection is its own interest: rank the candidates against each collection
    # separately and take from those rankings in turn, so a small collection still gets its share.
    groups = {}
    for paper in corpus:
        for path in paper['paths'] or ['<no collection>']:
            groups.setdefault(path, []).append(paper)
    logger.info("Ranking against: " + ", ".join(f"{path} ({len(g)})" for path, g in groups.items()))
    rankings = []
    best_score = {}
    for group in groups.values():
        #sort group by date, from newest to oldest
        group = sorted(group,key=lambda x: datetime.strptime(x['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),reverse=True)
        time_decay_weight = 1 / (1 + np.log10(np.arange(len(group)) + 1))
        time_decay_weight = time_decay_weight / time_decay_weight.sum()
        group_feature = encoder.encode([paper['data']['abstractNote'] for paper in group])
        sim = encoder.similarity(candidate_feature,group_feature) # [n_candidate, n_group]
        scores = ((sim * time_decay_weight).sum(axis=1) * 10).tolist() # [n_candidate]
        for s,c in zip(scores,candidate):
            best_score[id(c)] = max(s, best_score.get(id(c), s))
        rankings.append([c for _,c in sorted(zip(scores,candidate),key=lambda x: x[0],reverse=True)])
    for c in candidate:
        c.score = best_score[id(c)]
    interleaved, seen = [], set()
    for row in zip(*rankings):
        for c in row:
            if id(c) not in seen:
                seen.add(id(c))
                interleaved.append(c)
    return interleaved
