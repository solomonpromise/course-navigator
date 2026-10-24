"""Lexical retrieval with attributed excerpts, kept separate from neural routing."""
from __future__ import annotations
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .data import ROOT


class LessonIndex:
    """Rank course sections by TF-IDF similarity plus a small topic preference.

    The neural model predicts a topic. This lexical index selects actual source
    excerpts; we do not claim that the classifier generates or explains answers.
    Indexing the full public library is normal retrieval, not classifier fitting.
    """
    def __init__(self):
        self.sections = json.loads((ROOT / "data/sections.json").read_text())
        self.vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode", stop_words="english",
                                        ngram_range=(1, 1), max_features=8000, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([row["title"]+" "+row["text"] for row in self.sections])

    def search(self, text: str, topic: int | None = None, limit: int = 4):
        """Return literal excerpts and relevance scores, never invented passages."""
        query = self.vectorizer.transform([text])
        scores = (self.matrix @ query.T).toarray().ravel()
        boosts = np.array([1.15 if row["label"] == topic else 1. for row in self.sections])
        order = (scores * boosts).argsort()[::-1][:limit]
        return [dict(self.sections[i], relevance=float(scores[i])) for i in order if scores[i] > 0]

    def export_browser(self):
        """Persist sparse normalized vectors and the train-independent IDF index."""
        rows = []
        for i, section in enumerate(self.sections):
            vector = self.matrix.getrow(i)
            rows.append(dict(id=section['id'], title=section['title'], topic=section['topic'],
                             label=section['label'], lesson=section['lesson'], url=section['url'],
                             text=section['text'], indices=vector.indices.tolist(), values=vector.data.tolist()))
        return dict(vocabulary={word:int(index) for word,index in self.vectorizer.vocabulary_.items()}, idf=self.vectorizer.idf_.tolist(), sections=rows,
                    stop_words=sorted(self.vectorizer.get_stop_words()))
