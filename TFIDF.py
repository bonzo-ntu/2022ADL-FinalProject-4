from typing import Union, List, Set, Tuple

from scipy.sparse import csr_matrix, vstack
import numpy as np
import pandas as pd
from tqdm import tqdm
import uuid
from pathlib import Path
import pickle as pkl
from sklearn.preprocessing import normalize


class Estimator:
    """Abstract class to estimate weights, frequency or scores"""

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


class Cache:
    def __init__(self, path: str = ".cache"):
        self.uuid = uuid.uuid4().hex
        self.path = Path(path) / self.uuid
        self.path.mkdir(parents=True, exist_ok=True)


class TFIDF(Estimator, Cache):
    def __init__(self, stop_words: Union[Set, List, Tuple] = set(), norm="l2", path: str = "./.cache"):
        """
        initiate all attributes to be NULL at this moment
        the content of attributes will be generated
        """
        Estimator.__init__(self)
        Cache.__init__(self, path=path)
        self.docs = [[]]
        self.doc_lengths = np.array([])
        self.stop_words = set(stop_words)
        self._words = []
        self.tf = csr_matrix(np.array([]))  # term frequency
        self.idf = np.array([])
        self.tfidf = np.array([])
        self.norm = norm
        self.tf_df = None
        self.idf_df = None
        self.tfidf_df = None

    def __call__(self, docs: List[List[str]]):
        """
        Return TFIDF value for docs tokens (without stop words)
        """
        self.__init__(stop_words=self.stop_words, norm=self.norm, path=self.path)
        self.docs = docs
        self._preprocess()
        self._compute()
        self._dataframe()
        return self.tfidf, self.tf, self.idf, self.words

    @property
    def words(self):
        return pkl.load(open(self.path / "words.pkl", "rb"))

    @words.setter
    def words(self, _words):
        pkl.dump(_words, open(self.path / "words.pkl", "wb"))

    @property
    def tf_df(self):
        return pkl.load(open(self.path / "tf_df.pkl", "rb"))

    @words.setter
    def tf_df(self, _df):
        pkl.dump(_df, open(self.path / "tf_df.pkl", "wb"))

    @property
    def idf_df(self):
        return pkl.load(open(self.path / "idf_df.pkl", "rb"))

    @words.setter
    def idf_df(self, _df):
        pkl.dump(_df, open(self.path / "idf_df.pkl", "wb"))

    @property
    def tfidf_df(self):
        return pkl.load(open(self.path / "tfidf_df.pkl", "rb"))

    @words.setter
    def tfidf_df(self, _df):
        pkl.dump(_df, open(self.path / "tfidf_df.pkl", "wb"))

    def _remove_stop_words(self):
        """
        remove stop words from each doc
        """
        for i, doc in enumerate(self.docs):
            self.docs[i] = [word for word in doc if word not in self.stop_words]

    def _doc_lengths_after_remove_stop_words(self):
        """
        get doc length after stop words removal
        """
        self.doc_lengths = np.array([[len(doc)] for doc in self.docs]).astype("uint32")

    def _preprocess(self):
        """
        main flow of preprocess
        """
        self._remove_stop_words()
        self._doc_lengths_after_remove_stop_words()

    def _compute_tf(self):
        """
        compute tf and save as a csr_matrix
        ref: https://vimsky.com/zh-tw/examples/usage/python-scipy.sparse.csr_matrix.html
        """
        indptr = [0]
        indices = []
        data = []
        words = {}
        pbar = tqdm(self.docs)
        for doc in pbar:
            pbar.set_description(f"Processing tf")
            for term in doc:
                index = words.setdefault(term, len(words))
                indices.append(index)
                data.append(1)
            indptr.append(len(indices))

        # dict guarantee the key order after python 3.7, thus can directly change the order
        self.words = list(words.keys())
        print(f"self.words = list(words.keys()) done!")
        self.tf = csr_matrix((data, indices, indptr), dtype="uint32")  # at this moment, actually is "term counts"
        print(f"self.tf = csr_matrix((data, indices, indptr), dtype='uint32') done!")

        # doing self.tf/self.doc_lengths
        stack = []
        pbar = tqdm(range(len(self.docs)))
        for ri in pbar:
            pbar.set_description(f"Processing tf/doc_lengths")
            stack.append(self.tf[ri] / self.doc_lengths[ri].item())

        self.tf = vstack(stack)  # now you get "term frequency"
        print(f"Processing tf end")

    def _compute_idf(self):
        # self._compute_idf_small()

        # len(self.words) should equals to self.tf.shape[1]
        if len(self.words) < 1000:
            self._compute_idf_small()
        else:
            self._compute_idf_large()

    def _compute_idf_small(self):
        """
        compute idf and save as a csr_matrix
        calculate df (document frequency) first, will avoid divide by zero error
        since each word will show in at least one doc, there won't be 0 df,
        that is np.log10(df) won't be -inf

        the origin process should be
        ```
        self.idf = (self.tf > 0).sum(axis=0).astype("uint32")
        n = np.array(len(self.docs)).astype("uint32")
        self.idf = self.idf / n # document frequency
        self.idf = -np.log10(self.idf).astype("float32")
        ```

        but in order to save time, the sequence of computation must be changed
        """
        print(f"Processing idf with small word set")
        self.idf = (self.tf > 0).sum(axis=0).astype("uint32")  # see if term occurs in a doc or not, term occurrence
        log_n = np.log(len(self.docs)).astype("float32")  # number of docs
        self.idf = -np.log(self.idf).astype("float32")
        self.idf = self.idf + (log_n + 1)  # self.idf = self.idf / n , now you get "inversed document frequency"
        # self.idf = csr_matrix(self.idf)
        print(f"Processing idf with small word set end")

    def _compute_idf_large(self):
        def right_seqments(full_size, num_of_seg=100):
            start, end = [0], [full_size]  # this line is for else logic
            if (step := full_size // num_of_seg) != 0:
                start = list(range(0, full_size, step))
                end = start[1:] + [full_size]

            return start, end

        start, end = right_seqments(len(self.words), 200)
        pbar = tqdm(zip(start, end))  # len(self.words) should equals to self.tf.shape[1]

        log_n = np.log(len(self.docs)).astype("float32") + 1  # number of docs, use 'uint16' to save memory
        stack = []
        for start, end in pbar:
            pbar.set_description(f"Processing idf with large word set")

            idf_partial = (self.tf[:, start:end] > 0).sum(axis=0).astype("uint32")
            # see if term occurs in a doc or not, term occurrence, use 'uint16' to save memory
            # idf_partial = idf_partial / n  # document frequency, posond this step to the last step
            idf_partial = -np.log(idf_partial).astype("float32")
            stack.append(idf_partial)

        self.idf = np.concatenate(stack, axis=1)
        self.idf = self.idf + (log_n + 1)
        # self.idf = csr_matrix(self.idf)
        print(f"Processing idf with large word set end")
        # pospond idf_partial / n until now, now you get "inversed document frequency"

    def _compute_tfidf(self):
        """
        can't use np.multiply directly, this will generate a big talbe
        thus have to multiply one row at a time then save it as a csr_matrix
        """
        stack = []
        pbar = tqdm(range(self.tf.shape[0]))
        for i in pbar:
            pbar.set_description(f"Processing tfidf")
            stack.append(
                csr_matrix(np.multiply(self.tf[i].toarray(), self.idf)).astype("float32")
            )  # use 'float32' instead of 'double64' to save memory

        self.tfidf = vstack(stack)
        self.tfidf = normalize(self.tfidf, norm=self.norm, axis=1)

    def _compute(self):
        """
        main flow of compute
        """
        self._compute_tf()
        self._compute_idf()
        self._compute_tfidf()

    def _dataframe(self):
        self.tf_df = pd.DataFrame(self.tf.toarray(), columns=self.words)
        self.idf_df = pd.DataFrame(np.asarray(self.idf), columns=self.words)
        self.tfidf_df = pd.DataFrame(self.tfidf.toarray(), columns=self.words)


if __name__ == "__main__":
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Documents
    doc_0 = "today is a nice day"
    doc_1 = "today is a bad day"
    doc_2 = "today i want to play all day"
    doc_3 = "i went to play all day yesterday"
    doc_all = [doc_0, doc_1, doc_2, doc_3]
    docs = [doc.split(" ") for doc in doc_all]

    # TF-IDF
    vectorizer = TfidfVectorizer(smooth_idf=False)
    tfidf = vectorizer.fit_transform(doc_all)
    result = pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names_out())

    mytfidf = TFIDF(stop_words=["a", "i"])
    mytfidf(docs)
    myresult = pd.DataFrame(mytfidf.tfidf.toarray(), columns=mytfidf.words)
    myresult = myresult[result.columns]
