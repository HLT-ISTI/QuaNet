from collections import OrderedDict, Counter
import itertools
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import pickle
from os.path import join, exists
import os
import ntpath
from datetime import datetime
from copy import copy

from feature_selection.round_robin import RoundRobin

UNK = 'UNK'

def fetch_reviews(file):
    with open(file, 'rt', encoding='utf-8', errors='ignore') as fin:
        documents = []
        for line in fin:
            line = line.strip()
            if line:
                fields = line.split('\t')
                year,userid = fields[0].split()
                text = '\n'.join(fields[1:-1])
                rating = int(fields[-1])
                documents.append((year, userid, text, rating))
        return documents


def binarize_star_rating(reviews, split_point = 3, filter_split_point=True):
    return [(data,user,text,1 if rating > split_point else 0) for (data,user,text,rating) in reviews if not filter_split_point or rating != 3]


def group_by_date(reviews):
    """
    Groups reviews by date. Some dates are strings of the form '2005' while others are of the form '2005Aug'. This
    method deals with both formats (the entire list is expected to follow only one format).
    :param reviews: list of reviews, each being a tuple (date,user,text,label)
    :return: a sorted list of dates, and a (paired) list of reviews w/o the date (user,text,label)
    """
    dates = set([review[0] for review in reviews])
    totime = lambda date: datetime.strptime(date, '%Y%b') if len(date)==7 else datetime.strptime(date, '%Y')
    datetimes = sorted(map(totime,dates))
    od = OrderedDict([(date,[]) for date in datetimes])
    for review in reviews:
        date = totime(review[0]) # date in datetime format
        od[date].append(review[1:]) # userid, text, and label
    return zip(*list(od.items()))

def inspect_prevalences(dates, reviews, sentiment_labels=[0, 1], printfile=None):
    print('date', 'docs', sentiment_labels, file=printfile)
    all_reviews=0
    all_label_count = Counter()
    for i,date in enumerate(dates):
        n_reviews = len(reviews[i])
        label_count = Counter([r[-1] for r in reviews[i]])
        print(date, n_reviews, ' '.join([str(label_count[l]) for l in sentiment_labels]), file=printfile)
        all_reviews += n_reviews
        all_label_count += label_count
    print('Total',all_reviews,' '.join([str(all_label_count[l]) for l in sentiment_labels]), file=printfile)


class ReviewsDataset:

    def __init__(self,Xtr,ytr,Xte,yte,vocabulary,classes):
        self.Xtr = Xtr
        self.ytr = ytr
        self.Xte = Xte
        self.yte = yte
        self.vocabulary = vocabulary
        self.classes = classes

    def save(self, file):
        pickle.dump(self, open(file, 'wb'), pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, file):
        return pickle.load(open(file, 'rb'))

    def format(self):
        return 'seq' if isinstance(self.Xtr, list) else 'mat'

    def limit_vocabulary(self, max_words = 10000):
        if self.format() == 'seq':
            if len(self.vocabulary) > max_words:
                wordid_count = Counter(itertools.chain.from_iterable(self.Xtr)) #count all word-ids
                most_common_ids, _ = zip(*wordid_count.most_common(max_words-1)) #hold one value for the UNK
                tokeep_ids = set(most_common_ids) # keep only top-n (as a set)
                id2word = dict((id,word) for word,id in self.vocabulary.items()) #converts from id2word using the old vocabulary

                new_vocab = dict(zip([id2word[id] for id in tokeep_ids], range(max_words)))
                new_vocab[UNK] = len(new_vocab)

                def reconvert(sequence):
                    return [(new_vocab[id2word[id]] if id in tokeep_ids else new_vocab[UNK]) for id in sequence]

                self.Xtr = list(map(reconvert, self.Xtr))
                if self.ismultipletest(): #Xte is a list of test sets
                    self.Xte = [list(map(reconvert, Xte_i)) for Xte_i in self.Xte]
                else: #Xte is a single test set
                    self.Xte = list(map(reconvert, self.Xte))

                self.vocabulary = new_vocab
        else:
            # apply round-robin feature selection with information gain as the TSR
            fs = RoundRobin(k=max_words)
            self.Xtr = fs.fit_transform(self.Xtr, self.ytr)
            if self.ismultipletest():
                for i, Xtei in enumerate(self.Xte):
                    self.Xte[i] = fs.transform(Xtei)
            else:
                self.Xte = fs.transform(self.Xte)

        return self.get_devel_set(), self.get_test_set()

    def num_features(self):
        return self.Xtr.shape[1]

    def num_categories(self):
        if len(self.ytr.shape)==1:
            return 1
        return self.ytr.shape[1]

    def get_devel_set(self):
        return self.Xtr, self.ytr

    def get_test_set(self, index=None):
        if index:
            return self.Xte[index], self.yte[index]
        return self.Xte, self.yte

    def ismultipletest(self):
        if self.format() == 'seq':
            return isinstance(self.Xte[0][0],list)
        else:
            return isinstance(self.Xte,list)


def build_online_dataset(reviews_path, outdir, polarity_split_point=None, filterout_splitpoint=False):
    """
    Builds a series of datasets for quantification in two formats.
    Each dataset is associated with a date (e.g., '2005'); the test set corresponds to documents of that date, and the
    training set corresponds to documents before that date (e.g., from '1999' to '2004' included).
    Two representations are built: one which is in matrix form (tfidf-matrix) and other which is in sequential form
    (i.e., each document is a list of ids).
    The vocabulary and labels are the same for them both. The vocabulary is built exclusively on the training set, so
        unobserved terms in test will be ignored in the matrix representation, or replaced by <UNK> (id) token in the
        sequential representation.
    The datasets are given a code as name which follows this nomenclature:
    - <Seq> or <Mat>: sequential or matrix format
    - <TESTDATE> (.e.g, 2011_9 for September 2011): test date-slot, previous date-slots compose the training set
    - S<NUM>[F] or 5stars: the former indicates binary format (positive vs negative) with split point at NUM; an F
        following the code means the split point has been filtered out. 5stars instead means ordinar regression.
    A log is created describing some statistics of the dataset.
    :param reviews_path: path to the file containing the reviews, in format: <date> <user> \t <text> \t <rating> \n ...
    :param outdir: directory where the Datasets will be dumped
    :param polarity_split_point: if specified, binarizes the labels around this value; e.g., a 4 rating will be converted
            to 1 (positive) if polarity_split_point=3; if not specified, the 5star-rating is preserved (default: None)
    :param filterout_splitpoint: if True, reviews labelled with polarity_split_point will be discarded; if False, then
            the polarity_split_point will be considered as negative. Only used when polarity_split_point is not None
    :return:
    """

    config = ('OnlineS%d%s' % (polarity_split_point, 'F' if filterout_splitpoint else '') if polarity_split_point else '5stars')
    with open(join(outdir, config + '.log'), 'w') as loginfo_file:

        if not exists(outdir):
            os.makedirs(outdir)

        documents = fetch_reviews(reviews_path)
        print('Building online dataset for {} with split_point {} and filter {}'.format(reviews_path, polarity_split_point, filterout_splitpoint), file=loginfo_file)
        print('Read {} documents from file {}'.format(reviews_path, len(documents)), file=loginfo_file)
        print(documents[:10])

        if polarity_split_point:
            documents = binarize_star_rating(documents, polarity_split_point, filterout_splitpoint)
            sentiment_labels = np.array([0,1])
        else:
            sentiment_labels = np.array([1, 2, 3, 4, 5])

        ordered_dates, ordered_reviews = group_by_date(documents)
        inspect_prevalences(ordered_dates, ordered_reviews, sentiment_labels, loginfo_file)

        print('test\t#training\t#test\t#vocabulary', file=loginfo_file)
        for i in range(1,len(ordered_dates)):
            test_date = ordered_dates[i]
            print('Building {} (complete {}/{})'.format(test_date, i - 1, len(ordered_dates) - 1))

            text_label = lambda user_text_label:(user_text_label[1],user_text_label[2])
            train_docs, train_labels = zip(*[text_label(r) for r in itertools.chain.from_iterable(ordered_reviews[:i])])
            test_docs, test_labels = zip(*[text_label(r) for r in ordered_reviews[i]])
            classes = np.array(sentiment_labels)
            train_labels = np.array(train_labels)
            test_labels = np.array(test_labels)

            #tfidf-version
            tfidf_vect = TfidfVectorizer(min_df=5, sublinear_tf=True) # stop_words='english' ?
            X_train = tfidf_vect.fit_transform(train_docs)
            X_test = tfidf_vect.transform(test_docs)
            vocabulary = tfidf_vect.vocabulary_

            date_code = str(test_date.year)+'_'+str(test_date.month)
            outpath = join(outdir,'Mat' + date_code + config +'.pkl')
            ReviewsDataset(X_train, train_labels, X_test, test_labels, vocabulary, classes).save(outpath)


            #sequence-version
            analyzer = tfidf_vect.build_analyzer()
            vocabulary[UNK] = len(vocabulary)
            def seq2ids(sequence):
                return [vocabulary[token] if token in vocabulary else vocabulary[UNK] for token in analyzer(sequence)]
            S_train = [seq2ids(sequence) for sequence in train_docs]
            S_test  = [seq2ids(sequence) for sequence in test_docs]

            outpath = join(outdir, 'Seq' + date_code + config + '.pkl')
            ReviewsDataset(S_train, train_labels, S_test, test_labels, vocabulary, classes).save(outpath)

            print('{}\t{}\t{}\t{}'.format(test_date,len(train_docs),len(test_docs),len(vocabulary)), file=loginfo_file)
            loginfo_file.flush()


def build_single_dataset(reviews_path, outdir, training_slots=3, polarity_split_point=None, filter_split_point=False):
    """
    Builds one dataset consisting of one training set and a series of tests sets, in two formats.
    The training set includes the first n date-slots (parameter); the test sets is a list including the remaining date-slots.
    Two representations are built: one which is in matrix form (tfidf-matrix) and other which is in sequential form
    (i.e., each document is a list of ids).
    The vocabulary and labels are the same for them both. The vocabulary is built exclusively on the training set, so
        unobserved terms in test will be ignored in the matrix representation, or replaced by <UNK> (id) token in the
        sequential representation.
    The datasets are given a code as name which follows this nomenclature:
        - <Seq> or <Mat>: sequential or matrix format
        - Single<NUM>: single training composed by NUM date-slots (the rest are test sets)
        - S<NUM>[F] or 5stars: the former indicates binary format (positive vs negative) with split point at NUM; an F
            following the code means the split point has been filtered out. 5stars instead means ordinar regression.
    A log is created describing some statistics of the dataset.
    :param reviews_path: path to the file containing the reviews, in format: <date> <user> \t <text> \t <rating> \n ...
    :param outdir: directory where the Datasets will be dumped
    :param training_slots: number of date-slots to include in the training set (default: 3)
    :param polarity_split_point: if specified, binarizes the labels around this value; e.g., a 4 rating will be converted
            to 1 (positive) if polarity_split_point=3; if not specified, the 5star-rating is preserved (default: None)
    :param filter_split_point: if True, reviews labelled with polarity_split_point will be discarded, if False, then
            the polarity_split_point will be considered as negative. Only used when polarity_split_point is not None
    :return:
    """

    if not exists(outdir):
        os.makedirs(outdir)

    config = ('Single%dS%d%s' % (training_slots, polarity_split_point, 'F' if filter_split_point else '') if polarity_split_point else '5stars')
    with open(join(outdir, config + '.log'), 'w') as loginfo_file:

        documents = fetch_reviews(reviews_path)
        print('Building single dataset with {} date-slots for training, for {} with split_point {} and filter {}'
              .format(training_slots, reviews_path, polarity_split_point, filter_split_point), file=loginfo_file)
        print('Read {} documents from file {}'.format(reviews_path, len(documents)), file=loginfo_file)
        print(documents[:10])

        if polarity_split_point:
            documents = binarize_star_rating(documents, polarity_split_point, filter_split_point)
            sentiment_labels = np.array([0,1])
        else:
            sentiment_labels = np.array([1, 2, 3, 4, 5])

        ordered_dates, ordered_reviews = group_by_date(documents)
        inspect_prevalences(ordered_dates, ordered_reviews, sentiment_labels, loginfo_file)

        print('test\t#training\t#test\t#vocabulary', file=loginfo_file)
        text_label = lambda user_text_label: (user_text_label[1], user_text_label[2])
        train_docs, train_labels = zip(*[text_label(r) for r in itertools.chain.from_iterable(ordered_reviews[:training_slots])])
        classes = np.array(sentiment_labels)
        train_labels = np.array(train_labels)

        tfidf_vect = TfidfVectorizer(min_df=5, sublinear_tf=True)  # stop_words='english' ?
        X_train = tfidf_vect.fit_transform(train_docs)
        vocabulary = copy(tfidf_vect.vocabulary_)

        analyzer = tfidf_vect.build_analyzer()
        vocabulary[UNK] = len(vocabulary)

        def seq2ids(sequence):
            return [vocabulary[token] if token in vocabulary else vocabulary[UNK] for token in analyzer(sequence)]

        S_train = [seq2ids(sequence) for sequence in train_docs]

        X_test = []
        S_test = []
        Labels_test = []

        for i in range(training_slots,len(ordered_dates)):
            test_date = ordered_dates[i]
            print('Building {} (complete {}/{})'.format(test_date, i - training_slots, len(ordered_dates) - training_slots))

            test_docs, test_labels = zip(*[text_label(r) for r in ordered_reviews[i]])
            Labels_test.append(np.array(test_labels))

            #tfidf-version
            X_test.append(tfidf_vect.transform(test_docs))

            #sequence-version
            S_test.append([seq2ids(sequence) for sequence in test_docs])

            print('{}\t{}\t{}\t{}'.format(test_date,len(train_docs),len(test_docs),len(vocabulary)), file=loginfo_file)
            loginfo_file.flush()

        outpath = join(outdir, 'Mat' + config + '.pkl')
        ReviewsDataset(X_train, train_labels, X_test, Labels_test, vocabulary, classes).save(outpath)

        outpath = join(outdir, 'Seq' + config + '.pkl')
        ReviewsDataset(S_train, train_labels, S_test, Labels_test, vocabulary, classes).save(outpath)


if __name__ == '__main__':

    datasets_dir = '../../datasets/'
    #build_online_dir = join(datasets_dir, 'build', 'online')
    build_single_dir = join(datasets_dir, 'build', 'single')

    hp = join(datasets_dir, 'HP_reviews.txt')
    kindle = join(datasets_dir, 'Kindle_reviews.txt')

    #build_online_dataset(hp, join(build_online_dir, 'hp'), polarity_split_point=3, filterout_splitpoint=True)
    #build_online_dataset(kindle, join(build_online_dir, 'kindle'), polarity_split_point=3, filterout_splitpoint=True)

    build_single_dataset(hp, join(build_single_dir, 'hp'), training_slots=3, polarity_split_point=3, filter_split_point=True)
    #build_single_dataset(kindle, join(build_single_dir, 'kindle'), training_slots=3, polarity_split_point=3, filter_split_point=True)

    # data = Dataset.load('/home/moreo/pytecs/datasets/build/online/hp/Seq2008_1OnlineS3F.pkl')
    # data.limit_vocabulary(1000)
    # pass