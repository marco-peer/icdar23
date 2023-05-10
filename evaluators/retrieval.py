import numpy as np
import logging
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier


class Retrieval:
    def __init__(self, options={'distance' : 'euclidean'}):
        self.options = options
        self._distances = None

    def eval(self, features, labels, use_precomputed_distances=False):
        distances = self.calc_distances(features, labels, use_precomputed_distances=use_precomputed_distances)

        logger_result, csv_result = self.calc_map_from_distances(labels, distances)
        return logger_result, csv_result

    def get_precision(self, features, labels):
        def calc_weights(weights):
            w_new = []
            for w in weights:
                w_new.append([0 if i < np.finfo(np.float32).eps else 1 for i in w])
            w_new = np.array(w_new)
            assert w_new.shape == weights.shape, 'weight matrix does not have the same shape'
            return w_new

        logging.info('building up NN-Classifier')
        neigh = KNeighborsClassifier(n_neighbors=1 + 1, n_jobs=15,
                                     weights=calc_weights)  # 1 + 1 because first element has always 0 distance
        features = np.array(features)
        logging.info('KNN fitting data ({} features)'.format(features.shape))
        neigh.fit(features, labels)

        logging.info('classify data')
        pred = neigh.predict(features)
        correct = np.sum((pred == labels))
        precision = correct / len(labels)
        return precision

    def calc_distances(self, features, labels, use_precomputed_distances=False):
        def calc_weights(weights):
            w_new = []
            for w in weights:
                w_new.append([0 if i < np.finfo(np.float32).eps else 1 for i in w])
            w_new = np.array(w_new)
            assert w_new.shape == weights.shape, 'weight matrix does not have the same shape'
            return w_new

        logging.info('building up NN-Classifier')
        neigh = KNeighborsClassifier(n_neighbors=1 + 1, n_jobs=15,
                                     weights=calc_weights)  # 1 + 1 because first element has always 0 distance
        features = np.array(features)
        logging.info('KNN fitting data ({} features)'.format(features.shape))
        neigh.fit(features, labels)

        # logging.info('classify data')
        # pred = neigh.predict(features)
        # correct = np.sum((pred == labels))
        # precision = correct / len(labels)

        if use_precomputed_distances:
            assert self._distances, "self._distances is None and use_precomputed_distances is True"
            distances = self._distances
        else:
            distances = self.compute_distances(features)
        
        return distances#, precision


    # @staticmethod
    # def get_ranking_from_distances(labels, distances):
    #     rankings = []
    #     for i in range(0, len(labels)):
    #         cur_dists = distances[i, :]
    #         idxs = np.argsort(cur_dists).flatten()
    #         sorted_writers = np.array(labels)
    #         sorted_writers = sorted_writers[idxs]
    #         rankings.append(sorted_writers)

    @staticmethod
    def calc_map_from_distances(labels, distances):
        eval_range = range(1, 10)
        hard_eval, soft_eval, percentage_eval = [], [], []

        top1_correct_count = 0
        top1_wrong_count = 0
        avg_precision = []

        for i in range(0, len(labels)):
            cur_dists = distances[i, :]
            idxs = np.argsort(cur_dists).flatten()
            sorted_writers = np.array(labels)
            sorted_writers = sorted_writers[idxs]
            cur_writer = labels[i]

            cur_sum = 0.0

            correct = []

            for j in eval_range:
                correct.append(labels[idxs[0]] == labels[idxs[j]])

            # evalation
            hard, soft, percentage = [], [], []
            for j in eval_range:
                hard.append(True if np.sum(correct[:j]) == j else False)
                soft.append(True if np.sum(correct[:j]) >= 1 else False)
                percentage.append(float(np.sum(correct[:j]))/float(j))
            hard_eval.append(hard)
            soft_eval.append(soft)
            percentage_eval.append(percentage)


            # calculate average precision
            cur_writer_idxs = np.where(sorted_writers == cur_writer)[0]
            for j in range(1, len(cur_writer_idxs)):  # page with idx 0 is original page
                cur_sum = cur_sum + float(j) / cur_writer_idxs[j]
            if len(cur_writer_idxs) > 1:
                avg_precision.append(cur_sum / float(len(cur_writer_idxs) - 1))

                if labels[idxs[0]] == labels[idxs[1]]:
                    top1_correct_count = top1_correct_count + 1
                else:
                    top1_wrong_count = top1_wrong_count + 1
                    
            else:
                logging.warning("writer %d has only one page ... unable to calculate mean_average_precision, skipping"
                                % cur_writer)
        mean_average_precision = np.mean(avg_precision)
        top1_precsision = top1_correct_count / float(top1_correct_count + top1_wrong_count)

        logger_result = {'map': mean_average_precision, 'top1': top1_precsision}

        csv_result = {'map': mean_average_precision, 'top1': top1_precsision}

        for i in eval_range:
            csv_result['hard-{}'.format(i)] = np.mean(np.array(hard_eval)[:, i-1])
            csv_result['soft-{}'.format(i)] = np.mean(np.array(soft_eval)[:, i-1])
            csv_result['perc-{}'.format(i)] = np.mean(np.array(percentage_eval)[:, i-1])

        return logger_result, csv_result

    def precision_recall_curve(self, features, labels, use_precomputed_distances=False):
        if use_precomputed_distances:
            assert self._distances is not None, "self._distances is None and use_precomputed_distances is True"
            distances = self._distances
        else:
            distances = self.compute_distances(features)

        sort_dist = np.sort(distances.reshape(-1))[distances.shape[0]:]
        # thresholds = np.geomspace(1, len(sort_dist) - 1, self.options.get('number_of_thresholds', 100), dtype=int)
        thresholds = np.logspace(1, np.log2(len(sort_dist) - 1), self.options.get('number_of_thresholds', 100),
                                 dtype=int, base=2)
        res = []
        if distances.shape[0] < 1000:
            for t in tqdm.tqdm(sort_dist[thresholds], desc='generating precision recall curve (for all samples)'):
                res.append(self.precision_recall_complete(distances, labels, t))

        else:
            num_samples = 100000
            rand_tuple = np.random.randint(0, distances.shape[0], (num_samples, 2), dtype=int)
            for t in tqdm.tqdm(sort_dist[thresholds],
                               desc='generating precision recall curve (taking {} pairs)'.format(num_samples)):
                true_positives, true_negatives, false_positives, false_negatives = 0, 0, 0, 0
                precision, recall = 0.0, 0.0
                for r1, r2 in rand_tuple:
                    if r1 == r2:
                        continue
                    if distances[r1, r2] < t:
                        if labels[r1] == labels[r2]:
                            true_positives += 1
                        else:
                            false_positives += 1
                    else:
                        if labels[r1] == labels[r2]:
                            false_negatives += 1
                        else:
                            true_negatives += 1
                    precision = true_positives / (true_positives + false_positives + np.finfo(float).eps)
                    recall = true_positives / (true_positives + false_negatives + np.finfo(float).eps)

                res.append({'threshold': t, 'precision': precision, 'recall': recall,
                           'true_positives': true_positives,
                            'true_negatives': true_negatives, 'false_positives': false_positives,
                            'false_negatives': false_negatives})
        return res

    @staticmethod
    def precision_recall_complete(distances, labels, threshold):
        thresh_distances = distances < threshold
        labels_mat = labels[np.newaxis, :]
        labels_mat = labels_mat.repeat(labels_mat.shape[1], axis=0)
        for r in range(labels_mat.shape[0]):
            labels_mat[r] = labels_mat[r] == labels[r]
        labels_mat = labels_mat.astype(bool)

        # logging.info('start calculating tp, tn,...')

        # minus number of elements because of diagonal
        true_positives = np.sum(np.logical_and(thresh_distances, labels_mat)) - labels_mat.shape[0]
        true_negatives = np.sum(np.logical_and(np.logical_not(thresh_distances), np.logical_not(labels_mat)))
        false_positives = np.sum(np.logical_and(thresh_distances, np.logical_not(labels_mat)))
        false_negatives = np.sum(np.logical_and(np.logical_not(thresh_distances), labels_mat))

        precision = true_positives / (true_positives + false_positives + np.finfo(float).eps)
        recall = true_positives / (true_positives + false_negatives + np.finfo(float).eps)

        return {'threshold': threshold, 'precision': precision, 'recall': recall, 'true_positives': true_positives,
                'true_negatives': true_negatives, 'false_positives':  false_positives,
                'false_negatives': false_negatives}

    def compute_distances(self, features):
        if self.options.get('distance') == 'euclidean':
            logging.info('using euclidean distance')
            self._distances = pairwise_distances(features, metric='euclidean', n_jobs=15)
        elif self.options.get('distance') == 'cosine':
            logging.info('using cosine distance')
            self._distances = pairwise_distances(features, metric='cosine', n_jobs=15)
        elif self.options.get('distance') == 'canberra':
            self._distances = pairwise_distances(features, metric='canberra', n_jobs=15)
        elif self.options.get('distance') == 'manhattan':
            self._distances = manhattan_distances(features)
        else:
            assert False, 'unkown distance function'
        return self._distances



       