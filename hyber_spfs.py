import numpy as np
import pandas as pd
import math
import itertools
from collections import OrderedDict
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split
from sklearn import metrics
from xgboost import XGBClassifier as XGBC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
import csv
import copy
from ITMO_FS.filters.multivariate.measures import JMIM, NJMIM, MRMR, CMIM, IWFS
from ITMO_FS_master.ITMO_FS.filters.multivariate import *
# from scikit_feature_master.skfeature.function.information_theoretical_based import FCBF
# from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
import os
import sys
from tqdm import tqdm  # 用于显示进度条
from sklearn.model_selection import ShuffleSplit



class SPFS:
    def __init__(self, data):
        self.data = data
        self.numInstances = len(data)
        self.numFeatures = len(data.columns) - 1
        self.su = [0] * self.numFeatures
        self.Rel = np.zeros((self.numFeatures, self.numFeatures))
        self.calculateSU()
        self.ranked_features = []
        self.gain_least = []
        self.gain_least_before = []
        # self.calculaterRel()


    def calculateSU(self):
        for i in range(self.numFeatures):
            self.su[i] = self.SU(i, self.numFeatures)

    def SU(self, index1, index2):
        ig = self.informationGain(index1, index2)
        e1 = self.entropy(index1)
        e2 = self.entropy(index2)

        if (e1 + e2) != 0:
            return (2 * ig) / (e1 + e2)
        else:
            return 1.0

    def informationGain(self, index1, index2):
        return self.entropy(index1) + self.entropy(index2) - self.jointEntropy(index1, index2)

    def jointEntropy(self, index1, index2):
        jointEntropy = 0

        m1_labels = self.data.iloc[:, index1].unique()
        m2_labels = self.data.iloc[:, index2].unique()
        p = 0
        for val2 in m2_labels:
            for val1 in m1_labels:
                p_xy = self.data[(self.data.iloc[:, index1] == val1) & (self.data.iloc[:, index2] == val2)].shape[
                           0] / self.numInstances
                # p_xy = sum((self.data.iloc[:, index1] == val1) & (self.data.iloc[:, index2] == val2)) / self.numInstances
                # p += p_xy
                if p_xy != 0:
                    jointEntropy += p_xy * math.log(p_xy, 2)

        return -jointEntropy

    def conditional_entropy(self, feature_subset, target):
        # 计算特征子集和目标变量的联合概率分布
        data = self.data
        feature_subset = [x-1 for x in feature_subset]
        joint_probs = {}
        total_samples = len(data)

        for index, row in data.iterrows():
            subset_key = tuple(row[feature_subset])
            target_value = row[target]

            if subset_key not in joint_probs:
                joint_probs[subset_key] = {'total': 0, 'counts': {0: 0, 1: 0}}
                # joint_probs[subset_key] = {'value_counts': [0,] * (data.values[:, subset_key].max() + 1), 'class_counts': {0: 0, 1: 0}}

            joint_probs[subset_key]['total'] += 1
            joint_probs[subset_key]['counts'][target_value] += 1
            # joint_probs[subset_key]['value_counts'][] += 1
            # joint_probs[subset_key]['counts'][target_value] += 1

        # 计算条件概率和条件熵
        conditional_entropy = 0
        for subset_key, values in joint_probs.items():
            p_x = values['total'] / total_samples
            p_y_given_x = [count / values['total'] for count in values['counts'].values()]

            conditional_entropy += p_x * sum([-p * math.log2(p) if p > 0 else 0 for p in p_y_given_x])

        return conditional_entropy


    def entropy(self, index):
        entropy = 0

        for val in self.data.iloc[:, index].unique():
            p_x = self.data[self.data.iloc[:, index] == val].shape[0] / self.numInstances
            if p_x != 0:
                entropy += p_x * math.log(p_x, 2)

        return -entropy

    def getRankedFeatures(self):
        self.ranked_features = sorted(list(enumerate(self.su)), key=lambda x: x[1], reverse=True)
        return self.ranked_features

    def threeIG(self, index1, index2, index3):
        return -(-self.informationGain(index1, index3) - self.informationGain(index2, index3) + self.entropy(index3) + self.jointEntropy(index1, index2) - self.treeJointEntropy(index1, index2, index3))

    def treeJointProb(self, index1, s1, index2, s2, index3, s3):
        count = 0
        for i in range(self.numInstances):
            if self.data.iat[i, index3] == s3:
                if self.data.iat[i, index2] == s2:
                    if self.data.iat[i, index1] == s1:
                        count += 1

        return count / self.numInstances

    def treeJointEntropy(self, index1, index2, index3):
        jointEntropy = 0
        for i in range(self.data.iloc[:, index3].nunique()):
            s3 = self.data.iloc[:, index3].unique()[i]
            for j in range(self.data.iloc[:, index2].nunique()):
                s2 = self.data.iloc[:, index2].unique()[j]
                for k in range(self.data.iloc[:, index1].nunique()):
                    s1 = self.data.iloc[:, index1].unique()[k]
                    temp = self.treeJointProb(index1, s1, index2, s2, index3, s3)
                    if temp != 0:
                        jointEntropy += temp * math.log(temp, 2)
        # print(-jointEntropy)
        return -jointEntropy

    def calculateEnt_subset(self, best):
        joint_entropy = 0
        num_instances = len(self.data)
        best = [x-1 for x in best]

        subset_data = self.data.iloc[:, best]

        # 计算特征子集的联合概率分布
        unique_combinations = [list(comb) for comb in
                               itertools.product(*[self.data.iloc[:, feature].unique() for feature in best])]

        for combo in unique_combinations:
            p_combo = np.sum(np.all(subset_data == combo, axis=1)) / num_instances
            if p_combo > 0:
                joint_entropy -= p_combo * math.log2(p_combo)

        return joint_entropy


    def calculaterRel(self):
        for i,score in self.ranked_features:
            for j,score in self.ranked_features:
                if self.SU(i, j) == 0: # i,j之间没有相关关系
                    self.Rel[i,j] = -1000000
                elif self.SU(j, self.numFeatures) == 0: # j, label之间没有关系
                    self.Rel[i, j] = -200000
                elif self.threeIG(i, j, self.numFeatures) < 0:
                    self.Rel[i, j] = -100
                else:
                    self.Rel[i,j] = (self.threeIG(i, j, self.numFeatures) / self.SU(i, j)) - \
                                (self.threeIG(i, j, self.numFeatures) / self.SU(j, self.numFeatures))
        # self.Rel = np.array(self.Rel)
        return self.Rel


    def calculate_best_k_stale(self, k_range, model, X, y, cv_folds=10):
        best_k_stale = None
        best_score = -float("inf")
        best_features = None

        self.getRankedFeatures()  # 确保获取特征排名
        self.calculaterRel()      # 确保计算相关性矩阵

        kf = KFold(n_splits=cv_folds)
        for k_stale in k_range:
            scores = []
            candidate_feature_list = [x[0] + 1 for x in self.ranked_features]
            selected_features = self.featureselection(candidate_feature_list, self.Rel, self.ranked_features, k_stale)
            selected_features = [x-1 for x in selected_features]
            for train_index, val_index in kf.split(X):
                train_data = X.iloc[train_index]
                val_data = X.iloc[val_index]
                train_target = y.iloc[train_index]
                val_target = y.iloc[val_index]
                model.fit(train_data.iloc[:, selected_features], train_target)
                predictions = model.predict(val_data.iloc[:, selected_features])
                score = accuracy_score(val_target, predictions)
                scores.append(score)

            average_score = sum(scores) / len(scores)
            if average_score > best_score:
                best_k_stale = k_stale
                best_score = average_score
                best_features = selected_features

        return best_k_stale, best_features

    def featureselection(self, candidate_feature_list, Rel, ranked_features, k_stale):
        ranked_features_i = np.array(ranked_features)[:,0].astype(int).tolist()
        ranked_features_su = np.array(ranked_features)[:,1].tolist()
        while len(candidate_feature_list) > 0:
            if len(candidate_feature_list) == self.numFeatures:
                best = []
                best_before = [0] * len(candidate_feature_list)
                k = 0
                best.append(candidate_feature_list[0])
                candidate_feature_list.remove(best[-1])
                best_temp = -float('inf')
                count = 0
                count_zero = 0
            path_score = [0] * len(candidate_feature_list)
            for i,feature in enumerate(candidate_feature_list):
                path_length = len(best)
                for j in range(path_length):
                    path_score[i] = path_score[i] + math.sqrt(ranked_features_su[ranked_features_i.index(best[j]-1)])*Rel[best[j]-1, feature-1]
            ranked_score = sorted(enumerate(path_score), key=lambda x: x[1], reverse=True)
            best_num = candidate_feature_list[ranked_score[0][0]]
            best_temp = ranked_score[0][1]
            if best_temp > 0:
                best.append(best_num)
                best_before[k] = copy.deepcopy(best)
                candidate_feature_list.remove(best_num)
                best_temp = 0
            if self.conditional_entropy(best, self.numFeatures) == 0:
                print(best[-1])
                ans = self.conditional_entropy(best, self.numFeatures)
            self.gain_least.append((self.entropy(self.numFeatures) - self.conditional_entropy(best, self.numFeatures)) / self.calculateEnt_subset(best))
            k = k + 1
            if len(self.gain_least) == 1:
                continue
            elif self.gain_least[-1]-self.gain_least[-2] == 0:
                count = count + 1
                count_zero = count_zero + 1
                print(self.gain_least[-1] - self.gain_least[-2])
            elif self.gain_least[-1]-self.gain_least[-2] < 0:
                count = count + 1
                print(self.gain_least[-1]-self.gain_least[-2])
            if count > k_stale:
                # best_zero_num = best_before.count(0)
                k_stale = k_stale - count_zero
                ranked_gain_stale = sorted(enumerate(self.gain_least[-(k_stale+1):]), key=lambda x: x[1], reverse=True)
                best_before = [item for item in best_before if item != 0]
                selected_features = best_before[-((k_stale+1)-ranked_gain_stale[0][0])]
                return selected_features





