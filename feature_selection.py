#!/usr/bin/env python3
'''
General feature selection process
V 0.3b
What'new:
1, stepwise combination calculation
2, multiprocess
3, use F1 score to rank the feature and h(theta) cutoff

Usage: ./feature_selection.py [output.txt]

TODO:
1, able to display how many combination has be calculated
'''
import os
import sys
import time
import math
import common
import collections
import datetime
import shutil
import itertools
import os.path as path
import numpy as np
import scipy
import pandas as pd
import pprint as pp
import multiprocessing
from multiprocessing import Pool, Value

from sklearn.impute import SimpleImputer
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold, KFold, RepeatedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss, EditedNearestNeighbours, TomekLinks

pd.set_option('display.width', None)
pd.set_option('display.max_columns', 0)
pd.set_option('display.max_rows', 0)

# ===================================customed paramters==========================================
# a tab-seperated data file with headers
#DATA_FILE = '~/sda1/WPS/tmb_analysis/support/significant.genes.for.python.cutoff.10.csv'   # data file to read
DATA_FILE = '~/sda1/significant.genes.for.python.cutoff.10.csv'

# which columns are features ?
# could be set as 'auto' or an object that can be iterated, such as range(15) or [0, 2, 4, 5] etc. (first column in data file indexed with 0, and if the "range" function is used the end of the range should be last feature index plus 1)
# 'auto' means that except the last one column all others are feature columns
FEATURE_COLUMN = 'auto'

# which column is label ? It could be a integer or label string
# 'auto' means the last column will be marked as 'label'. Currently multi-outputs are not yet implemented
LABEL_COLUMN = 'auto'


SEARCH_METHOD = 'stepwise'  # 'stepwise' or 'brutal'.
FEATURE_AMOUNT_TO_PICKUP = [7]  # how many features in the classifier ?

# If SEARCH_METHOD is 'brutal', an integer N larger than 0 indicates to use fast top-N feature rank method to narrow down feature candidates, 0 or False disable this function.
# If SEARCH_METHOD is 'stepwise', then this paramater will be ignored.
USE_FEATURE_RANK = 25

#  IS_RESAMPLE = False  # current RESAMPLE method is removed
# OVERSAMPLER = SMOTE(kind='svm', m_neighbors=5, k_neighbors=2)
# UNDERSAMPLER = TomekLinks()

# 'remove', 'median', 'mean', 'most_frequent', or other string or integer to fill the NA
DEAL_NA = 'remove'

CLASSIFIER = [
    svm.SVC(kernel='rbf', probability=True, gamma='scale'),
    # svm.SVC(kernel='linear', probability=True),
    # svm.SVC(kernel='poly', probability=True, degree=3, gamma='scale'),
    LogisticRegression(solver='liblinear', multi_class='auto', max_iter=10000, tol=0.0001)   # For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones. For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; ‘liblinear’ is limited to one-versus-rest schemes.
    # GaussianNB(),
    # BernoulliNB(),
    # KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='brute'),
    # KNeighborsClassifier(n_neighbors=7, weights='uniform', algorithm='brute'),
    # KNeighborsClassifier(n_neighbors=11, weights='uniform', algorithm='brute')
    #RandomForestClassifier(n_estimators= 100)
]

REPEAT_FOLD = 5
VALIDATION_FOLD = 4
THREAD = 2  # set to None to use all cpus


# ==============================================================================================
# =================================DO NOT change the code below=================================
# ===========================unless you really know what you are doing==========================
# ==============================================================================================

# this is the subclassifier - a wrapper used by other function
#  return str(clf.__class__), feature_column, test_error_rate, train_error_rate, AUC, F1_score, counter_str, customed_dict
#  usage: clf_str, column_index_lst, test_error_rate, train_error_rate, AUC, F1_score, counter_str, customed_dict = subclassifier(clf, data_pd, [feature_column], label_int)
def subclassifier(clf, data: pd.DataFrame, feature_column: list, label_column: int, *customed_tuple):
    '''
    Use clf classifier to classify data, 'NA' in data will be properly dealed with.
    Return multiple values.

    Parameters:
        **clf**: classifier
            scikit_learn classifier

        **data**: pandas.DataFrame
            A pandas.DataFrame containing multiple features and one label (multiple label not implemented)

        **feature_column**: integer list
             Which column are features? (like [20, 8, 25, 0, 16, 28, 26, 42, 39], 0-based index)

        **label_column**: integer
             Which column are the label? (0-based index)

        **customed_tuple**: every thing
            All arguments here will return in a tuple as the same as they are inputed

    Returns:
        **str(clf.__class__)**: string
            classifier's classname

        **feature_column**: integer list
            the feature culumn list, the same as argument

        **test_error_float**: float
            test sets error rate

        **overall_error_float**: float
            overall test sets error rate

        **AUC_float**: float
            AUC of ROC

        **overfit_score_float**: float
            overfir score (between 0 and 1, 1 is the worst)

        **counter_str**: string
            a string display label counter (like 'False 71, True 38')

        **customed_tuple**: tuple
            The custom tuple will be returned as the same as they are inputed
    '''

    if isinstance(feature_column, list):
        for i in feature_column:
            if not isinstance(i, int):
                message = 'Parameter feature_column is expected as an integer list, current is {0} ({1})'.format(type(i), i)
                raise TypeError(message)
    else:
        message = 'Parameter feature_column is expected as a list, current is {0} ({1})'.format(type(feature_column), feature_column)
        raise TypeError(message)

    if not isinstance(label_column, int):
        message = 'Parameter label_column is expected as an integer, current is {0} ({1})'.format(type(label_column), label_column)
        raise TypeError(message)

    # =======================================
    if DEAL_NA == 'remove':
        keep_se = data.iloc[:, feature_column].notna().all(axis=1)  # only keep those that don't contain any NA in rows
        X = data.loc[keep_se, data.columns[feature_column]].values
        if len(X) == 0:
            message = 'Feature combination {} has no samples/observations or are all NaN.'.format(feature_column)
            print(message)
            return None
        Y = data.loc[keep_se, data.columns[[label_column]]].values.reshape(sum(keep_se))
    elif DEAL_NA in ['median', 'mean', 'most_frequent']:
        X = data.iloc[:, feature_column].values
        IMP = SimpleImputer(missing_values=np.nan, strategy=DEAL_NA)
        IMP.fit(X)
        X = IMP.transform(X)
        Y = data.iloc[:, label_column].values.reshape(len(data))
    else:
        X = data.iloc[:, feature_column].values
        IMP = SimpleImputer(missing_values=np.nan, strategy = 'constant', fill_value = DEAL_NA)
        IMP.fit(X)
        X = IMP.transform(X)
        Y = data.iloc[:, label_column].values.reshape(len(data))

    # test if Y only has one class
    if len(set(Y)) == 1:
        message = 'Only one class {label} in non-NaN feature-label pair (feature {feature_column}). This feature will be skipped.'.format(label=set(Y),
                                                                                                                                          feature_column=feature_column)
        print(message)
        return None

    #  intialize the results variables
    train_error_lst = []  # to store total false positive error plue false negative error rate
    train_error_float = 0.0
    test_error_lst = []  # to store prediction error rate
    test_error_float = 0.0
    AUCs_lst = []
    AUC_float = 0.0
    overfit_score_lst = []
    overfit_score_float = 0.0
    F1_score_lst = []
    F1_score_float = 0.0

    kf = StratifiedShuffleSplit(n_splits=REPEAT_FOLD, test_size=1 / VALIDATION_FOLD)  # maximum the splits, we will get enough test data
    for trainIndex_ar, testIndex_ar in kf.split(X, Y):
        X_train_ar = X[trainIndex_ar]
        Y_train_ar = Y[trainIndex_ar]
        X_test_ar = X[testIndex_ar]
        Y_test_ar = Y[testIndex_ar]

        '''
        if IS_RESAMPLE:
            try:
                X_train_ar, Y_train_ar = OVERSAMPLER.fit_sample(X_train_ar, Y_train_ar)
                X_train_ar, Y_train_ar = UNDERSAMPLER.fit_sample(X_train_ar, Y_train_ar)
            except:
                continue
        '''
        # fit the classifier
        try:
            clf.fit(X_train_ar, Y_train_ar)
            Y_predict_ar = clf.predict(X_test_ar)
            Y_train_predict_ar = clf.predict(X_train_ar)
            Y_overall_predict_ar = clf.predict(X)
            probas_ar = clf.predict_proba(X_test_ar)
        except ValueError as ex:
            message = 'The following error occured in one of the folds of cross-validataion. This fold will be skipped.\n{ex}\nTrainindex:{trainindex}\nTestindex:{testindex}\n'.format(ex=ex,
                                                                                                                                                                                      trainindex=list(trainIndex_ar),
                                                                                                                                                                                      testindex=list(testIndex_ar))
            print(message)
            continue

        # calculate test data prediction error rate
        test_error_float = sum(Y_predict_ar != Y_test_ar) / len(Y_test_ar)
        test_error_lst.append(test_error_float)  # record test data set prediction error for this iteration

        # calculate train data prediction error
        train_error_float = sum(Y_train_predict_ar != Y_train_ar) / len(Y_train_ar)
        train_error_lst.append(train_error_float)

        # calculate AUC and F1 score
        pos_label_lst = list(set(Y_test_ar))
        pos_label_lst.sort()
        try:
            if len(pos_label_lst) == 1:  # if test set only has one class then there's no way to calculate false positive rate
                AUC_float = np.NaN
                F1_score_float = np.NaN
            elif len(pos_label_lst) == 2: # binary class
                TPR_ar, FPR_ar, thresholds_ar = roc_curve(Y_test_ar, probas_ar[:, 1], pos_label=pos_label_lst[0])
                AUC_float = auc(FPR_ar, TPR_ar)
                F1_score_float = f1_score(Y_test_ar, Y_predict_ar, average='binary', pos_label=pos_label_lst[0])
            else:
                for label in pos_label_lst: # multiple classes
                    TPR_ar, FPR_ar, thresholds_ar = roc_curve(Y_test_ar, probas_ar[:, 1], pos_label= label)
                    temp_auc = auc(FPR_ar, TPR_ar)
                    temp = []
                    if temp_auc < 0.5:
                        temp.append(1-temp_auc)
                    else:
                        temp.append(temp_auc)
                AUC_float = np.nanmean(temp)
                F1_score_float = f1_score(Y_test_ar, Y_predict_ar, average='marco')
        except:
            AUC_float = np.NaN
            F1_score_float = np.NaN
        AUCs_lst.append(AUC_float)   # record AUC
        F1_score_lst.append(F1_score_float)

        # calculate overfit score
        '''
        overall_error_float = sum(Y_overall_predict_ar != Y) / len(Y)
        if overall_error_float == 0.0 or test_error_float == 0.0:
            overfit_score_float = np.NaN
        else:
            error_ratio_float = test_error_float / overall_error_float
            overfit_score_float = (test_error_float / train_error_float) * test_error_float
            #overfit_score_float = (error_ratio_float - 1) / (VALIDATION_FOLD - 1)
            overfit_score_float = math.log(error_ratio_float, VALIDATION_FOLD)
            if overfit_score_float > 1:
                overfit_score_float = 1
            elif overfit_score_float < 0:
                overfit_score_float = 0
            else:
                pass
        overfit_score_lst.append(overfit_score_float)
        '''

        # covert collections.Counter() to string
        try:
            counter_lst = []
            for key, value in collections.Counter(Y).items():
                counter_lst.append(str(key) + ' ' + str(value))
            counter_str = ', '.join(counter_lst)
        except:
            counter_str = str(collections.Counter(Y))

    return str(clf.__class__), feature_column, np.mean(test_error_lst), np.mean(train_error_lst), np.nanmean(AUCs_lst), np.nanmean(F1_score_lst), counter_str, customed_tuple


def rank_features():
    '''
    Return a dict, every key is a classifier and value is rank list, each of which is score (more the better)
    such as classifier1: [[score1, featurn_index1], [score2, featurn_index2]], ......}
    '''

    data_file_str = path.realpath(path.expanduser(DATA_FILE))
    data_pd = pd.read_csv(data_file_str, sep='\t')
    # assure features_range and
    if FEATURE_COLUMN == 'auto':
        features_range = range(data_pd.shape[1]-1)
    else:
        features_range = FEATURE_COLUMN

    if LABEL_COLUMN == 'auto':
        label_int = data_pd.shape[1]-1
    elif isinstance(LABEL_COLUMN, int):
        label_int = LABEL_COLUMN
    elif isinstance(LABEL_COLUMN, str):
        try:
            label_int = data_pd.columns.get_loc(LABEL_COLUMN)
        except KeyError:
            message = 'KeyError: "{}" is not a valide lable name.'.format(LABEL_COLUMN)
            common.cprint(message)

    rank_dict = collections.defaultdict(list)   # {classifier1: [[score1, feature_index1], [score2, feature_index2]], ......}
    for clf in CLASSIFIER:
        print()
        print('fast ranking:\n', clf)
        for s in features_range:   # s is an integer
            print('Ranking features {}/{}...'.format(int(s) + 1, int(max(features_range)) + 1), end='\r')

            clf_str, column_index_lst, test_error_rate, train_error_rate, AUC, F1_score, counter_str, customed_dict = subclassifier(clf, data_pd, [s], label_int)

            score_float = AUC  # the larger the better

            rank_dict[clf].append([score_float, s])

        print()

    return rank_dict

def write_parameter(folder):
    '''
    write the parameter into a file so that we can check them later.
    '''
    data_file_str = path.realpath(path.expanduser(DATA_FILE))
    if not os.access(folder, os.R_OK and os.W_OK):
        os.mkdir(folder)

    try:
        shutil.copyfile(data_file_str, path.join(folder, path.split(data_file_str)[1]))
    except Exception as ex:
        print(ex)

    with open(folder + '/README', 'wt') as output_r:
        output_r.writelines('CLASSIFIER= ' + str(CLASSIFIER) + '\n')
        output_r.writelines('DATA_FILE= ' + DATA_FILE + '\n')
        output_r.writelines('FEATURE_COLUMN= ' + str(FEATURE_COLUMN) + '\n')
        output_r.writelines('LABEL_COLUMN= ' + str(LABEL_COLUMN) + '\n')
        output_r.writelines('SEARCH_METHOD= ' + SEARCH_METHOD + '\n')
        output_r.writelines('FEATURE_AMOUNT_TO_PICKUP= ' + str(FEATURE_AMOUNT_TO_PICKUP) + '\n')
        if SEARCH_METHOD == 'brutal':
            output_r.writelines('USE_FEATURE_RANK= ' + str(USE_FEATURE_RANK) + '\n')
        # output_r.writelines('IS_RESAMPLE= ' + str(IS_RESAMPLE) + '\n')
        if False:
            output_r.writelines('OVERSAMPLER= ' + str(OVERSAMPLER) + '\n')
            output_r.writelines('UNDERSAMPLER= ' + str(UNDERSAMPLER) + '\n')
        output_r.writelines('VALIDATION_FOLD= ' + str(VALIDATION_FOLD) + '\n')
        output_r.writelines('REPEAT_FOLD= ' + str(REPEAT_FOLD) + '\n')
        output_r.writelines('START_TIME= ' + folder + '\n')
        output_r.writelines('END_TIME= ' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '\n')
    return True

def main(argvList=sys.argv, argv_int=len(sys.argv)):

    start_time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')  # record the start time
    data_pd = pd.read_csv(path.realpath(path.expanduser(DATA_FILE)), sep='\t')

    if FEATURE_COLUMN == 'auto':
        features_range = range(data_pd.shape[1]-1)
    else:
        features_range = FEATURE_COLUMN

    if LABEL_COLUMN == 'auto':
        label_int = data_pd.shape[1]-1
    elif isinstance(LABEL_COLUMN, int):
        label_int = LABEL_COLUMN
    elif isinstance(LABEL_COLUMN, str):
        try:
            label_int = data_pd.columns.get_loc(LABEL_COLUMN)
        except KeyError:
            message = 'KeyError: "{}" is not a valide lable name.'.format(LABEL_COLUMN)
            common.cprint(message)
            raise KeyError(message)

    # =====================================================

    if SEARCH_METHOD != 'stepwise' and SEARCH_METHOD != 'brutal':
        message = 'SEARCH_METHOD must be either stepwise or brutal.'
        raise ValueError(message)

    combinations_pd = pd.DataFrame(columns=['Classifier', 'Features Combination', 'Test Sets Error', 'Train Sets Error', 'AUC', 'F1 Score', 'Counter'])    # main data frame to store all calculation results
    if SEARCH_METHOD == 'stepwise':
        classifier_index = 0
        for clf in CLASSIFIER:
            print('\n=============================================={}================================================'.format(time.ctime()))
            print(clf, end='\n\n')
            features_available_lst = list(features_range)
            best_feature_lst = []
            max_score_float = -1
            #  TODO: raise error when max(FEATURE_AMOUNT_TO_PICKUP) larger than overall features amount
            for i in range(max(FEATURE_AMOUNT_TO_PICKUP)):
                best_feature_int = -1
                for j in features_available_lst:
                    temp_lst = best_feature_lst.copy()
                    temp_lst.append(j)
                    clf_str, column_index_lst, test_error_rate, train_error_rate, AUC, F1_score, counter_str, customed_dict = subclassifier(clf, data_pd, temp_lst, label_int)
                    message = '{}\t{: >24}\t{:f}\t{:f}\t{:f}\t{:f}\t{}            '.format(classifier_index, str(temp_lst), test_error_rate, train_error_rate, AUC, F1_score, counter_str)
                    print(message, end='\r', flush=True)
                    temp_pd = pd.DataFrame([[classifier_index, temp_lst, test_error_rate, train_error_rate, AUC, F1_score, counter_str]], columns=['Classifier', 'Features Combination', 'Test Sets Error', 'Train Sets Error', 'AUC', 'F1 Score', 'Counter'])
                    combinations_pd = pd.concat([combinations_pd, temp_pd], sort=False)
                    if AUC + F1_score > max_score_float:
                        max_score_float = AUC + F1_score
                        best_feature_int = j

                if best_feature_int == -1:  # we can't improve the results by adding another feature, so we break
                    break

                features_available_lst.pop(features_available_lst.index(best_feature_int))
                best_feature_lst.append(best_feature_int)

            classifier_index += 1

    else:  # 'brutal algorithm'
        print('\n=============================================={}================================================'.format(time.ctime()))
        rank_qualified_dict = collections.defaultdict(list)
        if USE_FEATURE_RANK == 0 or USE_FEATURE_RANK == False:
            for clf in CLASSIFIER:
                rank_qualified_dict[clf] = list(features_range)
        else:
            rank_dict = rank_features()
            for clf in CLASSIFIER:
                temp_lst = rank_dict[clf].copy()
                temp_lst.sort(reverse = True)
                for i in range(USE_FEATURE_RANK):
                    try:
                        rank_qualified_dict[clf].append(temp_lst[i][1])
                    except IndexError:
                        break
                print()
                print('For {}, the qualified features are:'.format(clf.__class__))
                print(rank_qualified_dict[clf])

        print()
        combinations_lst = []  # [[clf, data, combination_tu, label_int, clf], [clf, data, combination_tu, label_int, clf], .....]
        q = multiprocessing.Queue()
        v = multiprocessing.Value('i')
        for key, value in rank_qualified_dict.items():  # key is clf, value is a list with length of USE_FEATURE_RANK
            for i in FEATURE_AMOUNT_TO_PICKUP:
                for tu in itertools.combinations(value, i):
                    combinations_lst.append((key, data_pd, list(tu), label_int, CLASSIFIER.index(key)))

        temp = time.time()
        with Pool(processes=THREAD) as pool:
            # result_lst = pool.starmap(subclassifier, combinations_lst)
            result = pool.starmap_async(subclassifier, combinations_lst)
            while not result.ready():
                print(round(time.time() - temp), end='\r', flush= True)
                time.sleep(1)

        # wtire the results to combinations_pd
        combinations_pd = pd.DataFrame(columns=['Classifier', 'Features Combination', 'Test Sets Error', 'Train Sets Error', 'AUC', 'F1 Score', 'Counter'])    # main data frame to store all calculation results
        for tu in result.get():
            # each item is a tuple, for example:
            # ("<class 'sklearn.svm.classes.SVC'>",
            # [20, 43],
            # 0.2545454545454545,
            # 0.21009174311926607,
            # 0.7370535714285713,
            # 0.12862222347471336,
            # 'False 71, True 38',
            # (0,))
            temp_pd = pd.DataFrame([[tu[7][0], tu[1], tu[2], tu[3], tu[4], tu[5], tu[6]]], columns=['Classifier', 'Features Combination', 'Test Sets Error', 'Train Sets Error', 'AUC', 'F1 Score', 'Counter'])
            combinations_pd = pd.concat([combinations_pd, temp_pd], sort=False)

    combinations_pd = combinations_pd.sort_values(by='AUC', ascending=False)
    combinations_pd.index = range(0, len(combinations_pd))
    if not os.access(start_time_str, os.R_OK and os.W_OK):
        os.mkdir(start_time_str)
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name_str = '{0}/{0}_{1}.csv'.format(start_time_str, time_str)
    combinations_pd.to_csv(file_name_str, sep='\t', index=False)
    write_parameter(start_time_str)
    print()
    print('done')
    print('\n=============================================={}================================================'.format(time.ctime()))

    return

main()
# print('done')
pd.reset_option('display.width')
pd.reset_option('display.max_columns')
pd.reset_option('display.max_rows')
