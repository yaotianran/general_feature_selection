'''
General feature selection process
V 0.2b_rc
What'new:
1, support sparse matrix classification, NaN value will be handled properly.
'''

#!/usr/bin/env python3
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
#import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold, KFold, RepeatedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss, EditedNearestNeighbours, TomekLinks

pd.set_option('display.width', None)
pd.set_option('display.max_columns', 0)
pd.set_option('display.max_rows', 0)

# ===================================customed paramters==========================================
# a tab-seperated data file with headers
#DATA_FILE = '~/HDD_Storage/projects/biomarker/pulmonary_tuberculosis/data_preprocess/all.data.clean.csv'   # data file to read
DATA_FILE = '~/HDD_Storage/projects/biomarker/pulmonary_tuberculosis/data_preprocess/all.data.clean.for.python.feature.selection.csv'   # data file to read

# which columns are features ?
# could be set as 'auto' or an object that can be iterated, such as range(15) or [0, 2, 4, 5] etc. (first column in data file indexed with 0, and if the "range" function is used the end of the range should be last feature index plus 1)
# 'auto' means the all columns are feature columns except the last one
FEATURE_COLUMN = 'auto'

# which column is label ? It could be a integer or label string
# 'auto' means the last column will be marked as 'label'. Currently multi-outputs are not yet implemented
LABEL_COLUMN = 'auto'

#
USE_FEATURE_RANK = 35 # False or an integer larger than 0
FEATURE_AMOUNT = [2,3]  # how many features in the classifier ?

IS_RESAMPLE = False
OVERSAMPLER = SMOTE(kind='svm', m_neighbors=5, k_neighbors=2)
UNDERSAMPLER = TomekLinks()

# 'remove', 'median', 'mean', 'most_frequent', or other string or integer to fill the NA
DEAL_NA = 'remove'

CLASSIFIER = [
              svm.SVC( kernel= 'rbf' , probability= True, gamma= 'auto'),
              svm.SVC( kernel= 'poly' , probability= True , degree= 2, gamma= 'auto'),
              #svm.SVC( kernel= 'poly' , probability= True , degree= 3, gamma= 'auto'),
              LogisticRegression( solver= 'sag', multi_class= 'auto', max_iter= 10000),
              GaussianNB(),
              BernoulliNB(),
              KNeighborsClassifier(n_neighbors=3, weights= 'uniform', algorithm='brute'),
              KNeighborsClassifier(n_neighbors=7, weights= 'uniform', algorithm='brute'),
              KNeighborsClassifier(n_neighbors=11, weights= 'uniform', algorithm='brute')
              #RandomForestClassifier(n_estimators= 100)
              ]


VALIDATION_FOLD = 5
REPEAT_FOLD = 20
TIME_OUT = 60  # in seconds


# ==============================================================================================
# =================================DO NOT change the code below=================================
# ===========================unless you really know what you are doing==========================
# ==============================================================================================

#Return a dict, every key is a classifier and value is rank list, each of which is score (less the better)
#such as classifier1: [[score1, featurn_index1], [score2, featurn_index2]], ......}
def rank_features():
    '''
    Return a dict, every key is a classifier and value is rank list, each of which is score (less the better)
    such as classifier1: [[score1, featurn_index1], [score2, featurn_index2]], ......}
    '''
    data_file_str = path.realpath(path.expanduser(DATA_FILE))
    data_pd = pd.read_csv(data_file_str, sep='\t')
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
            print('Ranking features {}/{}...'.format(s, max(features_range)), end= '\r' )

            if DEAL_NA == 'remove':
                keep_se = data_pd.iloc[:, s].notna()
                X = data_pd.loc[keep_se, data_pd.columns[[s]]].values
                if len(X) <= 1:
                    message = 'Feature {} has no samples/observations or are all NaN. This feature will be skipped.'.format(s)
                    print(message)
                    continue
                Y = data_pd.loc[keep_se, data_pd.columns[label_int]].values.reshape(sum(keep_se))
            elif DEAL_NA in ['median', 'mean', 'most_frequent']:
                X = data_pd.iloc[:, [s]].values
                IMP = SimpleImputer(missing_values=np.nan, strategy=DEAL_NA)
                IMP.fit(X)
                X = IMP.transform(X)
                Y = data_pd.iloc[:, [label_int]].values.reshape(len(data_pd))
            else:
                X = data_pd.iloc[:, [s]].values
                IMP = SimpleImputer(missing_values=np.nan, strategy = 'constant', fill_value = DEAL_NA)
                IMP.fit(X)
                X = IMP.transform(X)
                Y = data_pd.iloc[:, [label_int]].values.reshape(len(data_pd))

            # test if Y only has one class
            if len(set(Y)) == 1:
                message = 'Only one class "{label}" in non-NaN feature-label pair (feature {s}). This feature will be skipped.'.format(label= set(Y), s= s)
                print(message)
                continue

            kf = StratifiedShuffleSplit(n_splits= REPEAT_FOLD, test_size= 1/VALIDATION_FOLD)
            test_error_float = 0.0
            test_error_lst = []
            AUCs_lst = []
            for trainIndex_ar, testIndex_ar in kf.split(X, Y):

                X_train_ar = X[trainIndex_ar]
                Y_train_ar = Y[trainIndex_ar]
                X_test_ar = X[testIndex_ar]
                Y_test_ar = Y[testIndex_ar]
                if IS_RESAMPLE:
                    try:
                        X_train_ar, Y_train_ar = OVERSAMPLER.fit_sample(X[trainIndex_ar], Y[trainIndex_ar])
                        X_train_ar, Y_train_ar = UNDERSAMPLER.fit_sample(X_train_ar, Y_train_ar)
                    except:
                        continue
                    X_test_ar = X[testIndex_ar]
                    Y_test_ar = Y[testIndex_ar]

                # fit the classifier
                try:
                    clf.fit(X_train_ar, Y_train_ar)
                    Y_predict_ar = clf.predict(X_test_ar)
                    probas_ar = clf.predict_proba(X_test_ar)
                except ValueError as ex:
                    message = 'The following error occured in one of the folds of cross-validataion. This fold will be skipped.\n{ex}\nTrainindex:{trainindex}\nTestindex:{testindex}\nFeature:{s}'.format(ex= ex,
                                                                                                                                                                                                            trainindex= list(trainIndex_ar),
                                                                                                                                                                                                            testindex= list(testIndex_ar),
                                                                                                                                                                                                            s= s)
                    print(message)
                    continue
                error_ar = Y_predict_ar != Y_test_ar  # this is for catalogic prediction
                test_error_float = sum(error_ar) / len(Y_predict_ar)
                test_error_lst.append(test_error_float)  # record test data set prediction error

                # calculate AUC
                pos_label_lst = list(set(Y_test_ar))
                pos_label_lst.sort()
                temp = []
                try:
                    if len(pos_label_lst) == 1:  # if test set only has one class then there's no way to calculate false positive rate
                        AUC_float = np.NaN
                    elif len(pos_label_lst) == 2: # binary class
                        TPR_ar, FPR_ar, thresholds_ar = roc_curve(Y_test_ar, probas_ar[:, 1], pos_label=pos_label_lst[0])
                        AUC_float = auc(FPR_ar, TPR_ar)
                    else:
                        for label in pos_label_lst: # multiple classes
                            TPR_ar, FPR_ar, thresholds_ar = roc_curve(Y_test_ar, probas_ar[:, 1], pos_label= label)
                            temp_auc = auc(FPR_ar, TPR_ar)
                            if temp_auc < 0.5:
                                temp.append(1-temp_auc)
                            else:
                                temp.append(temp_auc)
                        AUC_float = np.nanmean(temp)
                except:
                    AUC_float = np.NaN
                AUCs_lst.append(AUC_float)   # record AUC

            rank_dict[clf].append( [np.nanmean(AUCs_lst), s] )
            #rank_dict[clf].append( [1 - np.nanmean(test_error_lst), s] )

        print()

    return rank_dict

def main(argvList=sys.argv, argv_int=len(sys.argv)):

    start_time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')  # record the start time
    data_file_str = path.realpath(path.expanduser(DATA_FILE))
    data_pd = pd.read_csv(data_file_str, sep='\t')

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

    if USE_FEATURE_RANK:
        rank_dict = rank_features()
    else:
        rank_dict = None

    combinations_pd = pd.DataFrame(columns=['Classifier', 'Features Combination', 'Test Sets Error', 'Overall Error', 'AUC', 'Overfit Score', 'Counter'])    # main data frame to store all calculation results
    for classifier_index in range(len(CLASSIFIER)):
        clf = CLASSIFIER[classifier_index]
        print('\n=============================================={}================================================'.format(time.ctime()))
        print(clf, end= '\n\n')
        if IS_RESAMPLE:
            print(OVERSAMPLER, end= '\n\n')
            print(UNDERSAMPLER, end= '\n\n')

        if rank_dict is not None: # use features ranking for fast filtering
            rank_lst = rank_dict[clf]   # [[score1, feature_index1], [score2, feature_index2], .....]
            rank_lst.sort(reverse = True)  # sort by score (decreasingly)
            print('All features rank:')
            pp.pprint(rank_lst)
            rank_qualified_lst = []  # [feature_index1, feature_index2, feature_index3 .....]

            for i in range(USE_FEATURE_RANK):
                try:
                    rank_qualified_lst.append(rank_lst[i][1])
                except IndexError:
                    break
            print('The following features\' combinations will be tested:')
            print(rank_qualified_lst)

            combinations_lst = []
            for i in FEATURE_AMOUNT:
                for tu in itertools.combinations(rank_qualified_lst, i):
                    combinations_lst.append(list(tu))
        else: # if rank_dict is None, we use all features
            message = 'Features ranking is disabled. We will use all features'
            print(message)
            combinations_lst = []
            for i in FEATURE_AMOUNT:
                for tu in itertools.combinations(features_range, i):
                    combinations_lst.append(list(tu))

        print('\n\nNo.\tClassifier\tFeatures Combination\tTest Sets Error\tOverall Error\tAUC\tOverfit Score')

        j = 1 # feature combination No.
        for s in combinations_lst:  # s is a list, such as [4, 6, 9, 11, 14, 19]

            if DEAL_NA == 'remove':
                keep_se = data_pd.iloc[:, s].notna().all(axis = 1)
                X = data_pd.loc[keep_se, data_pd.columns[s]].values
                if len(X) == 0:
                    message = 'Feature combination {} has no samples/observations or are all NaN. This combination will be skipped.'.format(s)
                    print(message)
                    continue
                Y = data_pd.loc[keep_se, data_pd.columns[label_int]].values.reshape(sum(keep_se))
            elif DEAL_NA in ['median', 'mean', 'most_frequent']:
                X = data_pd.iloc[:, s].values
                IMP = SimpleImputer(missing_values=np.nan, strategy=DEAL_NA)
                IMP.fit(X)
                X = IMP.transform(X)
                Y = data_pd.iloc[:, [label_int]].values.reshape(len(data_pd))
            else:
                X = data_pd.iloc[:, s].values
                IMP = SimpleImputer(missing_values=np.nan, strategy = 'constant', fill_value = DEAL_NA)
                IMP.fit(X)
                X = IMP.transform(X)
                Y = data_pd.iloc[:, [label_int]].values.reshape(len(data_pd))

            # test if Y only has one class
            if len(set(Y)) == 1:
                message = 'Only one class "{label}" in non-NaN feature-label pair (feature {s}). This feature will be skipped.'.format(label= set(Y), s= s)
                print(message)
                continue

            kf = StratifiedShuffleSplit(n_splits= REPEAT_FOLD, test_size= 1/VALIDATION_FOLD)  # maximum the splits, we will get enough test data
            overall_error_lst = []  # to store total false positive error plue false negative error rate
            overall_error_float = 0.0
            test_error_lst = []  # to store prediction error rate
            test_error_float = 0.0
            AUCs_lst = []
            AUC_float = 0.0
            overfit_score_lst = []
            overfit_score_float = 0.0

            i = 0
            time_int = round(time.time())
            for trainIndex_ar, testIndex_ar in kf.split(X, Y):

                X_train_ar = X[trainIndex_ar]
                Y_train_ar = Y[trainIndex_ar]
                X_test_ar = X[testIndex_ar]
                Y_test_ar = Y[testIndex_ar]

                if IS_RESAMPLE:
                    try:
                        X_train_ar, Y_train_ar = OVERSAMPLER.fit_sample(X[trainIndex_ar], Y[trainIndex_ar])
                        X_train_ar, Y_train_ar = UNDERSAMPLER.fit_sample(X_train_ar, Y_train_ar)
                    except:
                        continue
                    X_test_ar = X[testIndex_ar]
                    Y_test_ar = Y[testIndex_ar]

                # fit the classifier
                try:
                    clf.fit(X_train_ar, Y_train_ar)
                    Y_predict_ar = clf.predict(X_test_ar)
                    Y_overall_predict_ar = clf.predict(X)
                    probas_ar = clf.predict_proba(X_test_ar)
                except ValueError as ex:
                    message = 'The following error occured in one of the folds of cross-validataion. This fold will be skipped.\n{ex}\nTrainindex:{trainindex}\nTestindex:{testindex}\nFeature:{s}'.format(ex= ex,
                                                                                                                                                                                                            trainindex= list(trainIndex_ar),
                                                                                                                                                                                                            testindex= list(testIndex_ar),
                                                                                                                                                                                                            s= s)
                    print(message)
                    continue

                # calculate test data prediction error
                error_ar = Y_predict_ar != Y_test_ar  # this is for catalogic prediction
                test_error_float = sum(error_ar) / len(Y_predict_ar)
                test_error_lst.append(test_error_float)  # record test data set prediction error

                # calculate overall data prediction error
                error_ar = Y_overall_predict_ar != Y
                overall_error_float = sum(error_ar) / len(Y)
                overall_error_lst.append(overall_error_float)

                # calculate AUC
                pos_label_lst = list(set(Y_test_ar))
                pos_label_lst.sort()
                temp = []
                try:
                    if len(pos_label_lst) == 1:  # if test set only has one class then there's no way to calculate false positive rate
                        AUC_float = np.NaN
                    elif len(pos_label_lst) == 2: # binary class
                        TPR_ar, FPR_ar, thresholds_ar = roc_curve(Y_test_ar, probas_ar[:, 1], pos_label=pos_label_lst[0])
                        AUC_float = auc(FPR_ar, TPR_ar)
                    else:
                        for label in pos_label_lst: # multiple classes
                            TPR_ar, FPR_ar, thresholds_ar = roc_curve(Y_test_ar, probas_ar[:, 1], pos_label= label)
                            temp_auc = auc(FPR_ar, TPR_ar)
                            if temp_auc < 0.5:
                                temp.append(1-temp_auc)
                            else:
                                temp.append(temp_auc)
                        AUC_float = np.nanmean(temp)
                except:
                    AUC_float = np.NaN
                AUCs_lst.append(AUC_float)   # record AUC

                # calculate overfit score
                if overall_error_float == 0.0 or test_error_float == 0.0:
                    overfit_score_float = np.NaN
                else:
                    error_ratio_float = test_error_float / overall_error_float
                    #overfit_score_float = (error_ratio_float - 1) / (VALIDATION_FOLD - 1)
                    overfit_score_float = math.log(error_ratio_float, VALIDATION_FOLD)
                    #print(error_ratio_float, VALIDATION_FOLD, overfit_score_float)
                    if overfit_score_float > 1:
                        overfit_score_float = 1
                    elif overfit_score_float < 0:
                        overfit_score_float = 0
                    else:
                        pass
                overfit_score_lst.append(overfit_score_float)
                #overfit_score_lst.append(0) # we surpress overfit_score temporaryly

                i += 1
                if i == REPEAT_FOLD:
                    break
                elif round(time.time()) - time_int > TIME_OUT:
                    print('\n{} combination reaches maximum time limit ({}s), use {} iterations'.format(s, TIME_OUT, i))
                    break
                else:
                    continue


            #temp = np.mean(test_error_lst)
            #temp = np.mean(overall_error_lst)
            #temp = np.nanmean(AUCs_lst)
            #temp = np.nanmean(overfit_score_lst)
            print('{}\t{}\t{: >24}\t{:f}\t{:f}\t{:f}\t{:f}\t{}            '.format(j, classifier_index, str(s), np.mean(test_error_lst), np.mean(overall_error_lst), np.nanmean(AUCs_lst), np.nanmean(overfit_score_lst), collections.Counter(Y)), end='\r')
            j += 1

            temp = pd.DataFrame([[classifier_index, s, np.mean(test_error_lst), np.mean(overall_error_lst), np.nanmean(AUCs_lst), np.nanmean(overfit_score_lst), collections.Counter(Y)]], columns=['Classifier', 'Features Combination', 'Test Sets Error', 'Overall Error', 'AUC', 'Overfit Score', 'Counter'])
            combinations_pd = pd.concat([combinations_pd, temp], sort= False)

    combinations_pd = combinations_pd.sort_values(by='AUC', ascending= False)
    combinations_pd.index = range(0, len(combinations_pd))
    print(end='\r')
    print()
    print('==============================================All combinations================================================')
    print(combinations_pd)
    print()

    # write the output

    if not os.access(start_time_str, os.R_OK and os.W_OK):
        os.mkdir(start_time_str)
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name_str = '{0}/{0}_{1}.csv'.format(start_time_str, time_str)
    combinations_pd.to_csv(file_name_str, sep='\t', index=False)
    print('Top {} feature combinations writen to ./{}'.format(len(combinations_pd), file_name_str))
    print()


    print('=============================================={}================================================'.format(time.ctime()))
    try:
        shutil.copyfile(data_file_str, path.join(start_time_str, path.split(data_file_str)[1]))
    except Exception as ex:
        print(ex)

    with open(start_time_str + '/README', 'wt') as output_r:
        output_r.writelines('CLASSIFIER= ' + str(CLASSIFIER) + '\n')
        output_r.writelines('DATA_FILE= ' + DATA_FILE + '\n')
        output_r.writelines('FEATURE_COLUMN= ' + str(FEATURE_COLUMN) + '\n')
        output_r.writelines('LABEL_COLUMN= ' + str(LABEL_COLUMN) + '\n')
        output_r.writelines('FEATURE_AMOUNT= ' + str(FEATURE_AMOUNT) + '\n')
        #output_r.writelines('CLASS_MINIMUM_SAMPLE= ' + str(CLASS_MINIMUM_SAMPLE) + '\n')
        output_r.writelines('IS_RESAMPLE= ' + str(IS_RESAMPLE) + '\n')
        if IS_RESAMPLE:
            output_r.writelines('OVERSAMPLER= ' + str(OVERSAMPLER) + '\n')
            output_r.writelines('UNDERSAMPLER= ' + str(UNDERSAMPLER) + '\n')
        output_r.writelines('VALIDATION_FOLD= ' + str(VALIDATION_FOLD) + '\n')
        output_r.writelines('REPEAT_FOLD= ' + str(REPEAT_FOLD) + '\n')
        output_r.writelines('START_TIME= ' + start_time_str + '\n')
        output_r.writelines('END_TIME= ' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '\n')

    return



main()
#r = rank_features()
print('done')
pd.reset_option('display.width')
pd.reset_option('display.max_columns')
pd.reset_option('display.max_rows')
