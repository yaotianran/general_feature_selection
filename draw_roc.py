#!/usr/bin/env python3
'''
Draw ROC 0.3a_rc build190218
'''

import os, sys, time, math, common, collections, datetime, re, shutil
import os.path as path
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold, KFold, RepeatedKFold, RepeatedStratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier, RidgeClassifier, PassiveAggressiveClassifier

from sklearn.metrics import roc_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss, EditedNearestNeighbours, TomekLinks

from itertools import combinations

pd.set_option('display.width', None)
pd.set_option('display.max_columns', 0)
pd.set_option('display.max_rows', 0)

#===================================customed paramters==========================================

# a tab-seperated data file with headers
#DATA_FILE = '~/HDD_Storage/projects/biomarker/pulmonary_tuberculosis/data_preprocess/data.T1T10T31.normRobZ.tsv'   # data file to read
DATA_FILE = '~/HDD_Storage/projects/biomarker/pulmonary_tuberculosis/data_preprocess/all.data.clean.for.python.feature.selection.csv'   # data file to read


# which columns are features ?
#FEATURE_COLUMN = 'auto' # could be set as 'auto' or an object that can be iterated, such as range(15), [0, 2, 4, 5] etc. (first column in data file indexed with 0, and the end of the range should be last feature index plus 1)

# which column is label ? 'auto' means the last column will be marked as 'label'. Currently multi-outputs are not yet implemented
FEATURE_COMBINATION = [9,21]
LABEL_COLUMN = 'auto'


# for prediction task only, if you only want to evaluate prediction methods by drawing ROU, it will be ignored
#FEATURE_VALUE_FOR_PREDICT = [2.71446654, 0.91684694, 3.222903944, 0.06889418, 0.826491408, 1.94040036]  # HEB672
#FEATURE_VALUE_FOR_PREDICT = [1.61086937, -0.10539142, 0.006846153, -0.48808366, 0.636261751, 1.27160698] # 20180907001

IS_RESAMPLE = False
OVERSAMPLER = SMOTE(kind='svm', m_neighbors= 5, k_neighbors= 2)
UNDERSAMPLER = TomekLinks()

# 'remove', 'median', 'mean', 'most_frequent', or other string or integer to fill the NA
DEAL_NA = 'remove'

#CLASSIFIER = svm.SVC( kernel= 'rbf' , probability= True, gamma= 'auto')
#CLASSIFIER = svm.SVC( kernel= 'poly' , probability= True , degree= 2, gamma= 'auto')
CLASSIFIER = LogisticRegression( solver= 'sag', multi_class= 'auto', max_iter=10000)
#CLASSIFIER = GaussianNB()
#CLASSIFIER = KNeighborsClassifier(n_neighbors=3, weights= 'distance', algorithm='brute')
#CLASSIFIER = KNeighborsClassifier(n_neighbors=11, weights= 'distance', algorithm='brute')
#CLASSIFIER = RandomForestClassifier()

VALIDATION_FOLD = 5
REPEAT_FOLD = 20
#================================DO NOT change the code below================================
#=========================unless you really know what you are doing==========================
def main(argvList = sys.argv, argv_int = len(sys.argv)):

    data_file_str = path.realpath(path.expanduser( DATA_FILE ))
    data_pd = pd.read_csv( data_file_str, sep='\t')
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

    clf = CLASSIFIER
    print(clf)

    if DEAL_NA == 'remove':
        keep_se = data_pd.iloc[:, FEATURE_COMBINATION].notna().all(axis = 1)
        X = data_pd.loc[keep_se, data_pd.columns[FEATURE_COMBINATION]].values
        if len(X) == 0:
            message = 'Error: Feature combination {} has no samples/observations or are all NaN.'.format(FEATURE_COMBINATION)
            print(message)
            sys.exit()
        Y = data_pd.loc[keep_se, data_pd.columns[label_int]].values.reshape(sum(keep_se))
    elif DEAL_NA in ['median', 'mean', 'most_frequent']:
        X = data_pd.iloc[:, FEATURE_COMBINATION].values
        IMP = SimpleImputer(missing_values=np.nan, strategy=DEAL_NA)
        IMP.fit(X)
        X = IMP.transform(X)
        Y = data_pd.iloc[:, [label_int]].values.reshape(len(data_pd))
    else:
        X = data_pd.iloc[:, FEATURE_COMBINATION].values
        IMP = SimpleImputer(missing_values=np.nan, strategy = 'constant', fill_value = DEAL_NA)
        IMP.fit(X)
        X = IMP.transform(X)
        Y = data_pd.iloc[:, [label_int]].values.reshape(len(data_pd))

    # test if Y only has one class
    if len(set(Y)) == 1:
        message = 'Only one class "{label}" in non-NaN feature-label pair (feature {combination}). This feature will be skipped.'.format(label= set(Y), combination= FEATURE_COMBINATION)
        print(message)
        sys.exit()
    else:
        print('Y Classes: ', collections.Counter(Y))

    #kf = ShuffleSplit(n_splits= REPEAT_FOLD, test_size= 1/VALIDATION_FOLD)  # maximum the splits, we will get enough test data
    #kf = RepeatedKFold( VALIDATION_FOLD, REPEAT_FOLD )
    kf = StratifiedShuffleSplit(n_splits= REPEAT_FOLD, test_size= 1/ VALIDATION_FOLD)
    print(kf)

    overall_error_lst = []  # to store total false positive error plue false negative error rate
    overall_error_float = 0.0
    test_error_lst = []  # to store prediction error rate
    test_error_float = 0.0
    AUCs_lst = []
    AUC_float = 0.0
    overfit_score_lst = []
    overfit_score_float = 0.0

    threshold_proper_lst = []
    FPRmean_ar = np.linspace(0, 1, 100)
    TPRs_lst = []
    i = 0
    time_int = round(time.time())
    for trainIndex_ar, testIndex_ar in kf.split(X, Y):
        print('=======================================================')
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
            X_test_ar = X[testIndex_ar]; Y_test_ar = Y[testIndex_ar]

        # fit the classifier
        try:
            clf.fit( X_train_ar, Y_train_ar )
            Y_predict_ar = clf.predict( X_test_ar )
            Y_overall_predict_ar = clf.predict(X)
            probas_ar = clf.predict_proba( X_test_ar )
            probas_all_ar = clf.predict_proba(X)
        except ValueError as ex:
            message = 'The following error occured in one of the folds of cross-validataion. This fold will be skipped.\n{ex}\nTrainindex:{trainindex}\nTestindex:{testindex}'.format(ex= ex,
                                                                                                                                                                                      trainindex= list(trainIndex_ar),
                                                                                                                                                                                      testindex= list(testIndex_ar))
            print(message)
            continue

        #calculate test data prediction error
        error_ar = Y_predict_ar != Y_test_ar # this is for catalogic prediction
        test_error_float = sum(error_ar)/len(Y_predict_ar)
        test_error_lst.append( test_error_float )  # record test data set prediction error

        #calculate overall data prediction error
        error_ar = Y_overall_predict_ar != Y
        overall_error_float = sum(error_ar)/len(Y)
        overall_error_lst.append( overall_error_float )

        #calculate AUC
        pos_label_lst = list(set( Y_test_ar ))
        pos_label_lst.sort()
        TPR_ar, FPR_ar, thresholds_ar = roc_curve( Y_test_ar, probas_ar[:, 1], pos_label= pos_label_lst[0] )
        try:
            if len(set( Y_test_ar )) == 1: # if test set only has one class then there's no way to calculate false positive rate
                AUC_float = np.NaN
            else:
                AUC_float = auc(FPR_ar, TPR_ar)
        except:
            AUC_float = np.NaN
        AUCs_lst.append(AUC_float)   # record AUC

        # calculate overfit score
        if overall_error_float == 0.0 or test_error_float == 0.0:
            overfit_score_float = np.NaN
        else:
            error_ratio_float = test_error_float / overall_error_float
            #overfit_score_float = (error_ratio_float-1)/(VALIDATION_FOLD-1)
            overfit_score_float = math.log(error_ratio_float, VALIDATION_FOLD)
            if overfit_score_float > 1:
                overfit_score_float = 1
            elif overfit_score_float < 0:
                overfit_score_float = 0
            else:
                pass
        overfit_score_lst.append(overfit_score_float)

        # calculte the optimal threshold that makes TPR-FRP maximum
        temp_ar = TPR_ar - FPR_ar
        try:
            index_int = temp_ar.tolist().index(temp_ar.max())
            threshold_proper_float = thresholds_ar[index_int]
            TPR_proper_float = TPR_ar[index_int]
            FPR_proper_float = FPR_ar[index_int]
        except ValueError:
            threshold_proper_float = np.NaN
        threshold_proper_lst.append( threshold_proper_float )
        TPRs_lst.append( np.interp(FPRmean_ar, FPR_ar, TPR_ar) ) # a list with each item is a ndarray(100,)

        i += 1
        print('{}/{}:'.format(i, REPEAT_FOLD))
        print('Predict probabilities:\n', probas_ar)
        print('Train Idx:\n', trainIndex_ar )
        print(' Test Idx:\n', testIndex_ar )
        #print(' Distance: ', clf.decision_function(X_test_ar) )
        print('  Predict:\n', Y_predict_ar)
        print('     Real:\n', Y_test_ar)
        print()
        print('Overall predict probabilities:\n', probas_all_ar)
        print('   Overall Test:\n', str(Y))
        #print('       Distance:', clf.decision_function(X) )
        print('Overall Predict:\n', str(Y_overall_predict_ar))
        print()
        print( 'Threshold {:.3f} will give the maximun TPR-FPR: {:.3f} (TPR:{:.4f}  FPR:{:.4f})\n'.format( threshold_proper_float, temp_ar.max(), TPR_proper_float, FPR_proper_float ) )
        print('{}/{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}       '.format(i, REPEAT_FOLD, test_error_float, overall_error_float, AUC_float, overfit_score_float), end='\r' )

        # draw i-th ROC curve
        #plt.plot(FPR_ar, TPR_ar, lw=1, alpha=0.3, label='ROC fold {} (AUC = {:.2f})'.format(i, AUC_float) )
        plt.plot(FPR_ar, TPR_ar, lw=1, alpha=0.3 )
        #==========================iteration over==========================

    # draw ROC for "Random guess"
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random guess', alpha=0.8)

    # draw mean ROC
    TPRmean_ar = np.mean(TPRs_lst, axis=0)
    TPRmean_ar[-1] = 1.0
    TPRmean_ar[0] = 0.0
    AUCmean_float = auc(FPRmean_ar, TPRmean_ar)
    AUCstd_float = np.std( AUCs_lst )
    # draw the curve itself
    plt.plot(FPRmean_ar, TPRmean_ar, color='b', label='Mean ROC (AUC = {:.2f} \u00b1 {:.2f})'.format(AUCmean_float, AUCstd_float), lw=2, alpha=0.8)

    #draw shadow for standard deviation
    TPRstd_float = np.std( TPRs_lst, axis=0 )
    TPRupper_ar = np.minimum(TPRmean_ar + TPRstd_float, 1)
    TPRlower_ar = np.maximum(TPRmean_ar - TPRstd_float, 0)
    plt.fill_between(TPRmean_ar, TPRlower_ar, TPRupper_ar, color='grey', alpha=0.2, label='\u00b1 1 std. dev. Threshold: {:.3f}'.format( np.array( threshold_proper_lst ).mean() ) )

    # make the plot nicer
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic(ROC) curve')
    plt.legend(loc="lower right")
    print('\n==============================================================================================================')
    print('Features Combination,\tTest Sets Error,\tOverall Error,\tAUC,\tOverfit Score,\tCounter')
    print( '{}\t{:f}\t{:f}\t{:f}\t{:f}\t{}'.format(FEATURE_COMBINATION, np.mean(test_error_lst), np.mean(overall_error_lst), np.nanmean(AUCs_lst), np.nanmean(overfit_score_lst), collections.Counter(Y) ) )
    print()
    try:
        temp = TPRmean_ar - FPRmean_ar
        index_int = temp.tolist().index(temp.max())
        TPR_proper_float = TPRmean_ar[index_int]
        FPR_proper_float = FPRmean_ar[index_int]
        threshold_proper_float = np.nanmean(threshold_proper_lst)
    except:
        pass
    print( 'Threshold {:.3f} will give the maximun TPR-FPR: {:.3f} (TPR:{:.4f}  FPR:{:.4f})\n'.format( threshold_proper_float, temp.max(), TPR_proper_float, FPR_proper_float ) )
    plt.show()

    return


main()
