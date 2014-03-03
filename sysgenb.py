import os

import numpy as np
from scipy.stats import spearmanr
import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.cross_validation import KFold

DATA_DIR = "../data"
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')


def read_data_file(f):
    data = pd.read_csv(f, sep='\t').dropna(axis=1)
    return np.array(data).T


def read_training_data():
    genotype = read_data_file(os.path.join(
        TRAIN_DIR, 'DREAM5_SysGenB_TrainingGenotypeData.txt'))
    expression = read_data_file(os.path.join(
        TRAIN_DIR, 'DREAM5_SysGenB_TrainingExpressionData.txt'))
    phenotype = read_data_file(os.path.join(
        TRAIN_DIR, 'DREAM5_SysGenB_TrainingPhenotypeData.txt'))
    percent_present = phenotype[:, 0].ravel()
    scale_factor = phenotype[:, 1].ravel()
    return genotype, expression, percent_present, scale_factor


def read_gold_standard_data():
    # NOTE: do not use this until we know the final model
    genotype = read_data_file(os.path.join(
        TEST_DIR, 'DREAM5_SysGenB3_TestGenotypeData.txt'))
    expression = read_data_file(os.path.join(
        TEST_DIR, 'DREAM5_SysGenB3_TestExpressionData.txt'))
    phenotype = read_data_file(os.path.join(
        TEST_DIR, 'DREAM5_SysGenB3_GoldStandard.txt'))
    percent_present = phenotype[:, 0].ravel()
    scale_factor = phenotype[:, 1].ravel()
    return genotype, expression, percent_present, scale_factor


def make_features(genotype, expression):
    """concatenate genotype and expression data

    Returns the feature array and a list of feature names.

    """
    # standardize expression
    m = expression.mean(axis=0)
    s = expression.std(axis=0, ddof=1)
    expression = (expression - m) / s

    genotype_names = list("genotype_{}".format(i)
                          for i in range(genotype.shape[1]))
    expression_names = list("expression_{}".format(i)
                            for i in range(expression.shape[1]))
    feats = np.hstack((genotype, expression))
    names = genotype_names + expression_names

    return feats, names


if __name__ == "__main__":
    genotype, expression, percent_present, scale_factor = \
        read_training_data()
    features, feature_names = make_features(genotype, expression)

    model_present = Lasso(alpha=0.1)
    model_scale = Lasso(alpha=0.1)

    kf = KFold(len(features))

    k = 0
    for train_index, test_index in kf:
        print "fold: {}".format(k)
        k += 1
        train_data = features[train_index]
        train_present = percent_present[train_index]
        train_scale = scale_factor[train_index]

        test_data = features[test_index]
        test_present = percent_present[test_index]
        test_scale = scale_factor[test_index]

        model_present.fit(train_data, train_present)
        preds_present = model_present.predict(test_data)
        present_n_feats = np.count_nonzero(model_present.coef_)
        present_r, present_p = spearmanr(test_present, preds_present)
        print "percent_present n_features : {}, r: {}, p: {}".format(
            present_n_feats, present_r, present_p)

        model_scale.fit(train_data, train_scale)
        preds_scale = model_scale.predict(test_data)
        scale_n_feats = np.count_nonzero(model_scale.coef_)
        scale_r, scale_p = spearmanr(test_scale, preds_scale)
        print "scale_factor n_features : {}, r: {}, p: {}".format(
            scale_n_feats, scale_r, scale_p)

        score = -np.log(present_p * scale_p)
        print "overall score: {}".format(score)
