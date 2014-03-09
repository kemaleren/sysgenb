from __future__ import division

import os

import numpy as np
from scipy.stats import spearmanr
import pandas as pd

from sklearn.linear_model import ElasticNetCV, LinearRegression
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


def read_name_file(filename):
    with file(filename) as f:
        return f.read().split('\n')


def get_names():
    gene_file = os.path.join(DATA_DIR, 'DREAM5_SysGenB_GenotypeMarkerIDs.csv')
    expression_file = os.path.join(DATA_DIR,
                                   'DREAM5_SysGenB_ExpressionProbeIDs.csv')
    return read_name_file(gene_file), read_name_file(expression_file)


def make_data(genotype, expression):
    """concatenate genotype and expression data

    Returns the feature array and a list of feature names.

    """
    # standardize expression
    m = expression.mean(axis=0)
    s = expression.std(axis=0, ddof=1)
    expression = (expression - m) / s

    genotype_names, expression_names = get_names()
    feats = np.hstack((genotype, expression))
    names = genotype_names + expression_names

    return genotype, genotype_names
    return feats, names


def make_rates(coefs):
    return (coefs != 0).sum(axis=0) / coefs.shape[0]


def train_model(data, target, n_iter=3, rate=0.5):
    """Bootstraps, trains ElasticNetCV model, selects features, and
    trains final linear regression model.

    Returns model and selected features.

    """
    coefs = []
    for i in range(n_iter):
        print "bootstrap iter {}".format(i)
        indices = np.random.choice(len(data), size=len(data), replace=True)
        sample_data = data[indices]
        sample_target = target[indices]
        model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                             max_iter=10000)
        model.fit(sample_data, sample_target)
        coefs.append(model.coef_)
    coefs = np.vstack(coefs)
    rate_selected = make_rates(coefs)
    selected = np.nonzero(rate_selected >= rate)[0]
    model = LinearRegression()
    model.fit(data[:, selected], target)
    return model, selected, coefs


def do_fold(name, data, target, data_test, target_test):
    model, selected, coefs = train_model(data, target)
    preds = model.predict(data_test[:, selected])
    n_feats = len(selected)
    r, p = spearmanr(target_test, preds)
    print "{} n_features : {}, r: {}, p: {}".format(
        name, n_feats, r, p)
    return coefs, p


if __name__ == "__main__":
    genotype, expression, percent_present, scale_factor = \
        read_training_data()
    data, feature_names = make_data(genotype, expression)

    kf = KFold(len(data), n_folds=2, random_state=0)

    k = 0

    all_coefs_percent = []
    all_coefs_scale = []
    for train_index, test_index in kf:
        print "fold: {}".format(k)
        k += 1
        data_train = data[train_index]
        percent_present_train = percent_present[train_index]
        scale_factor_train = scale_factor[train_index]

        data_test = data[test_index]
        percent_present_test = percent_present[test_index]
        scale_factor_test = scale_factor[test_index]

        coefs_percent, p_percent = do_fold(
            'percent present', data_train, percent_present_train, data_test,
            percent_present_test)
        all_coefs_percent.append(coefs_percent)

        coefs_scale, p_scale = do_fold(
            'scale_factor', data_train, scale_factor_train, data_test,
            scale_factor_test)
        all_coefs_scale.append(coefs_scale)

        score = -np.log(p_percent * p_scale)
        print "overall score: {}".format(score)

    rates_percent = make_rates(np.vstack(all_coefs_percent))
    rates_scale = make_rates(np.vstack(all_coefs_scale))

    # save features and their selection rate
    df = pd.DataFrame(zip(feature_names, rates_percent, rates_scale),
                      columns=('feature', 'rate_percent_present',
                               'rate_scale'))
    df.to_csv('feature_selection_rates.csv')

    # make biplots
