import argparse
import os
import re

import pandas
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer


def _create_model():
    """Creates classifier model"""
    return RandomForestClassifier(n_estimators=100)


def _normalize_ingredient(ingredient):
    return " ".join(re.sub(r'\([^)]*\)', '', ingredient).lower().split())


def _train_model(model, train_df):
    """
    Trains classifier model
    @param model: model to train
    @param train_df: data-frame to use for training
    @return: (ingredients transformer, cuisines transformer)
    """
    cuisines_transformer = DictVectorizer(dtype=numpy.uint8)
    cuisines_transformer.fit({r['cuisine']: 1} for _, r in train_df.iterrows())

    ingredients_transformer = DictVectorizer(dtype=numpy.uint8)
    params = ingredients_transformer.fit_transform(dict((_normalize_ingredient(i), 1)
                                                        for i in r['ingredients']) for _, r in train_df.iterrows())
    outputs = numpy.fromiter((cuisines_transformer.vocabulary_[r['cuisine']]
                              for _, r in train_df.iterrows()), numpy.uint8)

    model.fit(params, outputs)

    return ingredients_transformer, cuisines_transformer


def _verify_model(model, verify_df, ingredients_transformer, cuisines_transformer):
    """Verify trained model on specified data-frame"""
    verify_params = ingredients_transformer.transform(dict((_normalize_ingredient(i), 1)
                                                           for i in r['ingredients']) for _, r in verify_df.iterrows())
    verify_outputs = numpy.fromiter((cuisines_transformer.vocabulary_[r['cuisine']]
                                     for _, r in verify_df.iterrows()), numpy.uint8)

    return model.score(verify_params, verify_outputs)


def verify_handler(args):
    if not os.path.isfile(args.train_file):
        print "Train file not found: {0}".format(args.file)
        return

    if not os.path.isfile(args.verify_file):
        print "Verify file not found: {0}".format(args.file)
        return

    print "Loading data..."
    train_df = pandas.read_json(args.train_file)

    print "Training model..."
    model = _create_model()
    ingredients_transformer, cuisines_transformer = _train_model(model, train_df)

    print "Verification..."
    verify_df = pandas.read_json(args.verify_file)
    result = _verify_model(model, verify_df, ingredients_transformer, cuisines_transformer)
    print "Accuracy: {0:.4f}".format(result)


def evaluate_handler(args):
    if not os.path.isfile(args.train_file):
        print "Train file not found: {0}".format(args.file)
        return

    if not os.path.isfile(args.evaluate_file):
        print "Evaluation file not found: {0}".format(args.file)
        return

    print "Loading data..."
    train_df = pandas.read_json(args.train_file)

    print "Training model..."
    model = _create_model()
    ingredients_transformer, cuisines_transformer = _train_model(model, train_df)

    evaluate_df = pandas.read_json(args.evaluate_file)
    params = ingredients_transformer.transform(dict((_normalize_ingredient(i), 1)
                                                    for i in r['ingredients']) for _, r in evaluate_df.iterrows())

    print "Evaluation..."
    results = model.predict(params)
    evaluate_df['cuisine'] = [cuisines_transformer.get_feature_names()[v] for v in results]
    evaluate_df[['id', 'cuisine']].to_csv(args.output_file, index=False)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_verify = subparsers.add_parser('verify', help='verify model on specified data-set')
    parser_verify.add_argument('-t', '--train-file', type=str, help='train data file location', required=True)
    parser_verify.add_argument('-v', '--verify-file', type=str, help='verification data file location', required=True)
    parser_verify.set_defaults(handler=verify_handler)

    parser_evaluate = subparsers.add_parser('evaluate', help='evaluate model on specified data-set')
    parser_evaluate.add_argument('-t', '--train-file', type=str, help='train data file location', required=True)
    parser_evaluate.add_argument('-e', '--evaluate-file', type=str, help='evaluation data file location', required=True)
    parser_evaluate.add_argument('-o', '--output-file', type=str, help='output csv file location', required=True)
    parser_evaluate.set_defaults(handler=evaluate_handler)

    args = parser.parse_args()
    args.handler(args)


if __name__ == '__main__':
    main()
