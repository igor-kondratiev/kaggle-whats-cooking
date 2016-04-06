import argparse
import os

import pandas
import numpy
from sklearn.ensemble import RandomForestClassifier

from data_description import describe
from data_operations import extract_cuisines, extract_ingredients


def _create_model():
    return RandomForestClassifier(n_estimators=100)


def describe_handler(args):
    if not os.path.isfile(args.file):
        print "File not found: {0}".format(args.file)
        return
    else:
        data = pandas.read_json(args.file)
        info = describe(data)
        for k, v in info:
            print "{0}: {1}".format(k, v)


def verify_handler(args):
    if not os.path.isfile(args.train_file):
        print "Train file not found: {0}".format(args.file)
        return

    if not os.path.isfile(args.verify_file):
        print "Verify file not found: {0}".format(args.file)
        return

    print "Extracting ingredients and cuisines..."
    train_df = pandas.read_json(args.train_file)
    cuisines = extract_cuisines(train_df)
    cuisines_dict = dict((name, index) for index, name in enumerate(cuisines))
    ingredients = extract_ingredients(train_df)
    ingredients_dict = dict((name, index) for index, name in enumerate(ingredients))

    print "Preparing model input data..."
    items_count, _ = train_df.shape
    params = numpy.zeros([items_count, len(ingredients)])
    outputs = numpy.zeros([items_count])
    for index, row in train_df.iterrows():
        outputs[index] = cuisines_dict[row['cuisine']]
        for ingredient in row['ingredients']:
            params[index, ingredients_dict[ingredient]] = 1

    print "Training model..."
    model = _create_model()
    model.fit(params, outputs)

    print "Preparing model verification data..."
    verify_df = pandas.read_json(args.train_file)
    items_count, _ = verify_df.shape
    params = numpy.zeros([items_count, len(ingredients)])
    outputs = numpy.zeros([items_count])
    for index, row in verify_df.iterrows():
        outputs[index] = cuisines_dict[row['cuisine']]
        for ingredient in row['ingredients']:
            params[index, ingredients_dict[ingredient]] = 1

    print "Verification..."
    result = model.score(params, outputs)
    print "Accuracy: {0:.4f}".format(result)


def evaluate_handler(args):
    if not os.path.isfile(args.train_file):
        print "Train file not found: {0}".format(args.file)
        return

    if not os.path.isfile(args.evaluate_file):
        print "Evaluation file not found: {0}".format(args.file)
        return

    print "Extracting ingredients and cuisines..."
    train_df = pandas.read_json(args.train_file)
    cuisines = extract_cuisines(train_df)
    cuisines_dict = dict((name, index) for index, name in enumerate(cuisines))
    ingredients = extract_ingredients(train_df)
    ingredients_dict = dict((name, index) for index, name in enumerate(ingredients))

    print "Preparing model input data..."
    items_count, _ = train_df.shape
    params = numpy.zeros([items_count, len(ingredients)])
    outputs = numpy.zeros([items_count])
    for index, row in train_df.iterrows():
        outputs[index] = cuisines_dict[row['cuisine']]
        for ingredient in row['ingredients']:
            params[index, ingredients_dict[ingredient]] = 1

    print "Training model..."
    model = _create_model()
    model.fit(params, outputs)

    print "Preparing model evaluation data..."
    evaluate_df = pandas.read_json(args.evaluate_file)
    items_count, _ = evaluate_df.shape
    params = numpy.zeros([items_count, len(ingredients)])
    for index, row in evaluate_df.iterrows():
        for ingredient in row['ingredients']:
            if ingredient in ingredients_dict:
                params[index, ingredients_dict[ingredient]] = 1

    print "Evaluation..."
    results = model.predict(params)
    evaluate_df['cuisine'] = [cuisines[int(v)] for v in results]
    evaluate_df[['id', 'cuisine']].to_csv(args.output_file, index=False)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_describe = subparsers.add_parser('describe', help='describes specified data file')
    parser_describe.add_argument('-f', '--file', type=str, help='data file location', required=True)
    parser_describe.set_defaults(handler=describe_handler)

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
