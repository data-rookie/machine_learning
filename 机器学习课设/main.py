import os
import sys
import datetime
import argparse
import numpy as np
import pandas as pd

from utils import *
from svm import SVM

np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser(description='Binary support vector classifer.')
parser.add_argument('--header_filename', dest='header_filename', type=str, default='alphabet_DU_training_header.csv', help='Training dataset with header csv. This is used to output hypothesis. (Default: "alphabet_DU_training_header.csv")')
parser.add_argument('--train_filename', dest='train_filename', type=str, default='alphabet_DU_training.csv', help='Training dataset csv. (Default: "alphabet_DU_training.csv")')
parser.add_argument('--test_filename', dest='test_filename', type=str, default='alphabet_DU_testing.csv', help='Training dataset csv. (Default: "alphabet_DU_testing.csv")')
parser.add_argument('--K', dest='K', type=int, default=None, help='Denotes for "K"-fold cross-validation for determine the optimal hyper-parameters for SVM. (Default: None)')
parser.add_argument('--C', dest='C', type=float, default=0.1, help='Parameter for penalty term. (Default: 0.1)')
parser.add_argument('--kernel_type', dest='kernel_type', type=str, default='linear', help='Kernel type to be used in SVM. Acceptable kernel type: "linear", "poly", "rbf". (Default: None)')
parser.add_argument('--poly_degree', dest='poly_degree', type=int, default=3, help='Degree of the polynomial kernel function ("poly"). Ignored by all other kernels. (Default: 3)')
parser.add_argument('--rbf_sigma', dest='rbf_sigma', type=float, default=0.5, help='Sigma term in RBF (guassian). Ignored by all other kernels. (Default: 0.5)')
parser.add_argument('--enable_heuristic', dest='enable_heuristic', action='store_true', default=False, help='Whether use Platts heuristics to train SVM. (Defualt: False)')
parser.add_argument('--enable_kernel_cache', dest='enable_kernel_cache', action='store_true', default=True, help='Whether precompute kernel results. This can speed up training but need time to initialize when data is large. (Defualt: True)')
parser.add_argument('--max_iteration', dest='max_iteration', type=int, default=3000, help='Max iteration for SMO training algorithm to avoid not converging. (Defualt: 3000)')

args = parser.parse_args()

if __name__ == '__main__':

    # Check file paths exist
    header_filepath = 'dataset/' + args.header_filename
    train_filepath = 'dataset/' + args.train_filename
    test_filepath = 'dataset/' + args.test_filename

    if not os.path.exists(header_filepath):
        print('[!] File "{}" does not exist.'.format(header_filepath))
        print('[!] Exit program.')
        sys.exit(0)
    if not os.path.exists(train_filepath):
        print('[!] File "{}" does not exist.'.format(train_filepath))
        print('[!] Exit program.')
        sys.exit(0)
    if not os.path.exists(test_filepath):
        print('[!] File "{}" does not exist.'.format(test_filepath))
        print('[!] Exit program.')
        sys.exit(0)

    # Load dataset.
    try:
        data = load_dataset(train_filepath=train_filepath, test_filepath=test_filepath)
    except Exception as e:
        print('[!] {}.'.format(e))
        print('[!] Exit program.')
        sys.exit(0)

    # Perform grid search K-fold cross validation.
    if args.K:
        if args.kernel_type is None:
            print('[!] Please specify argument --kernel_type for K-fold cross-validation.')
            print('[!] Acceptable kernel type: "linear", "poly", "rbf". (Default: None)')
            print('[!] Exit program.')
            sys.exit(0)

        log_filename = 'logs/svm-{}-{}-fold-[{}].txt'.format(args.kernel_type, args.K, datetime.datetime.now().strftime('%H:%M:%S'))

        # Print and write label mapping
        print_and_write(filename=log_filename, log='-'*100)
        print_and_write(filename=log_filename, log='[*] Class label mapping:')
        print_and_write(filename=log_filename, log=' -  {} => +1'.format(data['+1']))
        print_and_write(filename=log_filename, log=' -  {} => -1'.format(data['-1']))

        # Specify list of hyper-parameters for grid search K-fold cross validation.
        param_grid = {
            'C': np.linspace(0.1, 1.0, 6).round(2).tolist(),
            'kernel_type': [args.kernel_type]
        }
        if args.kernel_type == 'poly':
            param_grid['poly_degree'] = [2, 3, 4, 5]
        elif args.kernel_type == 'rbf':
            param_grid['rbf_sigma'] = np.linspace(0.1, 2.0, 4).round(2).tolist()

        cv = GridSearchCV(data=data, model=SVM, param_grid=param_grid, num_folds=args.K)
        cv.train()

        # Print and write CV results to log file.
        print_and_write(filename=log_filename, log='-'*100)
        print_and_write(filename=log_filename, log='[*] Cross validation history:')
        for cv_result in cv.cv_results:
            print_and_write(filename=log_filename, log=' -  Parameter: {} | Cross validation error: {}'.format(cv_result['param'], cv_result['score']))
        print_and_write(filename=log_filename, log='-'*100)
        print_and_write(filename=log_filename, log='[*] Best parameter: {}'.format(cv.best_param))
        print_and_write(filename=log_filename, log='[*] Best cross validation error: {}'.format(cv.best_score))
        print_and_write(filename=log_filename, log='[*] Start to train on full training data and evaluate on test data ...')

        # Train on full training data and evaluate on test data with the best hyper-parameters.
        model = SVM(data=data, param=cv.best_param, verbose=True)
        model.train()

        # Compute training and test error.
        train_error = zero_one_loss(y_truth=data['train_y'], y_pred=model.hypothesis(X=data['train_X']))
        test_error = zero_one_loss(y_truth=data['test_y'], y_pred=model.hypothesis(X=data['test_X']))

        # Print and write results to log file.
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        print_and_write(filename=log_filename, log='-'*100)
        print_and_write(filename=log_filename, log='[*] Train file path: "{}"'.format(train_filepath))
        print_and_write(filename=log_filename, log='[*] Test file path: "{}"'.format(test_filepath))
        print_and_write(filename=log_filename, log='[*] Datetime: {}'.format(timestamp))
        print_and_write(filename=log_filename, log='[*] Best parameter: {}'.format(cv.best_param))
        if model.use_w: print_and_write(filename=log_filename, log='[*] Weight vector: {}'.format(model.w))
        print_and_write(filename=log_filename, log='[*] Sample mean of bias: {}'.format(model.b_mean))
        print_and_write(filename=log_filename, log='[*] Sample std of bias: {}'.format(model.b_std))
        print_and_write(filename=log_filename, log='[*] Performance: Train error: {} | Test error: {}'.format(train_error, test_error))
        print_and_write(filename=log_filename, log='-'*100)

        # Output SVM hypothesis (need training data with header)
        hypothesis_filepath = 'hypothesis/SVM_hypothesis_header-[{}].csv'.format(timestamp)
        print_and_write(filename=log_filename, log='[*] Saving SVM hypothesis to "{}" ...'.format(hypothesis_filepath))
        try:
            csv_in = pd.read_csv(header_filepath)
            csv_in['lambda'] = model.alpha
            csv_in['offset'] = model.postcomputed_biases
            csv_in.to_csv(hypothesis_filepath, index=False)
            print_and_write(filename=log_filename, log='[*] Output SVM hypothesis to "{}" success.'.format(hypothesis_filepath))
        except Exception as e:
            print_and_write(filename=log_filename, log='[!] {}.'.format(e))
            print_and_write(filename=log_filename, log='[!] Output SVM hypothesis failed.')
        print_and_write(filename=log_filename, log='-'*100)

    else:
        if args.kernel_type is None:
            print('[!] Please specify argument --kernel_type.')
            print('[!] Acceptable kernel type: "linear", "poly", "rbf". (Default: None)')
            print('[!] Exit program.')
            sys.exit(0)

        # Print and write label mapping
        print('-'*100)
        print('[*] Class label mapping:')
        print(' -  {} => +1'.format(data['+1']))
        print(' -  {} => -1'.format(data['-1']))

        # Specify hyper-parameters
        param = {
            'C': args.C,
            'kernel_type': args.kernel_type,
        }
        if args.kernel_type == 'poly':
            param['poly_degree'] = args.poly_degree
        elif args.kernel_type == 'rbf':
            param['rbf_sigma'] = args.rbf_sigma

        # Train on full training data and evaluate on test data with the specified hyper-parameters
        model = SVM(data=data, param=param,
                    enable_heuristic=args.enable_heuristic,
                    enable_kernel_cache=args.enable_kernel_cache,
                    max_iteration=args.max_iteration,
                    verbose=True)

        model.train(info='Train on specified parameter: {}'.format(param))

        # Compute training and test error.
        train_error = zero_one_loss(y_truth=data['train_y'], y_pred=model.hypothesis(X=data['train_X']))
        test_error = zero_one_loss(y_truth=data['test_y'], y_pred=model.hypothesis(X=data['test_X']))

        # Print results
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        print('-'*100)
        print('[*] Datetime: {}'.format(timestamp))
        print('[*] Train file path: "{}"'.format(train_filepath))
        print('[*] Test file path: "{}"'.format(test_filepath))
        print('[*] Specified parameter: {}'.format(param))
        if model.use_w: print('[*] Weight vector: {}'.format(model.w))
        print('[*] Sample mean of bias: {}'.format(model.b_mean))
        print('[*] Sample std of bias: {}'.format(model.b_std))
        print('[*] Performance: Train error: {} | Test error: {}'.format(train_error, test_error))
        print('-'*100)

        # Output SVM hypothesis (need training data with header)
        hypothesis_filepath = 'hypothesis/SVM_hypothesis_header-[{}].csv'.format(timestamp)
        print('[*] Saving SVM hypothesis to "{}" ...'.format(hypothesis_filepath))
        try:
            csv_in = pd.read_csv(header_filepath)
            csv_in['lambda'] = model.alpha
            csv_in['offset'] = model.postcomputed_biases
            csv_in.to_csv(hypothesis_filepath, index=False)
            print('[*] Output SVM hypothesis to "{}" success.'.format(hypothesis_filepath))
        except Exception as e:
            print('[!] {}'.format(e))
            print('[!] Output SVM hypothesis failed.')
        print('-'*100)
