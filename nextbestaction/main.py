import config
import train as train
import test as test
import metric
import util
from preprocess import Preprocess_Manager
from datetime import datetime

accuracy_values = list()
accuracy_sum = 0.0
accuracy_value = 0.0
precision_values = list()
precision_sum = 0.0
precision_value = 0.0
recall_values = list()
recall_sum = 0.0
recall_value = 0.0
f1_values = list()
f1_sum = 0.0
f1_value = 0.0
training_time_seconds = list()

args = ""

if __name__ == '__main__':

    args = config.load()
    preprocess_manager = Preprocess_Manager(args)

    # k-fold cross validation
    if args.cross_validation:

        # iterate folds
        for iteration_cross_validation in range(0, args.num_folds):
            preprocess_manager.iteration_cross_validation = iteration_cross_validation
            training_time_seconds.append(train.train(args, preprocess_manager))
            args.iteration_cross_validation = iteration_cross_validation
            test.test(args, preprocess_manager)
            accuracy_value, precision_value, recall_value, f1_value = metric.calc_metrics(args)
            accuracy_values.append(accuracy_value)
            precision_values.append(precision_value)
            recall_values.append(recall_value)
            f1_values.append(f1_value)

        # final output
        for index in range(0, len(accuracy_values)):
            util.llprint("Accuracy of fold %i: %f\n" % (index + 1, accuracy_values[index]))
            util.llprint("Precision of fold %i: %f\n" % (index + 1, precision_values[index]))
            util.llprint("Recall of fold %i: %f\n" % (index + 1, recall_values[index]))
            util.llprint("F1-Score of fold %i: %f\n" % (index + 1, f1_values[index]))
            util.llprint("Training time of fold %i: %f seconds\n" % (index + 1, training_time_seconds[index]))

        util.llprint(
            "Average accuracy %i-fold cross-validation: %f\n" % (args.num_folds, sum(accuracy_values) / args.num_folds))
        util.llprint("Average precision precision %i-fold cross-validation: %f\n" % (
            args.num_folds, sum(precision_values) / args.num_folds))
        util.llprint(
            "Average recall %i-fold cross-validation: %f\n" % (args.num_folds, sum(recall_values) / args.num_folds))
        util.llprint(
            "Average f1-score %i-fold cross-validation: %f\n" % (args.num_folds, sum(f1_values) / args.num_folds))
        util.llprint("Average training time in seconds %i-fold cross-validation: %f\n" % (
            args.num_folds, sum(training_time_seconds) / args.num_folds))
        util.llprint(
            "Total training time %i-fold cross-validation: %f seconds\n" % (args.num_folds, sum(training_time_seconds)))

        # x/y split validation
    else:
        training_time = train.train(args, preprocess_manager)

        start_testing_time = datetime.now()
        test.test(args, preprocess_manager)
        testing_time = datetime.now() - start_testing_time
        testing_time = testing_time.total_seconds()

        accuracy_value, precision_value, recall_value, f1_value = metric.calc_metrics(args)

        # final output
        util.llprint(
            "Accuracy %i/%i: %f\n" % (100 * (1 / args.num_folds), 100 * (1 - 1 / args.num_folds), accuracy_value))
        util.llprint(
            "Precision %i/%i: %f\n" % (100 * (1 / args.num_folds), 100 * (1 - 1 / args.num_folds), precision_value))
        util.llprint("Recall %i/%i: %f\n" % (100 * (1 / args.num_folds), 100 * (1 - 1 / args.num_folds), recall_value))
        util.llprint("F1-Score %i/%i: %f\n" % (100 * (1 / args.num_folds), 100 * (1 - 1 / args.num_folds), f1_value))
        util.llprint("Training time %i/%i: %f seconds\n" % (
            100 * (1 - 1 / args.num_folds), 100 * (1 / args.num_folds), training_time))
        util.llprint("Testing time %i/%i: %f seconds\n" % (
            100 * (1 - 1 / args.num_folds), 100 * (1 / args.num_folds), testing_time))
