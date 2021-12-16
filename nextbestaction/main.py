import nextbestaction.config as config
import nextbestaction.train as train
import nextbestaction.test as test
import nextbestaction.metric as metric
from nextbestaction.preprocess import Preprocess_Manager
from datetime import datetime

if __name__ == '__main__':

    args = config.load()
    preprocess_manager = Preprocess_Manager(args)

    training_time = train.train(args, preprocess_manager)
    start_testing_time = datetime.now()
    test.test(args, preprocess_manager)

    testing_time = (datetime.now() - start_testing_time).total_seconds()
    accuracy_value, precision_value, recall_value, f1_value = metric.calc_metrics(args)
    print("Accuracy: %f\n" % accuracy_value)
    print("Precision (w): %f\n" % precision_value)
    print("Recall (w): %f\n" % recall_value)
    print("F1-score (w): %f\n" % f1_value)
    print("Training time: %f seconds\n" % training_time)
    print("Testing time: %f seconds\n" % testing_time)