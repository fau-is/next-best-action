import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def calc_metrics(args):
    prefix = 0
    prefix_all_enabled = 1
    prediction = list()
    gt_label = list()
    result_dir = args.result_dir
    result_dir_fold = result_dir + args.data_set.split(".csv")[0] + "_" + args.task + ".csv"

    with open(result_dir_fold, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        next(reader)

        for row in reader:
            if row == []:
                continue
            else:
                if int(row[1]) == prefix or prefix_all_enabled == 1:
                    gt_label.append(row[2])
                    prediction.append(row[3])

    return accuracy_score(gt_label, prediction), precision_score(gt_label, prediction, average='weighted'), \
           recall_score(gt_label, prediction, average='weighted'), f1_score(gt_label, prediction, average='weighted')