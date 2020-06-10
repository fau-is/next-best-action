import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
import util


def calc_metrics(args):
    
    prefix = 0
    prefix_all_enabled = 1
    
    prediction = list()
    gt_label = list()
    
    result_dir = args.result_dir
    if not args.cross_validation:
        result_dir_fold = result_dir + args.data_set.split(".csv")[0] + "__" + args.task + "_0.csv"    
    else:
        result_dir_fold = result_dir + args.data_set.split(".csv")[0] + "__" + args.task + "_%d" % args.iteration_cross_validation + ".csv"
    
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
            
    util.llprint("\n\n")
    util.llprint("Metrics:\n")  
    util.llprint("Accuracy: %f\n" % accuracy_score(gt_label, prediction))
    util.llprint("Precision (weighted): %f\n" % precision_score(gt_label, prediction,average='weighted'))
    util.llprint("Recall (weighted): %f\n" % recall_score(gt_label, prediction,average='weighted'))
    util.llprint("F1-Score (weighted): %f\n" % f1_score(gt_label, prediction,average='weighted'))
    util.llprint("Precision (macro): %f\n" % precision_score(gt_label, prediction,average='macro'))
    util.llprint("Recall (macro): %f\n" % recall_score(gt_label, prediction,average='macro'))
    util.llprint("F1-Score (macro): %f\n" % f1_score(gt_label, prediction,average='macro'))
    util.llprint("Precision (micro): %f\n" % precision_score(gt_label, prediction,average='micro'))
    util.llprint("Recall (micro): %f\n" % recall_score(gt_label, prediction,average='micro'))
    util.llprint("F1-Score (micro): %f\n\n" % f1_score(gt_label, prediction,average='micro'))

    return accuracy_score(gt_label, prediction), \
           precision_score(gt_label, prediction, average='weighted'), \
           recall_score(gt_label, prediction, average='weighted'), \
           f1_score(gt_label, prediction, average='weighted')