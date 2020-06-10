import os
import argparse
import util


def load():
    parser = argparse.ArgumentParser()

    # dnn
    parser.add_argument('--task', default="next_best_activity")
    parser.add_argument('--dnn_num_epochs', default=100, type=int)
    parser.add_argument('--dnn_architecture', default=0, type=int)
    parser.add_argument('--learning_rate', default=0.002, type=float)

    # evaluation
    parser.add_argument('--num_folds', default=3, type=int)
    parser.add_argument('--cross_validation', default=False, type=util.str2bool)
    parser.add_argument('--tax_features', default=True, type=util.str2bool)  # five control-flow feature from Tax et al. (2017)
    parser.add_argument('--batch_size_train', default=256, type=int)
    parser.add_argument('--batch_size_test', default=1, type=int)

    # data
    parser.add_argument('--data_set', default="helpdesk_converted.csv")
    parser.add_argument('--data_dir', default="../data/")
    parser.add_argument('--checkpoint_dir', default="./checkpoints/")
    parser.add_argument('--result_dir', default="./results/")
    parser.add_argument('--dcr_path', default="helpdesk_dcr.xml")

    # next best event
    parser.add_argument('--next_best_action', default=True, type=util.str2bool)
    parser.add_argument('--num_candidates', default=15, type=int)
    parser.add_argument('--semantic_check', default=True, type=util.str2bool)

    # gpu processing
    parser.add_argument('--gpu_ratio', default=1.0, type=float)
    parser.add_argument('--cpu_num', default=6, type=int)
    parser.add_argument('--gpu_device', default="0", type=str)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    return args
