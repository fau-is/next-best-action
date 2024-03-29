import os
import argparse
import nextbestaction.util as util


def load():
    parser = argparse.ArgumentParser()

    # dnn
    parser.add_argument('--task', default="next_best_activity")
    parser.add_argument('--dnn_num_epochs', default=100, type=int)
    parser.add_argument('--learning_rate', default=0.002, type=float)  # see Tax et al. (2017)

    # evaluation
    parser.add_argument('--tax_features', default=True, type=util.str_to_bool)  # 5 features from Tax et al. (2017)
    parser.add_argument('--validation_size', default=0.2, type=float)
    parser.add_argument('--train_size', default=0.67, type=float)
    parser.add_argument('--batch_size_train', default=32, type=int)

    # data
    parser.add_argument('--data_set', default="helpdesk.csv")  # helpdesk.csv, bpi2019.csv
    parser.add_argument('--data_dir', default="../data/")
    parser.add_argument('--checkpoint_dir', default="../checkpoints/")
    parser.add_argument('--result_dir', default="../results/")
    parser.add_argument('--dcr_path', default="helpdesk_dcr.xml")  # helpdesk_dcr.xml, bpi2019_dcr.xml

    # next best event
    parser.add_argument('--next_best_action', default=False, type=util.str_to_bool)
    parser.add_argument('--num_candidates', default=50, type=int)
    parser.add_argument('--semantic_check', default=True, type=util.str_to_bool)

    # gpu processing
    parser.add_argument('--gpu_device', default="0", type=str)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    return args
