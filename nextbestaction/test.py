from tensorflow.keras.models import load_model
import csv
import distance
from jellyfish._jellyfish import damerau_levenshtein_distance
import nextbestaction.util as util
from datetime import timedelta
from sklearn import metrics
import pickle
import copy


def predict_suffix_and_time_for_prefix(args, model, preprocess_manager, current, predict, ground_truth):
    in_time = 1

    start = len(ground_truth["prefix_event"])

    for i in range(start, predict["size"]):

        input_vec, num_features_all, num_features_activities = preprocess_manager.encode_test_set_add(current["line"], current["times"], current["times3"], current["line_add"], args.batch_size_test)

        Y = model.predict(input_vec, verbose=0)
        y_event = Y[0][0]
        y_time = Y[1][0][0]
        prediction = preprocess_manager.get_symbol(y_event)

        # update of prefix (event)
        current["line"] += prediction
        predict["predicted"] += prediction

        # update of prefix for suffix prediction (time + context)
        if prediction == '!':
            y_time = 0
            # note times2 will be automatically calculated based on times in the method "encode_test_set_add()"
            current["times"].append(y_time)
            current["times3"].append(current["times3"][-1] + timedelta(seconds=y_time))
            predict["suffix_time"] = predict["suffix_time"] + y_time
            current["line_add"].append(current["line_add"][0])
        else:
            # update of prefix for suffix prediction (time + context)
            if y_time < 0:
                y_time = 0
            y_time = y_time * preprocess_manager.divisor3

            current["times"].append(y_time)
            current["times3"].append(current["times3"][-1] + timedelta(seconds=y_time))
            predict["suffix_time"] = predict["suffix_time"] + y_time
            current["line_add"].append(current["line_add"][0])

        # termination; predict size = max sequence length - 1
        if prediction == '!' or len(current["line"]) == predict["size"]:
            util.ll_print('\n! termination suffix prediction ... \n')
            break

        util.ll_print("Prefix+Suffix-Time-%s: %f" % (i, ground_truth["prefix_time"] + predict["suffix_time"]))
        util.ll_print("Baseline-%s: %f" % (i, preprocess_manager.avg_time_training_cases))

    if ground_truth["prefix_time"] + predict["suffix_time"] > preprocess_manager.avg_time_training_cases:
        in_time = 0
    deviation_in_time = (ground_truth["prefix_time"] + predict[
        "suffix_time"]) / preprocess_manager.avg_time_training_cases

    return predict, in_time, deviation_in_time


def predict_suffix_and_time_for_prefix_next_best_event(args, model, preprocess_manager, current_temp, predict, ground_truth, num_corrections):
    # 0.) store initial prefix
    current = copy.deepcopy(current_temp)
    current_with_next_best_event = copy.deepcopy(current)
    predict_initial = copy.deepcopy(predict)
    in_time = 1
    deviation_in_time = 0

    # represents the actual prefix
    start = len(current_with_next_best_event["line"])
    for j in range(start, predict["size"]):

        # 1.) suffix + time for a given prefix
        # note upper bound = max. predict_size and not size of the ground truth
        # suffix, because that is part of the inductive learning task of the suffix prediction
        start_sub = start
        first_prediction = True
        current_first_prediction = dict()
        for i in range(start_sub, predict["size"]):

            # preprocess prefix
            input_vec, num_features_all, num_features_activities = preprocess_manager.encode_test_set_add(current[
                "line"], current["times"], current["times3"], current["line_add"], args.batch_size_test)

            # make prediction
            Y = model.predict(input_vec, verbose=0)

            # get output of predictor output
            y_event = Y[0][0]
            y_time = Y[1][0][0]
            prediction = preprocess_manager.get_symbol(y_event)

            # update of prefix (event)
            current["line"] += prediction
            predict["predicted"] += prediction

            # update of prefix for suffix prediction (time + context)
            if prediction == '!':
                y_time = 0
                current["times"].append(y_time)
                current["times3"].append(current["times3"][-1] + timedelta(seconds=y_time))
                predict["suffix_time"] = predict["suffix_time"] + y_time
                # retrieve case-based context attributes
                # current["line_add"].append(['0'])
                current["line_add"].append(current["line_add"][0])
            else:
                # update of prefix for suffix prediction (time + context)
                if y_time < 0:
                    y_time = 0
                y_time = y_time * preprocess_manager.divisor3
                current["times"].append(y_time)
                current["times3"].append(current["times3"][-1] + timedelta(seconds=y_time))
                predict["suffix_time"] = predict["suffix_time"] + y_time
                # retrieve case-based context attributes
                current["line_add"].append(current["line_add"][0])

            # save first prediction
            if first_prediction:
                current_first_prediction = copy.deepcopy(current)
                first_prediction = False

            # termination
            if prediction == '!' or len(current["line"]) == predict["size"]:

                util.ll_print('\n! termination suffix prediction ... \n')
                break

        util.ll_print("Prefix+Suffix-Time-%s-%s: %f" % (j, i, (
                current_with_next_best_event["times3"][len(current_with_next_best_event["times3"]) - 1] -
                current_with_next_best_event["times3"][0]).total_seconds() + predict["suffix_time"]))

        util.ll_print("Baseline-%s-%s: %f" % (j, i, preprocess_manager.avg_time_training_cases))

        # 1.) update of current_with_next_best_event
        # assumption: total time in current_with_next_best_event is at the end of times2
        if (current_with_next_best_event["times3"][len(current_with_next_best_event["times3"]) - 1] -
            current_with_next_best_event["times3"][0]).total_seconds() + predict[
            "suffix_time"] <= preprocess_manager.avg_time_training_cases:

            util.ll_print("Do not determine next best event ...")
            # 1.1) update next best with prediction (of next event -> first prediction of suffix)
            current_with_next_best_event = copy.deepcopy(current_first_prediction)

        # 2.) update next best event 
        else:
            util.ll_print("Determine next best event ...")

            num_corrections += 1

            # 2.1) select candidates
            # input params: current; length of current prefix
            model_candidate_selection = pickle.load(open(
                "%smodel_candidate_selection_%s" % (args.checkpoint_dir, preprocess_manager.iteration_cross_validation),
                'rb'))

            # suffix
            current_vector = preprocess_manager.transform_current_instance_to_suffix_vector(current, len(
                current_with_next_best_event["line"]), predict["size"])
            distance_candidates, index_candidates = model_candidate_selection.query(current_vector,
                                                                                    k=args.num_candidates)

            # 2.2) get best candidate from training set with similar suffix
            # input params: index of candidates; length of current prefix
            # print(index_candidates)
            current_candidate_total_time, current_candidate_event_time, current_candidate_add, current_candidate_event, current_candidate = preprocess_manager.select_best_candidate_from_training_set(
                index_candidates, current_with_next_best_event["line"], args)

            if current_candidate == []:
                # 2.3.1) if no candidate is conform to the DCR graphs
                current_with_next_best_event = copy.deepcopy(current_first_prediction)
            else:
                # 2.3.2) update of next likely event through next best event
                current_with_next_best_event["line"] += current_candidate_event
                current_with_next_best_event["times"].append(current_candidate_event_time)
                current_with_next_best_event["times3"].append(
                    current["times3"][-1] + timedelta(seconds=current_candidate_event_time))
                current_with_next_best_event["line_add"].append(current_candidate_add)

        # termination
        if len(current_with_next_best_event["line"]) == predict["size"] or '!' in current_with_next_best_event["line"]:

            util.ll_print('\n! termination prefix ... \n')
            break

        # update of current with next best event
        current = copy.deepcopy(current_with_next_best_event)

        # update of predict with init predict 
        predict = copy.deepcopy(predict_initial)

    # final check if correction has a positive effect
    if (current_with_next_best_event["times3"][len(current_with_next_best_event["times3"]) - 1] -
        current_with_next_best_event["times3"][0]).total_seconds() > preprocess_manager.avg_time_training_cases:
        in_time = 0
    deviation_in_time = (current_with_next_best_event["times3"][len(current_with_next_best_event["times3"]) - 1] -
                         current_with_next_best_event["times3"][
                             0]).total_seconds() / preprocess_manager.avg_time_training_cases

    # construct predict object for output
    predict["predicted"] = current_with_next_best_event["line"][len(ground_truth["prefix_event"]):]
    predict["suffix_time"] = sum(current_with_next_best_event["times"][len(ground_truth["prefix_event"]):])

    return predict, in_time, deviation_in_time, num_corrections


def test(args, preprocess_manager):
    result_dir = args.result_dir
    task = args.task

    # get test set
    if preprocess_manager.num_features_additional > 0:
        lines, caseids, lines_t, lines_t2, lines_t3, lines_add, sequence_max_length, num_features_all, num_features_activities = preprocess_manager.create_test_set()
    else:
        lines, caseids, lines_t, lines_t2, lines_t3, sequence_max_length, num_features_all, num_features_activities = preprocess_manager.create_test_set()

    # load model
    model_suffix_prediction = load_model(
        '%smodel_suffix_prediction_%s.h5' % (args.checkpoint_dir, preprocess_manager.iteration_cross_validation))

    # set options for result output
    data_set_name = args.data_set.split('.csv')[0]
    generic_result_dir = result_dir + data_set_name + "__" + task
    fold_result_dir = generic_result_dir + "_%d%s" % (preprocess_manager.iteration_cross_validation, ".csv")
    result_dir = fold_result_dir

    # start prediction
    with open(result_dir, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(
            ["CaseID", "Prefix length", "Ground truth", "Predicted", "Levenshtein", "Damerau", "Jaccard",
             "Ground truth times", "Predicted times", "MAE", "In time", "Dev. in time", "Num corrections"])

        for line, caseid, times, times2, times3, line_add in zip(lines, caseids, lines_t, lines_t2, lines_t3, lines_add):

            # for each prefix of a case with a size > 1
            for prefix_size in range(2, sequence_max_length):

                num_corrections = 0

                util.ll_print("\nPrefix size: %d\n" % prefix_size)

                # preparation for next best event determination
                # get prefix; one output for each prefix of a case
                current = dict()
                predict = dict()
                ground_truth = dict()

                # current = ground truth prefix + predicted suffix
                current = {
                    "line": ''.join(line[:prefix_size]),
                    "times": times[:prefix_size],
                    "times2": times2[:prefix_size],
                    "times3": times3[:prefix_size],
                    "line_add": line_add[:prefix_size],
                }

                # termination
                if '!' in current["line"]:
                    break

                ground_truth = {
                    "total_event": ''.join(line[:]),
                    "prefix_event": ''.join(line[:prefix_size]),
                    "suffix_event": ''.join(line[prefix_size:]),

                    "total_time": times2[len(times2) - 1],
                    "prefix_time": times2[prefix_size - 1],
                    "suffix_time": times2[len(times2) - 1] - times2[prefix_size - 1]
                }

                predict = {
                    "size": sequence_max_length - 1,
                    "predicted": '',
                    "suffix_time": 0
                }

                # result for each prefix of a case
                if args.next_best_action:

                    # check prefix conformance
                    if preprocess_manager.check_candidate(args, preprocess_manager.transform_new_instance(ground_truth["prefix_event"])):

                        predict, in_time, deviation_in_time, num_corrections = predict_suffix_and_time_for_prefix_next_best_event(
                            args,
                            model_suffix_prediction,
                            preprocess_manager,
                            current, predict,
                            ground_truth,
                            num_corrections)
                    else:
                        break

                else:
                    predict, in_time, deviation_in_time = predict_suffix_and_time_for_prefix(args, model_suffix_prediction,
                                                                                             preprocess_manager, current,
                                                                                             predict, ground_truth)

                # termination
                if predict["predicted"] == "":
                    continue

                output = []
                if len(ground_truth["suffix_event"]) > 0:

                    output.append(caseid)
                    output.append(prefix_size)
                    output.append(str(ground_truth["suffix_event"]).encode("utf-8"))
                    output.append(str(predict["predicted"]).encode("utf-8"))
                    output.append(1 - distance.nlevenshtein(predict["predicted"], ground_truth["suffix_event"]))

                    dls = 1 - (damerau_levenshtein_distance(str(predict["predicted"]),
                                                            str(ground_truth["suffix_event"])) / max(
                        len(predict["predicted"]), len(ground_truth["suffix_event"])))
                    if dls < 0:
                        dls = 0
                    output.append(dls)
                    output.append(1 - distance.jaccard(predict["predicted"], ground_truth["suffix_event"]))
                    output.append(ground_truth["suffix_time"])
                    output.append(predict["suffix_time"])
                    output.append(metrics.mean_absolute_error([ground_truth["suffix_time"]], [predict["suffix_time"]]))
                    output.append(in_time)
                    output.append(deviation_in_time)
                    if num_corrections > 0:
                        output.append(num_corrections)
                    else:
                        output.append(0)

                    spamwriter.writerow(output)
