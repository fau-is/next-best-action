from tensorflow.keras.models import load_model
import csv
import distance
from jellyfish._jellyfish import damerau_levenshtein_distance
from datetime import timedelta
from sklearn import metrics
import pickle
import copy


def predict_suffix_and_time_for_prefix(model, preprocess_manager, current, predict, ground_truth):
    in_time = 1
    start = len(ground_truth["prefix_event"])

    for i in range(start, predict["size"]):

        # Get prediction
        X, num_features_all, num_features_activities = preprocess_manager.encode_test_set(current["line"], current["times"], current["times3"])
        Y = model.predict(X, verbose=0)
        y_event = Y[0][0]
        y_time = Y[1][0][0]
        prediction = preprocess_manager.get_symbol(y_event)

        # Update of prefix (event)
        current["line"] += prediction
        predict["predicted"] += prediction

        # Update of prefix for suffix prediction (time + context)
        if prediction == '!':
            y_time = 0
            current["times"].append(y_time)
            current["times3"].append(current["times3"][-1] + timedelta(seconds=y_time))
            predict["suffix_time"] = predict["suffix_time"] + y_time
        else:
            # update of prefix for suffix prediction
            if y_time < 0:
                y_time = 0
            y_time = y_time * preprocess_manager.divisor3

            current["times"].append(y_time)
            current["times3"].append(current["times3"][-1] + timedelta(seconds=y_time))
            predict["suffix_time"] = predict["suffix_time"] + y_time

        # termination; predict size = max sequence length - 1
        if prediction == '!' or len(current["line"]) == predict["size"]:
            print('! termination suffix prediction ... ')
            break

        print("Prefix+Suffix-Time-%s: %f" % (i, ground_truth["prefix_time"] + predict["suffix_time"]))
        print("Baseline-%s: %f" % (i, preprocess_manager.avg_time_training_cases))

    if ground_truth["prefix_time"] + predict["suffix_time"] > preprocess_manager.avg_time_training_cases:
        in_time = 0
    deviation_in_time = (ground_truth["prefix_time"] + predict["suffix_time"]) / preprocess_manager.avg_time_training_cases

    return predict, in_time, deviation_in_time


def predict_suffix_and_time_for_prefix_next_best_event(args, model, preprocess_manager, current_temp, predict, ground_truth):

    # 0.) Store initial prefix
    current = copy.deepcopy(current_temp)
    current_with_next_best_event = copy.deepcopy(current)
    predict_initial = copy.deepcopy(predict)
    in_time = 1
    deviation_in_time = 0
    num_interventions = 0

    start = len(current_with_next_best_event["line"])
    for j in range(start, predict["size"]):

        # 1.) Suffix + time for a given prefix
        # Note: upper bound is max predict size and not ground truth size
        start_sub = start
        first_prediction = True
        current_first_prediction = dict()

        for i in range(start_sub, predict["size"]):

            # Get prediction
            X, num_features_all, num_features_activities = preprocess_manager.encode_test_set(current["line"], current["times"], current["times3"]) # preprocess prefix
            Y = model.predict(X, verbose=0)  # make prediction
            y_act = Y[0][0]  # get output of predictor output
            y_time = Y[1][0][0]
            prediction = preprocess_manager.get_symbol(y_act)

            # Update of prefix (event)
            current["line"] += prediction
            predict["predicted"] += prediction

            # Update of prefix for suffix prediction
            if prediction == '!':
                y_time = 0
                current["times"].append(y_time)
                current["times3"].append(current["times3"][-1] + timedelta(seconds=y_time))
                predict["suffix_time"] = predict["suffix_time"] + y_time
            else:
                # Update of prefix for suffix prediction (time + context)
                if y_time < 0:
                    y_time = 0
                y_time = y_time * preprocess_manager.divisor3
                current["times"].append(y_time)
                current["times3"].append(current["times3"][-1] + timedelta(seconds=y_time))
                predict["suffix_time"] = predict["suffix_time"] + y_time

            # save first prediction
            if first_prediction:
                current_first_prediction = copy.deepcopy(current)
                first_prediction = False

            if prediction == '!' or len(current["line"]) == predict["size"]:  # termination
                print('\n! Termination suffix prediction ############################################# \n')
                break

        print("Prefix+Suffix-Time-%s-%s: %f" % (j, i, (current_with_next_best_event["times3"][-1] - current_with_next_best_event["times3"][0]).total_seconds() + predict["suffix_time"]))
        print("\nBaseline-%s-%s: %f" % (j, i, preprocess_manager.avg_time_training_cases))

        # 2.) Update of next best action ('current_with_next_best_event')
        if (current_with_next_best_event["times3"][-1] - current_with_next_best_event["times3"][0]).total_seconds() + \
                predict["suffix_time"] <= preprocess_manager.avg_time_training_cases:

            print("\nDo not determine next best action ...")
            current_with_next_best_event = copy.deepcopy(current_first_prediction)

        else:
            print("\nDetermine next best action ...")

            num_interventions += 1

            # 2.1) select candidates
            # Input params: current; length of current prefix
            model_candidate_selection = pickle.load(open("%smodel_candidate_selection" % args.checkpoint_dir, 'rb'))

            current_vector = preprocess_manager.transform_current_instance_to_suffix_vector(current, len(
                current_with_next_best_event["line"]), predict["size"])
            distance_candidates, index_candidates = model_candidate_selection.query(current_vector, k=args.num_candidates)

            # 2.2) Get best candidate from training set with similar suffix
            # Input params: index of candidates; length of current prefix
            current_candidate_total_time, current_candidate_event_time, current_candidate_event, current_candidate = preprocess_manager.select_best_candidate_from_training_set(
                index_candidates, current_with_next_best_event["line"], args)

            if current_candidate == []:
                # 2.3.1) If no candidate is conform to the DCR graphs
                current_with_next_best_event = copy.deepcopy(current_first_prediction)
            else:
                # 2.3.2) Update of next likely event through next best event
                current_with_next_best_event["line"] += current_candidate_event
                current_with_next_best_event["times"].append(current_candidate_event_time)
                current_with_next_best_event["times3"].append(current_with_next_best_event["times3"][-1] + timedelta(seconds=int(current_candidate_event_time)))

        # 3.) Termination
        if len(current_with_next_best_event["line"]) == predict["size"] or '!' in current_with_next_best_event["line"]:
            print('\n! Termination prefix ... \n')
            break

        # 4.) Update
        current = copy.deepcopy(current_with_next_best_event)  # Update of current with next best event
        predict = copy.deepcopy(predict_initial)  # Update of predict with init predict

    # 5.) Final check
    if (current_with_next_best_event["times3"][-1] - current_with_next_best_event["times3"][0]).total_seconds() > \
            preprocess_manager.avg_time_training_cases:
        in_time = 0

    # 6.) Calculate deviation
    deviation_in_time = (current_with_next_best_event["times3"][-1] - current_with_next_best_event["times3"][0]).total_seconds() / \
                        preprocess_manager.avg_time_training_cases

    # 7.) Construct predict object for output
    predict["predicted"] = current_with_next_best_event["line"][len(ground_truth["prefix_event"]):]
    predict["suffix_time"] = sum(current_with_next_best_event["times"][len(ground_truth["prefix_event"]):])

    return predict, in_time, deviation_in_time, num_interventions


def test(args, preprocess_manager):

    # get test set
    lines, case_ids, lines_t, lines_t2, lines_t3, seq_max_length, num_features_all, num_features_activities = preprocess_manager.create_test_set()
    model_suffix_prediction = load_model('%smodel_suffix_prediction.h5' % args.checkpoint_dir)  # load model

    data_set_name = args.data_set.split('.csv')[0]   # set options for result output
    generic_result_dir = args.result_dir + data_set_name + "_" + args.task
    result_dir = generic_result_dir + ".csv"

    # start prediction
    with open(result_dir, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(
            ["CaseID", "Prefix length", "Ground truth", "Predicted", "Levenshtein", "Damerau", "Jaccard",
             "Ground truth times", "Predicted times", "MAE", "In time", "Dev. in time", "Num interventions"])

        for line, case_id, times, times2, times3 in zip(lines, case_ids, lines_t, lines_t2, lines_t3):
            for prefix_size in range(2, seq_max_length):  # size > 1

                print("\nPrefix size: %d" % prefix_size)

                # current = ground truth prefix + predicted suffix
                current = {
                    "line": ''.join(line[:prefix_size]),
                    "times": times[:prefix_size],
                    "times2": times2[:prefix_size],
                    "times3": times3[:prefix_size]
                }

                if '!' in current["line"]:  # termination
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
                    "size": seq_max_length - 1,
                    "predicted": '',
                    "suffix_time": 0
                }

                if args.next_best_action:
                    # Check prefix conformance
                    if preprocess_manager.check_candidate(args, preprocess_manager.transform_new_instance(ground_truth["prefix_event"])):
                        predict, in_time, deviation_in_time, num_interventions = predict_suffix_and_time_for_prefix_next_best_event(
                            args, model_suffix_prediction, preprocess_manager, current, predict, ground_truth)
                    else:
                        break
                else:
                    # Check prefix conformance
                    if preprocess_manager.check_candidate(args, preprocess_manager.transform_new_instance(ground_truth["prefix_event"])):
                        predict, in_time, deviation_in_time = predict_suffix_and_time_for_prefix(model_suffix_prediction,
                                                                                                 preprocess_manager, current,
                                                                                                 predict, ground_truth)
                    else:
                        break

                if predict["predicted"] == "":  # termination
                    continue

                output = []
                if len(ground_truth["suffix_event"]) > 0:

                    output.append(case_id)
                    output.append(prefix_size)
                    output.append(str(ground_truth["suffix_event"]).encode("utf-8"))
                    output.append(str(predict["predicted"]).encode("utf-8"))
                    output.append(1 - distance.nlevenshtein(predict["predicted"], ground_truth["suffix_event"]))

                    dls = 1 - (damerau_levenshtein_distance(str(predict["predicted"]), str(ground_truth["suffix_event"])) / max(
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
                    if args.next_best_action:
                        output.append(num_interventions)
                    else:
                        output.append(0)
                    spamwriter.writerow(output)
