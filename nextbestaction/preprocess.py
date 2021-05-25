import csv
import copy
import numpy as np

import time
from datetime import datetime
import nextbestaction.util as util
from sklearn.model_selection import KFold
from nextbestaction.dcr_graph import DCRGraph
from nextbestaction.dcr_marking import Marking


class Preprocess_Manager(object):
    dcr = None
    num_folds = 1
    data_dir = ""
    tax_features = False
    ascii_offset = 161
    date_format = "%d.%m.%Y-%H:%M:%S"

    # meta structures
    train_index_per_fold = list()
    test_index_per_fold = list()
    iteration_cross_validation = 0

    chars = list()
    char_indices = dict()
    indices_char = dict()
    target_char_indices = dict()
    target_indices_char = dict()
    divisor = 0
    divisor2 = 0
    divisor3 = 0
    elems_per_fold = 0

    lines = list()
    caseids = list()
    timeseqs = list()
    timeseqs2 = list()
    timeseqs3 = list()
    timeseqs4 = list()
    features_additional_sequences = list()

    num_features_all = 0
    num_features_activities = 0
    num_features_cf = 5  # five additional control-flow features from Tax et al. (2017)
    max_sequence_length = 0
    num_features_additional = 0
    num_attributes_standard = 3  # each event log starts with case id, activity id and timestamp

    # structure for next best event
    X = []
    X_case_based_suffix = []
    X_case_based_suffix_time = []
    X_case_based_suffix_add = []

    avg_time_training_cases = 0

    @classmethod
    def __init__(cls, args):
        cls.num_folds = args.num_folds
        cls.data_dir = args.data_dir + args.data_set
        cls.tax_features = args.tax_features

        util.ll_print("Create structures...")
        lines = []  # these are all the activity seq
        caseids = []
        timeseqs = []  # time sequences (differences between two events that are next to each other)
        timeseqs2 = []  # time sequences (differences between the current and first)
        timeseqs3 = []  # absolute time of previous event
        timeseqs4 = []  # same as timeseqs3
        lastcase = ''
        line = ''
        firstLine = True
        numlines = 0
        times = []
        times2 = []
        times3 = []
        times4 = []
        casestarttime = None
        lasteventtime = None
        check_additional_features = True

        features_additional_attributes = []
        features_additional_events = []
        features_additional_sequences = []

        csvfile = open(cls.data_dir, 'r')
        spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
        next(spamreader, None)

        for row in spamreader:

            if check_additional_features == True:
                if len(row) == cls.num_attributes_standard:
                    util.ll_print("No additional attributes.\n")
                else:
                    cls.num_features_additional = len(row) - cls.num_attributes_standard
                    util.ll_print("Number of additional attributes: %d\n" % cls.num_features_additional)
                check_additional_features = False

            # creates a datetime object from timestamp in row[2]
            t = time.strptime(row[2], cls.date_format)
            # reset list for next case
            if row[0] != lastcase:
                caseids.append(row[0])
                casestarttime = t
                lasteventtime = t
                lastcase = row[0]
                # ensure that each trace consist of min. one event
                if not firstLine:
                    lines.append(line)
                    timeseqs.append(times)
                    timeseqs2.append(times2)
                    timeseqs3.append(times3)
                    timeseqs4.append(times4)
                    if cls.num_features_additional > 0:
                        features_additional_sequences.append(features_additional_events)
                line = ''
                times = []
                times2 = []
                times3 = []
                times4 = []
                if cls.num_features_additional > 0:
                    features_additional_events = []
                numlines += 1

            # get values of additional attributes
            if cls.num_features_additional > 0:
                for index in range(cls.num_attributes_standard,
                                   cls.num_attributes_standard + cls.num_features_additional):
                    features_additional_attributes.append(row[index])
                features_additional_events.append(features_additional_attributes)
                features_additional_attributes = []

            # add activity to a case
            line += chr(int(row[1]) + cls.ascii_offset)

            timesincelastevent = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(
                time.mktime(lasteventtime))
            timesincecasestart = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(
                time.mktime(casestarttime))

            timediff = 86400 * timesincelastevent.days + timesincelastevent.seconds
            timediff2 = 86400 * timesincecasestart.days + timesincecasestart.seconds
            timediff3 = datetime.fromtimestamp(time.mktime(t))
            timediff4 = datetime.fromtimestamp(time.mktime(t))

            times.append(timediff)
            times2.append(timediff2)
            times3.append(timediff3)
            times4.append(timediff4)

            lasteventtime = t
            firstLine = False

        lines.append(line)
        timeseqs.append(times)
        timeseqs2.append(times2)
        timeseqs3.append(times3)
        timeseqs4.append(times4)
        if cls.num_features_additional > 0:
            features_additional_sequences.append(features_additional_events)
        numlines += 1

        util.ll_print("Calculate divisors...\n")
        # average time of current and previous events across all cases
        cls.divisor = np.mean([item for sublist in timeseqs for item in sublist])
        util.ll_print("divisor: %d\n" % cls.divisor)
        # average time between current and first events across all cases
        cls.divisor2 = np.mean([item for sublist in timeseqs2 for item in sublist])
        util.ll_print("divisor2: %d\n" % cls.divisor2)

        seq2_avg = 0
        seq2_temp = list()
        for seq2 in timeseqs2:
            element_avg = 0
            element_temp = list()
            for element in seq2:
                element_temp.append(seq2[len(seq2) - 1] - element)
            element_avg = np.mean(element_temp)
            seq2_temp.append(element_avg)
        seq2_avg = np.mean(seq2_temp)
        cls.divisor3 = seq2_avg

        util.ll_print("divisor3: %d\n" % cls.divisor3)

        # get elements per fold in case of split evaluation
        util.ll_print("Loading Data starts... \n")
        # if args.cross_validation == False:
        cls.elems_per_fold = int(round(numlines / cls.num_folds))

        util.ll_print("Calc max length of sequence\n")
        lines = list(map(lambda x: x + '!', lines))
        cls.max_sequence_length = max(map(lambda x: len(x), lines))
        util.ll_print("Max length of sequence: %d\n" % cls.max_sequence_length)

        util.ll_print("Start calculation of total chars and total target chars... \n")
        cls.chars = list(map(lambda x: set(x), lines))
        cls.chars = list(set().union(*cls.chars))
        cls.chars.sort()
        cls.target_chars = copy.copy(cls.chars)
        cls.chars.remove('!')
        util.ll_print("Total chars: %d, target chars: %d\n" % (len(cls.chars), len(cls.target_chars)))

        util.ll_print("Start creation of dicts for char handling... \n")
        cls.char_indices = dict((c, i) for i, c in enumerate(cls.chars))
        cls.indices_char = dict((i, c) for i, c in enumerate(cls.chars))
        cls.target_char_indices = dict((c, i) for i, c in enumerate(cls.target_chars))
        cls.target_indices_char = dict((i, c) for i, c in enumerate(cls.target_chars))
        util.ll_print("Dics for char handling created\n")

        # if tax features should not be considered
        if not cls.tax_features:
            cls.num_features_cf = 0

        # set feature variables
        cls.num_features_activities = len(cls.chars)
        cls.num_features_all = cls.num_features_activities + cls.num_features_cf + cls.num_features_additional

        # set structure variables
        cls.lines = lines
        cls.caseids = caseids
        cls.timeseqs = timeseqs
        cls.timeseqs2 = timeseqs2
        cls.timeseqs3 = timeseqs3
        cls.timeseqs4 = timeseqs4
        if cls.num_features_additional > 0:
            cls.features_additional_sequences = features_additional_sequences

        kFold = KFold(n_splits=cls.num_folds, random_state=0, shuffle=True)
        for train_index, test_index in kFold.split(lines):
            cls.train_index_per_fold.append(train_index)
            cls.test_index_per_fold.append(test_index)

    @classmethod
    def create_and_encode_training_set(cls, args):

        # 1) Create create folds for training set
        lines = list()
        lines_t = list()
        lines_t2 = list()
        lines_t3 = list()
        lines_t4 = list()
        lines_add = list()

        # get lines by train index
        for index, value in enumerate(cls.train_index_per_fold[cls.iteration_cross_validation]):
            lines.append(cls.lines[value])
            lines_t.append(cls.timeseqs[value])
            lines_t2.append(cls.timeseqs2[value])
            lines_t3.append(cls.timeseqs3[value])
            lines_t4.append(cls.timeseqs4[value])
            if cls.num_features_additional > 0:
                lines_add.append(cls.features_additional_sequences[value])

        # calculate average time of training cases
        for index in range(len(lines_t2)):
            cls.avg_time_training_cases = cls.avg_time_training_cases + lines_t2[index][len(lines_t2[index]) - 1]
        cls.avg_time_training_cases = cls.avg_time_training_cases / len(lines_t2)

        # 2) Create training set
        step = 1
        sentences = []
        next_chars = []

        sentences_t = []
        sentences_t2 = []
        sentences_t3 = []
        sentences_t4 = []
        if cls.num_features_additional > 0:
            sentences_add = []
        next_chars_t = []
        next_chars_t2 = []
        next_chars_t3 = []
        next_chars_t4 = []

        if cls.num_features_additional > 0:
            for line, line_t, line_t2, line_t3, line_t4, line_add in zip(lines, lines_t, lines_t2, lines_t3, lines_t4,
                                                                         lines_add):
                # i = number of activities
                # step = difference between each number in the sequence
                for i in range(0, len(line), step):
                    if i == 0:
                        continue
                    # add iteratively, first symbol of the line, then two first, three
                    sentences.append(line[0:i])
                    sentences_t.append(line_t[0:i])
                    sentences_t2.append(line_t2[0:i])
                    sentences_t3.append(line_t3[0:i])
                    sentences_t4.append(line_t4[0:i])
                    sentences_add.append(line_add[0:i])
                    # next activity
                    next_chars.append(line[i])

                    if i == len(line) - 1:  # special case to deal time of end character
                        next_chars_t.append(0)
                        next_chars_t2.append(0)
                        next_chars_t3.append(0)
                        next_chars_t4.append(0)
                    else:
                        next_chars_t.append(line_t[i])
                        next_chars_t2.append(line_t2[i])
                        next_chars_t3.append(line_t3[i])
                        next_chars_t4.append(line_t4[i])

            util.ll_print("\n nb sequences: %d" % len(sentences))
            util.ll_print("\n add sequences: %d" % len(sentences_add))
        else:
            for line, line_t, line_t2, line_t3, line_t4 in zip(lines, lines_t, lines_t2, lines_t3, lines_t4):
                for i in range(0, len(line), step):
                    if i == 0:
                        continue
                    sentences.append(line[0: i])
                    sentences_t.append(line_t[0:i])
                    sentences_t2.append(line_t2[0:i])
                    sentences_t3.append(line_t3[0:i])
                    sentences_t4.append(line_t4[0:i])
                    next_chars.append(line[i])

                    if i == len(line) - 1:  # special case to deal time of end character
                        next_chars_t.append(0)
                        next_chars_t2.append(0)
                        next_chars_t3.append(0)
                        next_chars_t4.append(0)
                    else:
                        next_chars_t.append(line_t[i])
                        next_chars_t2.append(line_t2[i])
                        next_chars_t3.append(line_t3[i])
                        next_chars_t4.append(line_t4[i])

            util.ll_print("\nnb sequences: %d" % len(sentences))

        # create numpy array
        X = np.zeros((len(sentences), cls.max_sequence_length, cls.num_features_all), dtype=np.float64)
        Y_Act = np.zeros((len(sentences), len(cls.target_chars)), dtype=np.float64)
        Y_Time = np.zeros((len(sentences)), dtype=np.float32)

        for i, sentence in enumerate(sentences):

            leftpad = cls.max_sequence_length - len(sentence)
            next_t = next_chars_t[i]
            times = sentences_t[i]
            times2 = sentences_t2[i]
            times3 = sentences_t3[i]
            times4 = sentences_t4[i]
            if cls.num_features_additional > 0:
                sentence_add = sentences_add[i]

            # set data
            for t, char in enumerate(sentence):
                midnight = times3[t].replace(hour=0, minute=0, second=0, microsecond=0)
                timesincemidnight = times3[t] - midnight

                for c in cls.chars:
                    if c == char:
                        X[i, t + leftpad, cls.char_indices[c]] = 1.0

                if cls.tax_features:
                    X[i, t + leftpad, len(cls.chars)] = t + 1
                    X[i, t + leftpad, len(cls.chars) + 1] = times[t]
                    X[i, t + leftpad, len(cls.chars) + 2] = times2[t]
                    X[i, t + leftpad, len(cls.chars) + 3] = timesincemidnight.seconds
                    X[i, t + leftpad, len(cls.chars) + 4] = times4[t].weekday()

                    # add additional attributes
                    if cls.num_features_all > 0:
                        for x in range(0, cls.num_features_additional):
                            X[i, t + leftpad, len(cls.chars) + (cls.num_features_cf + x)] = sentence_add[t][x]
                else:
                    # add additional attributes
                    if cls.num_features_all > 0:
                        for x in range(0, cls.num_features_additional):
                            X[i, t + leftpad, len(cls.chars) + (cls.num_features_cf + x)] = sentence_add[t][x]

            # set label
            for c in cls.target_chars:
                if c == next_chars[i]:
                    Y_Act[i, cls.target_char_indices[c]] = 1
                else:
                    Y_Act[i, cls.target_char_indices[c]] = 0

            Y_Time[i] = next_t / cls.divisor

        num_features_all = cls.num_features_all
        num_features_activities = cls.num_features_activities

        cls.X = X

        return X, Y_Act, Y_Time, cls.max_sequence_length, num_features_all, num_features_activities

    @classmethod
    def create_test_set(self):
        lines = list()
        caseids = list()
        lines_t = list()
        lines_t2 = list()
        lines_t3 = list()
        lines_add = list()

        for index, value in enumerate(self.test_index_per_fold[self.iteration_cross_validation]):
            lines.append(self.lines[value])
            caseids.append(self.caseids[value])
            lines_t.append(self.timeseqs[value])
            lines_t2.append(self.timeseqs2[value])
            lines_t3.append(self.timeseqs3[value])
            if self.num_features_additional > 0:
                lines_add.append(self.features_additional_sequences[value])

        if self.num_features_additional > 0:
            return lines, caseids, lines_t, lines_t2, lines_t3, lines_add, self.max_sequence_length, self.num_features_all, self.num_features_activities
        else:
            return lines, caseids, lines_t, lines_t2, lines_t3, self.max_sequence_length, self.num_features_all, self.num_features_activities

    @classmethod
    def encode_test_set(cls, sentence, times, times3, batch_size):

        X = np.zeros((batch_size, cls.max_sequence_length, cls.num_features_all), dtype=np.float32)
        leftpad = cls.max_sequence_length - len(sentence)
        times2 = np.cumsum(times)

        for t, char in enumerate(sentence):
            midnight = times3[t].replace(hour=0, minute=0, second=0, microsecond=0)
            timesincemidnight = times3[t] - midnight

            for c in cls.chars:
                if c == char:
                    X[0, t + leftpad, cls.char_indices[c]] = 1.0

            if cls.tax_features:
                X[batch_size - 1, t + leftpad, len(cls.chars)] = t + 1  # id of event within a case
                X[batch_size - 1, t + leftpad, len(cls.chars) + 1] = times[t]  # attribute two: the average time difference between the current and the previous event of a process instance
                X[batch_size - 1, t + leftpad, len(cls.chars) + 2] = times2[t]  # attribute three: the average time difference between the current and the first event of a process instance
                X[batch_size - 1, t + leftpad, len(cls.chars) + 3] = timesincemidnight.seconds  # attribute four: the time difference between  current  event  and  midnight
                X[batch_size - 1, t + leftpad, len(cls.chars) + 4] = times3[t].weekday()  # attribute five: the  time  difference  between  current event  and  the  beginning  of  week
        return X

    @classmethod
    def encode_test_set_add(self, sentence, times, times3, sentence_add, batch_size):

        X = np.zeros((1, self.max_sequence_length, self.num_features_all), dtype=np.float32)
        leftpad = self.max_sequence_length - len(sentence)
        times2 = np.cumsum(times)

        for t, char in enumerate(sentence):
            midnight = times3[t].replace(hour=0, minute=0, second=0, microsecond=0)
            timesincemidnight = times3[t] - midnight

            for c in self.chars:
                if c == char:
                    X[0, t + leftpad, self.char_indices[c]] = 1.0

            if self.tax_features:
                X[batch_size - 1, t + leftpad, len(self.chars)] = t + 1
                X[batch_size - 1, t + leftpad, len(self.chars) + 1] = times[t]
                X[batch_size - 1, t + leftpad, len(self.chars) + 2] = times2[t]
                X[batch_size - 1, t + leftpad, len(self.chars) + 3] = timesincemidnight.seconds
                X[batch_size - 1, t + leftpad, len(self.chars) + 4] = times3[t].weekday()

                for x in range(0, self.num_features_additional):
                    X[batch_size - 1, t + leftpad, len(self.chars) + (self.num_features_cf + x)] = \
                        sentence_add[t][x]

            else:
                for x in range(0, self.num_features_additional):
                    X[batch_size - 1, t + leftpad, len(self.chars) + (self.tax_features + x)] = sentence_add[t][x]

        num_features_all = self.num_features_all
        num_features_activities = self.num_features_activities

        return X, num_features_all, num_features_activities

    @classmethod
    def get_symbol(cls, predictions):
        maxPrediction = 0
        symbol = ''
        i = 0
        for prediction in predictions:
            if prediction >= maxPrediction:
                maxPrediction = prediction
                symbol = cls.target_indices_char[i]
            i += 1
        return symbol

    @classmethod
    def transform_tensor_to_matrix(cls, X):

        number_cases = cls.find_number_of_cases_in_tensor(X)

        case_id = 0
        X_case_based = np.zeros(
            (number_cases, cls.max_sequence_length + cls.max_sequence_length * cls.num_features_additional),
            dtype=np.float32)
        case = list()
        case_time = list()
        cases_time = list()
        cases_time_suffix = list()
        case_add = list()
        cases_add_suffix = list()

        # add n-1 cases
        for index in range(X.shape[0]):

            # index > 0 -> at the second case look back to the first case
            if index > 0 and X[index, cls.max_sequence_length - 1, cls.num_features_activities] == 1:

                # iterate over all time steps of the case by event id of the case
                # assumption: the latest time step is at the end
                case_time_steps = int(X[index - 1, cls.max_sequence_length - 1, cls.num_features_activities])

                for row_index in range(case_time_steps):
                    for column_index in range(cls.num_features_all):

                        # event
                        if column_index < cls.num_features_activities:
                            if X[index - 1, cls.max_sequence_length - case_time_steps + row_index, column_index] == 1:
                                # + 1 since index at 0; will be later corrected
                                case.append(column_index + 1)
                                case_time.append(X[
                                                     index - 1, cls.max_sequence_length - case_time_steps + row_index, cls.num_features_activities + 2])

                        # context attributes
                        if column_index >= (cls.num_features_all - cls.num_features_additional):
                            case.append(
                                X[index - 1, cls.max_sequence_length - case_time_steps + row_index, column_index])

                # update of structure
                for case_index in range(len(case)):
                    X_case_based[case_id, case_index] = case[case_index]
                cases_time.append(case_time)
                case_time = []
                case = []
                case_id = case_id + 1

        # add last case
        case_time_steps = int(X[-1, cls.max_sequence_length - 1, cls.num_features_activities])

        for row_index in range(case_time_steps):
            for column_index in range(cls.num_features_all):

                # event
                if column_index < cls.num_features_activities:
                    if X[-1, cls.max_sequence_length - case_time_steps + row_index, column_index] == 1:
                        # +1 since index at 0
                        case.append(column_index + 1)
                        case_time.append(X[
                                             index - 1, cls.max_sequence_length - case_time_steps + row_index, cls.num_features_activities + 2])

                # context attributes
                if column_index >= (cls.num_features_all - cls.num_features_additional):
                    case.append(X[-1, cls.max_sequence_length - case_time_steps + row_index, column_index])

        # update of structure with last case
        for case_index in range(0, len(case)):
            X_case_based[case_id, case_index] = case[case_index]
        cases_time.append(case_time)
        case_time = []
        print(X_case_based)

        # get number of suffixes
        number_suffixes = 0
        for index_case in range(0, len(X_case_based)):
            for index_suffix in range(0, len(cases_time[index_case])):  # -1
                number_suffixes = number_suffixes + 1

        # create suffixes
        # note self.max_sequence_length -1 = predict_size from testing script
        X_case_based_suffix = np.zeros((number_suffixes, cls.max_sequence_length - 1 + (
                cls.max_sequence_length - 1) * cls.num_features_additional), dtype=np.float32)
        number_suffixes = 0

        for index_case in range(0, len(X_case_based)):
            length_suffix = len(cases_time[index_case]) + len(cases_time[index_case]) * cls.num_features_additional

            # pick out max. suffix
            case_suffix = X_case_based[index_case][:length_suffix]
            print(case_suffix)

            for suffix_index in range(0, len(case_suffix), cls.num_features_additional + 1):

                index_suffix_one_step_time = 0
                for index_suffix_pos in range(0, suffix_index + 1, cls.num_features_additional + 1):

                    try:
                        print("%s-%s" % (index_suffix_pos, suffix_index))
                        # update of event
                        X_case_based_suffix[number_suffixes, index_suffix_pos] = X_case_based[
                            index_case, index_suffix_pos]

                        # update of attributes
                        case_add_pos = []
                        for index_attribute in range(0, cls.num_features_additional):
                            X_case_based_suffix[number_suffixes, index_suffix_pos + index_attribute + 1] = X_case_based[
                                index_case, index_suffix_pos + index_attribute + 1]
                            case_add_pos.append(X_case_based[index_case, index_suffix_pos + index_attribute + 1])
                        case_add.append(case_add_pos)

                        # update of time
                        case_time.append(cases_time[index_case][index_suffix_one_step_time])
                        index_suffix_one_step_time = index_suffix_one_step_time + 1
                    except ValueError:
                        pass

                number_suffixes = number_suffixes + 1
                cases_time_suffix.append(case_time)
                cases_add_suffix.append(case_add)
                case_time = []
                case_add = []

        cls.X_case_based_suffix = X_case_based_suffix
        cls.X_case_based_suffix_time = cases_time_suffix
        cls.X_case_based_suffix_add = cases_add_suffix

        util.ll_print(str(len(cls.X_case_based_suffix)))
        util.ll_print(str(len(cls.X_case_based_suffix_time)))
        util.ll_print(str(len(cls.X_case_based_suffix_add)))

        return X_case_based_suffix

    @classmethod
    def find_number_of_cases_in_tensor(self, X):
        number_cases = 0
        for index in range(X.shape[0]):
            # index > 0 -> check retrospective the cases during the iteration
            if index > 0 and X[index, self.max_sequence_length - 1, self.num_features_activities] == 1:
                # iterate over n-1 cases
                number_cases = number_cases + 1

        # add case n
        return number_cases + 1

    # creates the input for the candidate selection
    # note current includes the prefix and the suffix
    @classmethod
    def transform_current_instance_to_suffix_vector(cls, current, prefix_length, predict_size):

        vector_length = len(current["line"]) + len(current["line"]) * cls.num_features_additional
        vector_case_based = np.zeros(
            (cls.max_sequence_length + cls.max_sequence_length * cls.num_features_additional,), dtype=np.float32)
        vector_case_based_suffix = []
        vector_case_based_suffix_final = np.zeros((predict_size + predict_size * cls.num_features_additional,),
                                                  dtype=np.float32)
        events = current["line"]
        attributes = current["line_add"]

        # change representation
        index_event = 0
        for index in range(0, vector_length, cls.num_features_additional + 1):

            # set event
            # note, +1, as in the training script since column index begins with 0
            vector_case_based[index] = cls.target_char_indices[events[index_event]] + 1

            # set attributes
            for index_attributes in range(0, cls.num_features_additional):
                vector_case_based[index + index_attributes + 1] = attributes[index_event][index_attributes]

            index_event = index_event + 1

        vector_case_based = vector_case_based.reshape(1, -1)

        # select suffix     
        vector_case_based_suffix = vector_case_based[0][prefix_length + prefix_length * cls.num_features_additional:]
        for index in range(0, len(vector_case_based_suffix)):
            vector_case_based_suffix_final[index] = vector_case_based_suffix[index]

        return vector_case_based_suffix_final.reshape(1, -1)

    @classmethod
    def check_candidate(cls, args, new_instance):

        if args.semantic_check:
            if cls.dcr is None:
                cls.dcr = DCRGraph(args.data_dir + args.dcr_path)
            dcr = cls.dcr
            marking = Marking.get_initial_marking()
            for event in new_instance:
                node = dcr.get_node_by_name(str(event))
                if not marking.perform_transition_node(node):
                    return False

            if len(marking.PendingResponse) != 0:
                for pending in marking.PendingResponse:
                    if pending in marking.Included:
                        return False
            return True
        else:
            return True

    @classmethod
    def transform_new_instance(cls, new_instance):
        return [cls.target_char_indices[element] for element in new_instance]

    @classmethod
    def create_new_instance(cls, args, prefix, suffix):

        suffix_no_context = ""
        for index in range(0, len(suffix), 2):
            if suffix[index] > 0:
                suffix_no_context = suffix_no_context + cls.target_indices_char[suffix[index] - 1]
            else:
                break
                # first padded zero of the vector
        new_instance = prefix + suffix_no_context

        return [cls.target_char_indices[element] for element in new_instance]

    @classmethod
    def select_best_candidate_from_training_set(cls, candidates, prefix, args):

        X_case_based_suffix_temp = cls.X_case_based_suffix
        X_case_based_suffix_time_temp = cls.X_case_based_suffix_time
        X_case_based_suffix_add_temp = cls.X_case_based_suffix_add

        current_candidate = []
        current_candidate_event_time = []
        current_candidate_total_time = []
        current_candidate_add = []
        current_candidate_event = []

        # select suffix candidate with minimum total suffix time out of k suffix candidates
        for candidate in candidates[0]:

            new_instance = cls.create_new_instance(args, prefix, X_case_based_suffix_temp[candidate])
            if cls.check_candidate(args, new_instance):

                if current_candidate == []:
                    current_candidate = X_case_based_suffix_temp[candidate]
                    current_candidate_event_time = X_case_based_suffix_time_temp[candidate][0]
                    current_candidate_total_time = sum(X_case_based_suffix_time_temp[candidate])
                    current_candidate_add = X_case_based_suffix_add_temp[candidate][0]
                    # Note: before, we added 1 to the column index to differ between no value and first element of dict 'target_indices_char'
                    current_candidate_event = cls.target_indices_char[current_candidate[0] - 1]

                else:
                    if sum(X_case_based_suffix_time_temp[candidate]) < current_candidate_total_time:
                        current_candidate = X_case_based_suffix_temp[candidate]
                        current_candidate_event_time = X_case_based_suffix_time_temp[candidate][0]
                        current_candidate_total_time = sum(X_case_based_suffix_time_temp[candidate])
                        current_candidate_add = X_case_based_suffix_add_temp[candidate][0]
                        # Note: before, we added 1 to the column index to differ between no value and first element of dict 'target_indices_char'
                        current_candidate_event = cls.target_indices_char[current_candidate[0] - 1]

        return current_candidate_total_time, current_candidate_event_time, current_candidate_add, current_candidate_event, current_candidate
