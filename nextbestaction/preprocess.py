import csv
import copy
import numpy as np
import time
from datetime import datetime
from nextbestaction.dcr_graph import DCRGraph
from nextbestaction.dcr_marking import Marking


class Preprocess_Manager(object):
    dcr = None
    data_dir = ""
    ascii_offset = 161
    date_format = "%Y-%m-%d %H:%M:%S"

    train_indices, test_indices = [], []

    chars = []
    char_indices = dict()
    indices_char = dict()
    target_char_indices = dict()
    target_indices_char = dict()
    divisor = 0
    divisor2 = 0
    divisor3 = 0
    lines = []
    caseids = []
    timeseqs, timeseqs2, timeseqs3, timeseqs4 = [], [], [], []

    num_features_all = 0
    num_features_activities = 0
    num_features_cf = 5  # 5 features from Tax et al. (2017)
    max_seq_length = 0

    X = []  # structure for next best event
    X_case_based_suffix = []
    X_case_based_suffix_time = []
    avg_time_training_cases = 0

    @classmethod
    def __init__(cls, args):

        cls.data_dir = args.data_dir + args.data_set
        cls.tax_features = args.tax_features

        print("Create structures...\n")
        lines = []  # these are all the activity seq
        caseids = []
        timeseqs = []  # time sequences (differences between two events that are next to each other)
        timeseqs2 = []  # time sequences (differences between the current and first)
        timeseqs3 = []  # absolute time of previous event
        timeseqs4 = []  # same as timeseqs3
        lastcase = ''
        line = ''
        firstLine = True
        times = []
        times2 = []
        times3 = []
        times4 = []
        casestarttime = None
        lasteventtime = None

        csvfile = open(cls.data_dir, 'r')
        spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
        next(spamreader, None)

        for row in spamreader:
            t = time.strptime(row[2], cls.date_format)  # creates a datetime object from timestamp in row[2]
            if row[0] != lastcase:  # reset list for next case
                caseids.append(row[0])
                casestarttime = t
                lasteventtime = t
                lastcase = row[0]
                if not firstLine:  # ensure that each trace consist of min one event
                    lines.append(line)
                    timeseqs.append(times)
                    timeseqs2.append(times2)
                    timeseqs3.append(times3)
                    timeseqs4.append(times4)
                line = ''
                times = []
                times2 = []
                times3 = []
                times4 = []

            line += chr(int(row[1]) + cls.ascii_offset)  # add activity to a case
            timesincelastevent = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(lasteventtime))
            timesincecasestart = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(casestarttime))
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

        print("Calculate divisors...\n")
        cls.divisor = np.mean([item for sublist in timeseqs for item in sublist])  # average time of current and previous events across all cases
        print("Divisor: %d\n" % cls.divisor)
        cls.divisor2 = np.mean([item for sublist in timeseqs2 for item in sublist])  # average time between current and first events across all cases
        print("Divisor2: %d\n" % cls.divisor2)

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

        print("Divisor3: %d\n" % cls.divisor3)

        lines = list(map(lambda x: x + '!', lines))  # calculate max length of sequence
        cls.max_seq_length = max(map(lambda x: len(x), lines))

        print("Start calculation of total chars and total target chars... \n")

        cls.chars = list(map(lambda x: set(x), lines))
        cls.chars = list(set().union(*cls.chars))
        cls.chars.sort()
        cls.target_chars = copy.copy(cls.chars)
        cls.chars.remove('!')

        print("Total chars: %d, target chars: %d\n" % (len(cls.chars), len(cls.target_chars)))
        print("Start creation of dicts for char handling... \n")

        cls.char_indices = dict((c, i) for i, c in enumerate(cls.chars))
        cls.indices_char = dict((i, c) for i, c in enumerate(cls.chars))
        cls.target_char_indices = dict((c, i) for i, c in enumerate(cls.target_chars))
        cls.target_indices_char = dict((i, c) for i, c in enumerate(cls.target_chars))

        if not cls.tax_features:
            cls.num_features_cf = 0

        cls.lines = lines  # set structure variables
        cls.caseids = caseids
        cls.timeseqs = timeseqs
        cls.timeseqs2 = timeseqs2
        cls.timeseqs3 = timeseqs3
        cls.timeseqs4 = timeseqs4

        cls.num_features_activities = len(cls.chars)   # set feature variables
        cls.num_features_all = cls.num_features_activities + cls.num_features_cf

        indices = list(range(len(lines)))
        cls.train_indices = indices[:int(len(indices) * args.train_size)]
        cls.test_indices = indices[int(len(indices) * args.train_size):]

    @classmethod
    def create_and_encode_training_set(cls):

        # 1) Create lists for training set
        lines_a, lines_t, lines_t2, lines_t3, lines_t4 = [], [], [], [], []

        for index in range(0, len(cls.train_indices)):
            lines_a.append(cls.lines[index])
            lines_t.append(cls.timeseqs[index])
            lines_t2.append(cls.timeseqs2[index])
            lines_t3.append(cls.timeseqs3[index])
            lines_t4.append(cls.timeseqs4[index])

        # calculate average time of training cases
        for index in range(len(lines_t2)):
            cls.avg_time_training_cases = cls.avg_time_training_cases + lines_t2[index][len(lines_t2[index]) - 1]  # sum total times across cases
        cls.avg_time_training_cases = cls.avg_time_training_cases / len(lines_t2)

        # 2) Create training set
        prefixes_a = []
        prefixes_t = []
        prefixes_t2 = []
        prefixes_t3 = []
        prefixes_t4 = []
        next_events_a = []
        next_events_t = []
        next_events_t2 = []
        next_events_t3 = []
        next_events_t4 = []

        for line_a, line_t, line_t2, line_t3, line_t4 in zip(lines_a, lines_t, lines_t2, lines_t3, lines_t4):
            for i in range(0, len(line_a), 1):
                if i == 0:
                    continue
                prefixes_a.append(line_a[0: i])
                prefixes_t.append(line_t[0:i])
                prefixes_t2.append(line_t2[0:i])
                prefixes_t3.append(line_t3[0:i])
                prefixes_t4.append(line_t4[0:i])
                next_events_a.append(line_a[i])

                if i == len(line_a) - 1:  # special case to deal time of end character
                    next_events_t.append(0)
                    next_events_t2.append(0)
                    next_events_t3.append(0)
                    next_events_t4.append(0)
                else:
                    next_events_t.append(line_t[i])
                    next_events_t2.append(line_t2[i])
                    next_events_t3.append(line_t3[i])
                    next_events_t4.append(line_t4[i])

        X = np.zeros((len(prefixes_a), cls.max_seq_length, cls.num_features_all), dtype=np.float64)
        Y_Act = np.zeros((len(prefixes_a), len(cls.target_chars)), dtype=np.float64)
        Y_Time = np.zeros((len(prefixes_a)), dtype=np.float32)

        for i, prefix_a in enumerate(prefixes_a):

            leftpad = cls.max_seq_length - len(prefix_a)
            next_event_t = next_events_t[i]
            prefix_t = prefixes_t[i]
            prefix_t2 = prefixes_t2[i]
            prefix_t3 = prefixes_t3[i]
            prefix_t4 = prefixes_t4[i]

            for t, char in enumerate(prefix_a):  # set data
                midnight = prefix_t3[t].replace(hour=0, minute=0, second=0, microsecond=0)
                timesincemidnight = prefix_t3[t] - midnight

                for c in cls.chars:
                    if c == char:
                        X[i, t + leftpad, cls.char_indices[c]] = 1.0

                if cls.tax_features:
                    X[i, t + leftpad, len(cls.chars)] = t + 1
                    X[i, t + leftpad, len(cls.chars) + 1] = prefix_t[t]
                    X[i, t + leftpad, len(cls.chars) + 2] = prefix_t2[t]
                    X[i, t + leftpad, len(cls.chars) + 3] = timesincemidnight.seconds
                    X[i, t + leftpad, len(cls.chars) + 4] = prefix_t4[t].weekday()

            # set label
            for c in cls.target_chars:
                if c == next_events_a[i]:
                    Y_Act[i, cls.target_char_indices[c]] = 1
                else:
                    Y_Act[i, cls.target_char_indices[c]] = 0

            Y_Time[i] = next_event_t / cls.divisor

        cls.X = X
        num_features_all = cls.num_features_all
        num_features_activities = cls.num_features_activities

        return X, Y_Act, Y_Time, cls.max_seq_length, num_features_all, num_features_activities

    @classmethod
    def create_test_set(self):
        lines = []
        caseids = []
        lines_t = []
        lines_t2 = []
        lines_t3 = []

        for index in range(0, len(self.test_indices)):
            lines.append(self.lines[index])
            caseids.append(self.caseids[index])
            lines_t.append(self.timeseqs[index])
            lines_t2.append(self.timeseqs2[index])
            lines_t3.append(self.timeseqs3[index])

        return lines, caseids, lines_t, lines_t2, lines_t3, self.max_seq_length, self.num_features_all, self.num_features_activities


    @classmethod
    def encode_test_set(cls, sentence, prefix_t, prefix_t3):

        X = np.zeros((1, cls.max_seq_length, cls.num_features_all), dtype=np.float32)
        leftpad = cls.max_seq_length - len(sentence)
        prefix_t2 = np.cumsum(prefix_t)

        for t, char in enumerate(sentence):
            midnight = prefix_t3[t].replace(hour=0, minute=0, second=0, microsecond=0)
            timesincemidnight = prefix_t3[t] - midnight

            for c in cls.chars:
                if c == char:
                    X[0, t + leftpad, cls.char_indices[c]] = 1.0

            if cls.tax_features:
                X[0, t + leftpad, len(cls.chars)] = t + 1  # id of event within a case
                X[0, t + leftpad, len(cls.chars) + 1] = prefix_t[t]  # attribute two: the average time difference between the current and the previous event of a process instance
                X[0, t + leftpad, len(cls.chars) + 2] = prefix_t2[t]  # attribute three: the average time difference between the current and the first event of a process instance
                X[0, t + leftpad, len(cls.chars) + 3] = timesincemidnight.seconds  # attribute four: the time difference between  current  event  and  midnight
                X[0, t + leftpad, len(cls.chars) + 4] = prefix_t3[t].weekday()  # attribute five: the  time  difference  between  current event  and  the  beginning  of  week

        return X, cls.num_features_all, len(cls.chars)

    @classmethod
    def get_symbol(cls, predictions):
        max_prediction = 0
        symbol = ''
        i = 0
        for prediction in predictions:
            if prediction >= max_prediction:
                max_prediction = prediction
                symbol = cls.target_indices_char[i]
            i += 1
        return symbol

    @classmethod
    def transform_tensor_to_matrix(cls, X):
        """
        Note: X stores prefixes. Thus, for a case with n events, n-1 prefixes are included.
        """

        number_cases = cls.find_number_of_cases_in_tensor(X)

        case_id = 0
        X_case_based = np.zeros((number_cases, cls.max_seq_length), dtype=np.float32)
        case = list()
        case_time = list()
        cases_time = list()
        cases_time_suffix = list()

        for index in range(X.shape[0]):  # Add n-1 cases
            # Index > 0 -> at the second case look back to the first case
            if index > 0 and X[index, cls.max_seq_length - 1, cls.num_features_activities] == 1:
                # Iterate over all time steps of the case by event id of the case
                # Assumption: the latest time step is at the end
                case_time_steps = int(X[index - 1, cls.max_seq_length - 1, cls.num_features_activities])

                for row_index in range(case_time_steps):
                    for column_index in range(cls.num_features_all):

                        # Activity
                        if column_index < cls.num_features_activities:
                            if X[index - 1, cls.max_seq_length - case_time_steps + row_index, column_index] == 1:
                                case.append(column_index + 1)  # +1 as index starts at 0
                                case_time.append(X[index - 1, cls.max_seq_length - case_time_steps + row_index, cls.num_features_activities + 2])

                # Update of structure
                for case_index in range(len(case)):
                    X_case_based[case_id, case_index] = case[case_index]
                cases_time.append(case_time)
                case_time = []
                case = []
                case_id = case_id + 1

        # Add last case
        case_time_steps = int(X[-1, cls.max_seq_length - 1, cls.num_features_activities])
        for row_index in range(case_time_steps):
            for column_index in range(cls.num_features_all):

                # Activity
                if column_index < cls.num_features_activities:
                    if X[-1, cls.max_seq_length - case_time_steps + row_index, column_index] == 1:
                        case.append(column_index + 1)  # +1 since index at 0
                        case_time.append(X[index - 1, cls.max_seq_length - case_time_steps + row_index, cls.num_features_activities + 2])

        # Update of structure with last case
        for case_index in range(0, len(case)):
            X_case_based[case_id, case_index] = case[case_index]
        cases_time.append(case_time)
        case_time = []

        # Get number of suffixes
        number_suffixes = 0
        for index_case in range(0, len(X_case_based)):
            for index_suffix in range(0, len(cases_time[index_case])):
                number_suffixes = number_suffixes + 1

        # Create suffixes
        X_case_based_suffix = np.zeros((number_suffixes, cls.max_seq_length - 1), dtype=np.float32)
        number_suffixes = 0

        for index_case in range(0, len(X_case_based)):
            length_suffix = len(cases_time[index_case])

            # Pick out max suffix
            case_suffix = X_case_based[index_case][:length_suffix]

            for suffix_index in range(0, len(case_suffix), 1):
                index_suffix_one_step_time = 0
                for index_suffix_pos in range(0, suffix_index + 1, 1):

                    try:
                        # print("%s-%s" % (index_suffix_pos, suffix_index))

                        # Update of event
                        X_case_based_suffix[number_suffixes, index_suffix_pos] = X_case_based[index_case, index_suffix_pos]

                        # Update of time
                        case_time.append(cases_time[index_case][index_suffix_one_step_time])
                        index_suffix_one_step_time = index_suffix_one_step_time + 1

                    except ValueError:
                        pass

                number_suffixes = number_suffixes + 1
                cases_time_suffix.append(case_time)
                case_time = []

        cls.X_case_based_suffix = X_case_based_suffix
        cls.X_case_based_suffix_time = cases_time_suffix

        # print(str(len(cls.X_case_based_suffix)))
        # print(str(len(cls.X_case_based_suffix_time)))

        return X_case_based_suffix

    @classmethod
    def find_number_of_cases_in_tensor(self, X):
        number_cases = 0
        for index in range(X.shape[0]):
            if index > 0 and X[index, self.max_seq_length - 1, self.num_features_activities] == 1:  # index > 0 -> check retrospective the cases during the iteration
                number_cases = number_cases + 1  # iterate over n-1 cases

        return number_cases + 1

    @classmethod
    def transform_current_instance_to_suffix_vector(cls, current, prefix_length, predict_size):
        """
        # Creates the input for the candidate selection-
        # Note: "current" includes the prefix and the suffix.
        """

        vector_length = len(current["line"])
        vector_case_based = np.zeros((cls.max_seq_length), dtype=np.float32)
        vector_case_based_suffix_final = np.zeros((predict_size), dtype=np.float32)
        events = current["line"]

        index_event = 0
        for index in range(0, vector_length, 1):
            vector_case_based[index] = cls.target_char_indices[events[index_event]] + 1  # +1 column index begins with 0
            index_event = index_event + 1

        vector_case_based = vector_case_based.reshape(1, -1)

        # Select suffix
        vector_case_based_suffix = []
        vector_case_based_suffix = vector_case_based[0][prefix_length:]
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
    def create_new_instance(cls, prefix, suffix):

        suffix_no_context = ""
        for index in range(0, len(suffix), 2):
            if suffix[index] > 0:
                suffix_no_context = suffix_no_context + cls.target_indices_char[suffix[index] - 1]
            else:
                break
        new_instance = prefix + suffix_no_context

        return [cls.target_char_indices[element] for element in new_instance]

    @classmethod
    def select_best_candidate_from_training_set(cls, candidates, prefix, args):

        X_case_based_suffix_temp = cls.X_case_based_suffix
        X_case_based_suffix_time_temp = cls.X_case_based_suffix_time

        current_candidate = []
        current_candidate_event_time = []
        current_candidate_total_time = []
        current_candidate_event = []

        # select suffix candidate with minimum total suffix time out of k suffix candidates
        for candidate in candidates[0]:
            new_instance = cls.create_new_instance(prefix, X_case_based_suffix_temp[candidate])
            if cls.check_candidate(args, new_instance):

                if current_candidate == []:
                    current_candidate = X_case_based_suffix_temp[candidate]
                    current_candidate_event_time = X_case_based_suffix_time_temp[candidate][0]
                    current_candidate_total_time = sum(X_case_based_suffix_time_temp[candidate])
                    current_candidate_event = cls.target_indices_char[current_candidate[0] - 1]
                else:
                    if sum(X_case_based_suffix_time_temp[candidate]) < current_candidate_total_time:
                        current_candidate = X_case_based_suffix_temp[candidate]
                        current_candidate_event_time = X_case_based_suffix_time_temp[candidate][0]
                        current_candidate_total_time = sum(X_case_based_suffix_time_temp[candidate])
                        current_candidate_event = cls.target_indices_char[current_candidate[0] - 1]

        return current_candidate_total_time, current_candidate_event_time, current_candidate_event, current_candidate
