import pandas as pd
import numpy as np
import heapq
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import tree

MAX_SAMPLES_PER_STEP = 5  # max data samples per schedule step


def data_loader_for_training(paths, max_samples_per_step, enable_all_false_samples=False, random_seed=90):  # seed: 30
    np_random = np.random
    np_random.seed(random_seed)
    data = []
    for path in paths:
        with open(path) as file:
            shall_read = False
            true_sample = None
            false_samples = []

            for line in file:
                line = line.strip()
                if line == '':
                    if true_sample is not None and len(false_samples) != 0:  # merge multiple samples (one must be true)
                        random_samples_idx = set()
                        while len(random_samples_idx) < min(2 * max_samples_per_step, len(false_samples)):
                            random_samples_idx.add(np_random.randint(0, len(false_samples)))
                        counts = 0
                        for idx in random_samples_idx:
                            if counts < len(random_samples_idx) / 2:  # half for '10' samples
                                sample_10 = true_sample[: -1] + false_samples[idx][: -1] + ['10']
                                data.append(sample_10)
                            else:  # half for '01' samples
                                sample_01 = false_samples[idx][: -1] + true_sample[: -1] + ['01']
                                data.append(sample_01)
                            counts += 1

                        if enable_all_false_samples is True:  # for all-false sample, e.g. '00' if num_bits = 2
                            random_samples_idx = list(random_samples_idx)
                            num_samples = 0
                            for i in range(len(random_samples_idx)):
                                for j in range(i, len(random_samples_idx)):
                                    sample_10 = false_samples[random_samples_idx[i]][: -1] + false_samples[random_samples_idx[j]][: -1] + ['00']
                                    sample_01 = false_samples[random_samples_idx[j]][: -1] + false_samples[random_samples_idx[i]][: -1] + ['00']
                                    data.append(sample_10)
                                    data.append(sample_01)
                                    num_samples += 2
                                    if num_samples > max_samples_per_step:  # really tricky here ! Get acc 0.68.
                                        break

                if line.startswith('step'):
                    if line.endswith('True'):
                        shall_read = True
                    else:
                        shall_read = False
                    true_sample = None
                    false_samples = []
                    continue
                if shall_read:
                    vals = line.strip().split(',')
                    vals = [float(val) if (i != 0 and i != len(vals) - 1) else val
                            for i, val in enumerate(vals)]
                    if vals[-1] == 'True':
                        true_sample = vals[1:]
                    else:
                        false_samples.append(vals[1:])

    original_columns = ['run_time_cp', 'num_tasks_cp', 'run_time_overall', 'num_tasks_overall',
                        'run_time_smallest_ready_node', 'num_tasks_smallest_ready_node']
    columns = []
    for i in range(2):
        columns.extend([col + '_' + str(i) for col in original_columns])

    columns.append('be_scheduled')

    job_dataframe = pd.DataFrame(data, columns=columns)

    # for data statistics
    labels = job_dataframe['be_scheduled']
    print('data statistics: \n', labels.value_counts())
    return job_dataframe


def data_loader_for_testing(paths):
    data = []
    for path in paths:
        with open(path) as file:
            shall_read = False
            true_job_id = None
            all_samples = []

            for line in file:
                line = line.strip()
                if line == '':
                    if true_job_id is not None and true_job_id != 'None' and len(
                            all_samples) >= 2:  # merge multiple samples to one
                        data_current = {}
                        data_current['true_job_id'] = true_job_id
                        data_current['data_samples'] = []

                        for i in range(len(all_samples)):
                            for j in range(len(all_samples)):
                                if i == j:
                                    continue

                                merged_data_item = all_samples[i][1 : -1] + all_samples[j][1 : -1] + \
                                                   [all_samples[i][0], all_samples[j][0]]
                                data_current['data_samples'].append(merged_data_item)
                        data.append(data_current)
                if line.startswith('step'):
                    if line.endswith('True'):
                        shall_read = True
                    else:
                        shall_read = False
                    true_job_id = None
                    all_samples = []
                    continue
                if shall_read:
                    vals = line.strip().split(',')
                    vals = [float(val) if (i != 0 and i != len(vals) - 1) else val
                            for i, val in enumerate(vals)]
                    if vals[-1] == 'True':
                        true_job_id = vals[0]  # job id
                    all_samples.append(vals)

    original_columns = ['run_time_cp', 'num_tasks_cp', 'run_time_overall', 'num_tasks_overall',
                        'run_time_smallest_ready_node', 'num_tasks_smallest_ready_node']
    columns = []
    for i in range(2):
        columns.extend([col + '_' + str(i) for col in original_columns])

    original_label = 'job_id'
    for i in range(2):
        columns.append(original_label + '_' + str(i))

    for data_per_stage in data:
        data_per_stage['data_samples'] = pd.DataFrame(data_per_stage['data_samples'], columns=columns)

    # for data statistics
    print('data statistics (length): \n', len(data))
    return data


def rf_predictor(X, y):
    # best : n_estimator 15, max_samples 0.95, training result: precision 0.41 recall 0.38 accuracy 0.92 auc of roc 0.76903
    rf_classfier = RandomForestClassifier(n_estimators=15, oob_score=True, max_features=None, class_weight='balanced',
                                          max_samples=0.95, max_depth=9)

    rf_classfier.fit(X, y)
    return rf_classfier


def write_prediction_to_file(X, y, y_pred, path):
    true_label, false_label = '10', '01'
    with open(path, 'w') as file:
        for i in range(X.shape[0]):
            x, y_t, y_p = X[i], y[i], y_pred[i]
            label = true_label if y_t == 1 else false_label
            label_p = true_label if y_p == 1 else false_label
            file.write(str(x) + ' label:' + label + ' pred: ' + label_p + '\n')


def train(max_sample_per_step=MAX_SAMPLES_PER_STEP, enable_all_false_samples=False, num_bits_in_class_label=2):
    job_dataframe = data_loader_for_training(['./dataset/schedule_learn_30-jobs-1.txt',
                                                            './dataset/schedule_learn_30-jobs-2.txt',
                                                            './dataset/schedule_learn_30-jobs-3.txt',
                                                            './dataset/schedule_learn_30-jobs-4.txt'],
                                             max_samples_per_step=max_sample_per_step,
                                             enable_all_false_samples=enable_all_false_samples)
    X = job_dataframe.loc[:, : 'num_tasks_smallest_ready_node_' + str(num_bits_in_class_label - 1)]
    y = job_dataframe.loc[:, 'be_scheduled']
    predictor = rf_predictor(X, y)
    return predictor


def test_v1(predictor, num_bits_in_class_label=2, enable_all_false_samples=False):
    """load data by only merging the true sample with the false samples. It is used to purely test the model performance,
    e.g. precision, recall, roc"""
    test_dataframe = data_loader_for_training(['./dataset/schedule_learn_30-jobs-5.txt'],
                                              max_samples_per_step=MAX_SAMPLES_PER_STEP,
                                              enable_all_false_samples=enable_all_false_samples)  # test path to set
    X_test = test_dataframe.loc[:, : 'num_tasks_smallest_ready_node_1']
    y_test = test_dataframe.loc[:, 'be_scheduled']

    y_pred_prob = predictor.predict_proba(X_test)
    class_labels = predictor.classes_
    print('class labels:', class_labels)

    y_pred = np.argmax(y_pred_prob, axis=1)
    y_pred_labels = [class_labels[i] for i in y_pred]
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred_labels)
    accuracy = metrics.accuracy_score(y_test, y_pred_labels)
    print('testing result: accuracy %0.2f' % accuracy)
    print('confusion matrix: \n', confusion_matrix)

    metrics.plot_confusion_matrix(predictor, X_test, y_test,
                                  display_labels=['Neg-pos.', 'Pos-neg.'],
                                  cmap='PuBu'
                                  )
    plt.xticks(size=25)
    plt.yticks(size=25)
    plt.xlabel('True label', fontsize=32)
    plt.ylabel('Predicted  label', fontsize=32)
    plt.show()

    if len(class_labels) == 2:  # binary-class metrics: recall precision
        roc_auc = metrics.roc_auc_score(y_test, y_pred_prob[:, 1])
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob[:, 1], pos_label='10')
        precision = metrics.precision_score(y_test, y_pred_labels, pos_label='10')
        recall = metrics.recall_score(y_test, y_pred_labels, pos_label='10')
        print('testing result: recall %0.2f precision %0.2f' % (recall, precision))

        # write_prediction_to_file(test_dataframe.to_numpy(), y_test.to_numpy(), y_pred,
        #                          path='./results/prediction_dataset-5.txt')  # result path to set

        # draw the roc curve
        # plt.title('precision %0.2f recall %0.2f accuracy %0.2f auc of roc %0.2f'
                  # % (precision, recall, accuracy, roc_auc))
        plt.plot(fpr, tpr, color='darkred', lw=2, label='ROC (auc=%0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False positive rate', fontsize=32)
        plt.ylabel('True positive rate', fontsize=32)
        plt.xticks(size=25)
        plt.yticks(size=25)
        plt.legend(loc="lower right", fontsize=25)
        plt.show()


def test_v2(predictor, top_n=3, num_bits_in_class_label=2):
    """load data by merging all paired candidate samples (both true and false samples). It is used to test the accuracy
    of prediction in job sample is true."""
    test_data = data_loader_for_testing(['./dataset/schedule_learn_30-jobs-5.txt'])
    true_labels = [data_per_stage['true_job_id'] for data_per_stage in test_data]
    pred_labels = []
    class_labels = predictor.classes_
    print('class labels:', class_labels)

    stage = 0
    for data_per_stage in test_data:
        data_samples = data_per_stage['data_samples']  # dataframe
        X_test = data_samples.loc[:, : 'num_tasks_smallest_ready_node_' + str(num_bits_in_class_label - 1)]  # dataframe
        y_pred_prob = predictor.predict_proba(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_pred_labels = [class_labels[i] for i in y_pred]

        # get the chosen job id within each pair
        chosen_job_ids = {}  # the chosen job id for each pair in the current stage
        for i in range(len(data_samples)):
            d_ = data_samples.loc[i]
            y_ = y_pred_labels[i]
            job_id = None
            for k in range(len(y_)):
                if y_[k] == '1':
                    job_id = d_['job_id_' + str(k)]
                    break

            if job_id is not None:
                if job_id not in chosen_job_ids:
                    chosen_job_ids[job_id] = 1
                else:
                    chosen_job_ids[job_id] += 1
        # select the job id that is chosen mostly as the job for schedule at current stage
        pred_top_n = heapq.nlargest(top_n, chosen_job_ids.items(), key=lambda x: x[1])
        pred_labels.append(pred_top_n)
        stage += 1
    return true_labels, pred_labels


def get_predict_explanation(job_0, job_1, decision_tree_id):
    X = job_0 + job_1  # label: 10

    decision_tree = predictor.estimators_[decision_tree_id]
    print('classes:', predictor.classes_, decision_tree.classes_)

    pred_label = decision_tree.predict([X])
    path = decision_tree.decision_path([X])

    print('feature', X, 'pred_label:', pred_label)
    print('decision path:')
    for node_id in list(path.indices):
        feature_id = decision_tree.tree_.feature[node_id]
        threshold_val = decision_tree.tree_.threshold[node_id]
        if X[feature_id] > threshold_val:
            inequal_sign = '>'
        else:
            inequal_sign = '<'
        print('decision node', node_id, 'feature', feature_id, '=', X[feature_id], inequal_sign,
              threshold_val)

if __name__ == '__main__':
    predictor = train(max_sample_per_step=5, enable_all_false_samples=False, num_bits_in_class_label=2)
    # test_v1
    # test_v1(predictor, enable_all_false_samples=False, num_bits_in_class_label=2)

    # test_v2
    # true_labels, pred_labels = test_v2(predictor, top_n=3, num_bits_in_class_label=2)
    #
    # corrects = 0
    # for t_, p_ in zip(true_labels, pred_labels):
    #     is_correct = False
    #     for l_, _ in p_:
    #         if t_ == l_:
    #             corrects += 1
    #             is_correct = True
    #             break
    #     print(is_correct, ' label: ', t_, ' pred: ', p_)
    #
    # accuracy = corrects / len(true_labels)
    # print('accuracy:', accuracy)

    # test_v3
    job_0 = [112.44,614,115.88,616,4.62,2]  # job 13
    job_1 = [1385.17,1080,1600.87,1186,215.7,580]  # job 16
    get_predict_explanation(job_0, job_1, decision_tree_id=0)

    # tree_structure = tree.export_text(decision_tree)
    # print(tree_structure)

    # tree.plot_tree(decision_tree)
    # plt.show()




