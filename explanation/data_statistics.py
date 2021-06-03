import numpy as np
import matplotlib.pyplot as plt


def data_loader(paths):
    data = []
    for path in paths:
        with open(path) as file:
            samples_decided_by_locality = {'total_stages': 0, 'positive_samples': 0, 'negative_samples': 0}
            overall_statistics = {'total_stages': 0, 'positive_samples': 0, 'negative_samples': 0}

            for line in file:
                line = line.strip()
                if line == '':
                    if locality:
                        samples_decided_by_locality['total_stages'] += 1
                        if positive_locality:
                            samples_decided_by_locality['positive_samples'] += 1
                        else:
                            samples_decided_by_locality['negative_samples'] += 1

                    overall_statistics['total_stages'] += 1
                    overall_statistics['positive_samples'] += positive_samples
                    overall_statistics['negative_samples'] += negative_samples

                if line.startswith('step'):
                    words = line.split(' ')
                    chosen_jobs = []
                    for word in words:
                        if word.startswith('Job-'):
                            chosen_jobs.append(word)

                    if len(chosen_jobs) == 2:
                        locality = True
                        if chosen_jobs[0][: -1] == chosen_jobs[1]:
                            positive_locality = True
                        else:
                            positive_locality = False
                    else:
                        locality = False
                    positive_samples = 0
                    negative_samples = 0
                    continue

                vals = line.strip().split(',')
                # vals = [float(val) if (i != 0 and i != len(vals) - 1) else val
                        # for i, val in enumerate(vals)]
                if vals[-1] == 'True':
                    positive_samples += 1
                else:
                    negative_samples += 1

            data.append([overall_statistics, samples_decided_by_locality])
    return data


def get_entropy():
    original = {'p': 1577, 'n': 25811}
    locality = {'p': 1388, 'n': 177}
    nonlocality = {'p': 189, 'n': 25634}
    summ_ori = original['p'] + original['n']
    summ_loc = locality['p'] + locality['n']
    summ_non_loc = nonlocality['p'] + nonlocality['n']

    def entropy(info):
        summ = info['p'] + info['n']
        entropy = - info['p'] / summ * np.log2(info['p'] / summ) - info['n'] / summ * np.log2(info['n'] / summ)
        return entropy

    en_ori = entropy(original)
    en_loc = entropy(locality)
    en_non_loc = entropy(nonlocality)
    gain = en_ori - (summ_loc / summ_ori * en_loc + summ_non_loc / summ_ori * en_non_loc)
    iv = - summ_loc / summ_ori * np.log2(summ_loc / summ_ori) - summ_non_loc / summ_ori * np.log2(summ_non_loc / summ_ori)
    gain_ratio = gain / iv
    print('gain ratio', gain_ratio)  # 0.7272


def rf_model_accuracy():
    # random seed from 10 to 100
    # training accuracy on dataset 1
    top_1 = [0.609, 0.536, 0.634, 0.609, 0.609, 0.658, 0.609, 0.536, 0.536, 0.56]
    top_2 = [0.853, 0.780, 0.878, 0.853, 0.829, 0.853, 0.878, 0.829, 0.902, 0.805]
    top_3 = [0.926, 0.927, 0.926, 0.902, 0.951, 0.927, 0.975, 0.902, 0.975, 0.927]

    # testing accuracy on dataset 4
    top_1 = [0.477, 0.409, 0.477, 0.454, 0.386, 0.386, 0.432,  0.454, 0.477, 0.409]
    top_2 = [0.75, 0.727, 0.727, 0.659, 0.681, 0.636, 0.659, 0.659, 0.75, 0.704]
    top_3 = [0.886, 0.818,  0.795, 0.772, 0.75, 0.818, 0.795, 0.886, 0.863, 0.863]


    plt.boxplot([top_1, top_2, top_3], labels=['Top-1.', 'Top-2.', 'Top-3.'])
    plt.grid(axis='y')
    plt.ylim([0.0, 1.05])
    plt.ylabel('Accuracy', fontsize=32)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.show()


def get_statistics():
    data_statistics = data_loader(['./dataset/tmp/schedule_learn_30-jobs-1.txt',
                                   './dataset/tmp/schedule_learn_30-jobs-2.txt',
                                   './dataset/tmp/schedule_learn_30-jobs-3.txt',
                                   './dataset/tmp/schedule_learn_30-jobs-4.txt',
                                   './dataset/tmp/schedule_learn_30-jobs-5.txt'])
    i = 1
    for info_a, info_b in data_statistics:
        print('dataset:', i)
        print('overall:', info_a)
        print('locality related:', info_b)
        i += 1

    get_entropy()


def show_jct_benefit():
    original_jct = [38.042, 44.802, 85.069, 131.686, 165.963, 223.615, 295.984, 320.215, 485.691, 583.771]
    perturbed_jct = [25.287, 44.563,  79.491, 131.686, 146.028,  149.09, 270.334, 301.41, 437.679, 565.988]

    original_jct = np.array(original_jct)
    perturbed_jct = np.array(perturbed_jct)

    ratio = perturbed_jct / original_jct
    X = np.arange(len(original_jct))
    ratio_less_than_1 = [(i, r) for i, r in enumerate(ratio) if r < 1]
    ratio_equal_1 = [(i, r) for i, r in enumerate(ratio) if r == 1]
    X_0, y_0 = zip(*ratio_less_than_1)
    X_1, y_1 = zip(*ratio_equal_1)
    plt.bar(X_0, y_0, 0.5, zorder=3, color='brown', hatch='/', edgecolor='black', label='Benefited')
    plt.bar(X_1, y_1, 0.5, zorder=3, color='grey', edgecolor='black', label='Not changed')
    plt.plot(np.arange(-1, 11), np.ones(12), '-', color='darkred', linewidth=2)
    plt.xlim((-1.0, 10.0))
    plt.ylim((0, 1.2))
    plt.xticks(X, fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylabel('Normalized JCT', fontsize=32)
    plt.xlabel('Job #', fontsize=32)
    plt.grid(axis='y', zorder=1)
    plt.legend(ncol=2, loc=9, fontsize=20)
    plt.show()


if __name__ == '__main__':
    show_jct_benefit()


