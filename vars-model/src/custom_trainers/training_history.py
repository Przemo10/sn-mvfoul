import numpy as np


def create_set_name_all_results():
    return {
        'accuracy_offence_severity': [],
        'accuracy_action': [],
        'balanced_accuracy_offence_severity': [],
        'balanced_accuracy_action': [],
        'leaderboard_value': []
    }


TRAINING_RESULT_DICT =  {
    'train': create_set_name_all_results(),
    'valid': create_set_name_all_results(),
    'test':  create_set_name_all_results()
}


def update_epoch_results_dict(set_name,  epoch_results):
    for key, value in epoch_results.items():
        TRAINING_RESULT_DICT[set_name][key].append(value)


def find_highest_leaderboard_index(training_result_dict, set_name):
    leaderboard_values = training_result_dict[set_name]['leaderboard_value']
    if not leaderboard_values:
        return None
    max_value = max(leaderboard_values)
    max_index = leaderboard_values.index(max_value)
    return max_index


def get_best_n_metric_result(set_name, metric = 'leaderboard_value', best_n_metric=3):
    metric_array = np.array(TRAINING_RESULT_DICT[set_name][metric])
    return metric_array[np.argsort(metric_array)[-best_n_metric]]