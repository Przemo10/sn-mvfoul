import numpy as np
import pandas as pd



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


def save_training_history(model_dir_path):
    data = []
    for set_name, metrics_dict in TRAINING_RESULT_DICT.items():
        num_epochs = len(next(iter(metrics_dict.values())))
        for epoch in range(num_epochs):
            row = {'set': set_name, 'epoch': epoch + 1}
            for metric_name, values in metrics_dict.items():
                row[metric_name] = values[epoch]
            data.append(row)

    # Create the DataFrame
    df = pd.DataFrame(data)

    output_filename = f"{model_dir_path}/history.csv"

    # Save DataFrame to CSV
    df.to_csv(output_filename, index=False)

def get_leaderboard_summary(highest_valid_index, highest_test_index):
    leaderboard_summary = {
        "best_val_epoch": highest_valid_index + 1,
        "leaderboard_value_val_best_valid": TRAINING_RESULT_DICT["valid"]["leaderboard_value"][highest_valid_index],
        "leaderboard_value_test_best_valid": TRAINING_RESULT_DICT["test"]["leaderboard_value"][highest_valid_index],
        "best_test_epoch": highest_test_index + 1,
        "leaderboard_value_valid_best_test": TRAINING_RESULT_DICT["valid"]["leaderboard_value"][highest_test_index],
        "leaderboard_value_test_best_test": TRAINING_RESULT_DICT["test"]["leaderboard_value"][highest_test_index],
        "last_saved_epoch": len(TRAINING_RESULT_DICT["valid"]["leaderboard_value"]),
        "leaderboard_value_valid_last_epoch": TRAINING_RESULT_DICT["valid"]["leaderboard_value"][ 1],
        "leaderboard_value_test_last_epoch": TRAINING_RESULT_DICT["test"]["leaderboard_value"][- 1],
    }
    return leaderboard_summary