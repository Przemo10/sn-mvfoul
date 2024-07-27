import os
import re

def detect_wrong_actions(directory, split,  size_threshold_mb = 0.5):
    low_size_files = []
    size_threshold_kb = size_threshold_mb * 1024  # Convert megabytes to kilobytes


    input_dir = f'./{directory}/{split}'

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.mp4'):
                file_path = os.path.join(root, file)
                file_size_kb = os.path.getsize(file_path) / 1024  # File size in kilobytes

                if file_size_kb < size_threshold_kb:
                    low_size_files.append(file_path)

    action_ids = [re.search(r'action_(\d+)', p1).group(1) for p1 in low_size_files]


    return action_ids, len(action_ids)


print(detect_wrong_actions('dataset720p', split='train',  size_threshold_mb=0.5))