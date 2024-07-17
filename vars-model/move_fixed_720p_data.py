import os
import shutil
import zipfile

# Define the paths
zip_file_path = './mv-foul-720p-fix.zip'
mv_foul_dir = './dataset720p/mv-foul-720p-fix'
base_dir = './dataset720p/'


# Function to extract the zip file
def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f'Extracted {zip_path} to {extract_to}')


# Function to parse the filename and extract the relevant components
def parse_filename(filename):
    parts = filename.split('_')
    if len(parts) == 5 and parts[0] in ['valid', 'train', 'test']:
        dataset = parts[0]
        action = f"{parts[1]}_{parts[2]}"
        clip = f"{parts[3]}_{parts[4]}"
        return dataset, action, clip
    return None, None, None


# Function to move and rename the files
def move_and_rename_files(mv_foul_dir, base_dir):
    for filename in os.listdir(mv_foul_dir):
        if filename.endswith('.mp4'):
            dataset, action, clip = parse_filename(filename)
            if dataset and action and clip:
                # Construct the target directory and file paths
                target_dir = os.path.join(base_dir, dataset, action)
                # Rename clip from 'clips_2.mp4' to 'clip_2.mp4'
                target_filename = clip.replace('clips_', 'clip_')
                target_file = os.path.join(target_dir, target_filename)
                source_file = os.path.join(mv_foul_dir, filename)

                # Ensure the target directory exists
                os.makedirs(target_dir, exist_ok=True)

                # Remove the old file if it exists
                if os.path.exists(target_file):
                    os.remove(target_file)

                # Move and rename the new file
                shutil.move(source_file, target_file)
                print(f'Moved and renamed {source_file} to {target_file}')


# Extract the zip file
extract_zip(zip_file_path, './dataset720p/')

# Move and rename the files
move_and_rename_files(mv_foul_dir, base_dir)