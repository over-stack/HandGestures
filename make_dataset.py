import os
import random
import shutil


hagrid_dataset_path = 'hagrid_dataset_512'
my_dataset_path = 'dataset'
num_samples = 500

none_class = 'none'
target_class = 'stop'

target_none_class_path = os.path.join(my_dataset_path, none_class)
target_class_path = os.path.join(my_dataset_path, target_class)

# NONE
none_classes = list(set(os.listdir(hagrid_dataset_path)).difference(set(list([target_class]))))
print(none_classes)
num_samples_per_none_class = num_samples // len(none_classes)
for class_name in none_classes:
    class_dir_path = os.path.join(hagrid_dataset_path, class_name)
    class_filenames = os.listdir(class_dir_path)
    chosen_filenames = random.sample(class_filenames, num_samples_per_none_class)

    for filename in chosen_filenames:
        filepath = os.path.join(class_dir_path, filename)
        shutil.copy(filepath, target_none_class_path)

# TARGET
target_class_dir_path = os.path.join(hagrid_dataset_path, target_class)
target_class_filenames = os.listdir(target_class_dir_path)
target_chosen_filenames = random.sample(target_class_filenames, num_samples)
for filename in target_chosen_filenames:
    filepath = os.path.join(target_class_dir_path, filename)
    shutil.copy(filepath, target_class_path)
