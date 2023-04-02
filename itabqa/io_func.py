import fnmatch
import os  # folder directory navigation


def get_files_from_dir_stream(folder_path, extension="*"):
    for root, _, file_dirs in os.walk(folder_path):
        for file_dir in fnmatch.filter(file_dirs, "*.%s" % extension):
            if ".DS_Store" not in file_dir:
                yield os.path.join(root, file_dir)


def get_files_from_dir_subdir(folder_path, extension="*"):
    all_files = []
    for root, _, file_dirs in os.walk(folder_path):
        for file_dir in fnmatch.filter(file_dirs, "*.%s" % extension):
            if ".DS_Store" not in file_dir:
                all_files.append(os.path.join(root, file_dir))
    return all_files


def get_files_from_dir(
    folder_path, extension="*", limit_reader=-1, is_sort=False, reverse=False
):
    all_file_dirs = get_files_from_dir_subdir(folder_path, extension)

    if is_sort:
        file_with_size = [(f, os.path.getsize(f)) for f in all_file_dirs]
        file_with_size.sort(key=lambda f: f[1], reverse=reverse)
        all_file_dirs = [f for f, _ in file_with_size]
    if limit_reader < 0:
        limit_reader = len(all_file_dirs)
    return all_file_dirs[:limit_reader]
