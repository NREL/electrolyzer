import re
from pathlib import Path

import yaml


def load_yaml(filename, loader=yaml.SafeLoader) -> dict:
    if isinstance(filename, dict):
        return filename  # filename already yaml dict
    with Path.open(filename) as fid:
        return yaml.load(fid, loader)


def make_unique_case_name(folder, proposed_fname, fext):
    """Generate a filename that does not already exist in a user-defined folder.

    Args:
        folder (str | Path): directory that a file is expected to be created in.
        proposed_fname (str): filename (with extension) to check for existence and
            to use as the base file description of a new an unique file name.
        fext (str): file extension, such as ".csv", ".sql", ".yaml", etc.

    Returns:
        str: unique filename that does not yet exist in folder.
    """
    if "." not in fext:
        fext = f".{fext}"

    # if file(s) exist with the same base name, make a new unique filename
    file_base = proposed_fname.split(fext)[0]
    existing_files = [f for f in Path(folder).glob(f"**/*{fext}") if file_base in f.name]
    if len(existing_files) == 0:
        return proposed_fname

    # get past numbers that were used to make unique files by matching
    # filenames against the file base name followed by a number
    past_numbers = [
        int(re.findall(f"{file_base}[0-9]+", str(fname))[0].split(file_base)[-1])
        for fname in existing_files
        if len(re.findall(f"{file_base}[0-9]+", str(fname))) > 0
    ]

    if len(past_numbers) > 0:
        # if multiple files have the same basename followed by a number,
        # take the maximum unique number and add one
        unique_number = int(max(past_numbers) + 1)
        return f"{file_base}{unique_number}{fext}"
    else:
        # if no files have the same basename followed by a number,
        # but do have the same basename, then add a zero to the file basename
        return f"{file_base}0{fext}"
