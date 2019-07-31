#!/usr/bin/env python3

import argparse
import os
import glob
import inquirer
from termcolor import colored


def main(logdir, symlinks_dir):
    os.makedirs(symlinks_dir, exist_ok=True)

    def directory_content(dir):
        files = glob.glob(os.path.join(dir, "*"))
        files = list(map(os.path.basename, files))
        return files

    all_runs = sorted(directory_content(logdir))
    existing_symlinks = set(directory_content(symlinks_dir))

    # ask which symlinks to create or delete
    questions = [
        inquirer.Checkbox(
            "selected_runs",
            message="Select runs you want to display in Tensorboard - [Space] to select [Enter] to finish",
            choices=all_runs,
            default=existing_symlinks,
        )
    ]
    answers = inquirer.prompt(questions)
    selected_runs = answers["selected_runs"]

    # remove not selected symlinks
    for symlink in existing_symlinks:
        if symlink not in selected_runs:
            print(f"Delete link for run {symlink}")
            os.remove(os.path.join(symlinks_dir, symlink))

    # create missing symlinks
    relative_prefix = os.path.relpath(logdir, symlinks_dir)
    for symlink_target in selected_runs:
        symlink_file = os.path.join(symlinks_dir, symlink_target)
        if not os.path.exists(symlink_file):
            print(f"Create link for run {symlink_target}")
            symlink_target_file = os.path.join(relative_prefix, symlink_target)
            os.symlink(symlink_target_file, symlink_file)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Create symlinks for Tensorboard runs in another directory")
    parser.add_argument("--logdir", default="output/00_logdir", help="Tensorboard logdir containing logs")
    parser.add_argument("--symlinks_dir", default="tensorboard", help="Directory for links to original logs")
    # fmt: on
    args = args = parser.parse_args()

    main(args.logdir, args.symlinks_dir)
