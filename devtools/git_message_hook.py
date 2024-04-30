# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024 Mira Geoscience Ltd.                                     '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

"""Some Git pre-commit hooks implementations."""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys


def get_jira_id(text) -> str:
    """Detect a JIRA issue ID at the begging of the given text.

    :return: the JIRA issue ID if found, else empty string
    """

    class JiraPattern:
        """Internal class that encapsulates the regular expression for the JIRA pattern,
        making sure it gets compiled only once."""

        __pattern = re.compile(
            r"(?:GEOPY|GI|GA|GMS|VPem1D|VPem3D|VPmg|UBCGIF|LICMGR)-\d+"
        )

        @staticmethod
        def get():
            """:return: the compiled regular expression for the JIRA pattern"""
            return JiraPattern.__pattern

    # use re.match() rather than re.search() to enforce the JIRA reference to be at the beginning
    match = re.match(JiraPattern.get(), text.strip())
    return match.group(0) if match else ""


def get_branch_name() -> str | None:
    """:return: the name of the current branch"""

    git_proc = subprocess.run(
        shlex.split("git branch --list"),
        stdout=subprocess.PIPE,
        text=True,
        check=False
    )

    # cannot use HEAD during rebase
    # git_proc = subprocess.run(
    #        shlex.split('git symbolic-ref --short HEAD'), stdout=subprocess.PIPE, universal_newlines=True
    #    )
    # Note: version above suggested by Atlassian. Could also use: git rev-parse --abbrev-ref HEAD

    if git_proc.returncode != 0:
        return None

    current_branch = None
    # current branch is prefixed by '*'
    for line in git_proc.stdout.splitlines():
        stripped = line.strip()
        if stripped and stripped[0] == "*":
            current_branch = stripped[1:]
            break
    assert current_branch is not None

    class RebasingPattern:
        """Internal class that encapsulates the regular expression for the rebasing
        message pattern, making sure it gets compiled only once."""

        __pattern = re.compile(r"\(.*\s(\S+)\s*\)")

        @staticmethod
        def get():
            """:return: the compiled regular expression for the Rebasing pattern"""
            return RebasingPattern.__pattern

    match = re.match(RebasingPattern.get(), current_branch.strip())
    if match:
        return match.group(1)

    return current_branch


def check_commit_message(filepath: str) -> tuple[bool, str]:
    """Check if the branch name or the commit message starts with a reference to JIRA,
    and if the message meets the minimum required length for the summary line.

    The JIRA reference has to be at the beginning of the branch name, or of the commit
    message.
    :return: a tuple telling whether the commit message is valid or not, and an error
        message (empty in case the message is valid).
    """

    branch_jira_id = ""
    branch_name = get_branch_name()
    if branch_name:
        branch_jira_id = get_jira_id(branch_name)

    message_jira_id = ""
    first_line = None
    with open(filepath) as message_file:
        for line in message_file:
            if not line.startswith("#") and len(line.strip()) > 0:
                # test only the first non-comment line that is not empty
                # (should we reject messages with empty first line?)
                first_line = line
                message_jira_id = get_jira_id(first_line)
                break
    assert first_line is not None

    if not branch_jira_id and not (
        message_jira_id or first_line.strip().lower().startswith("merge")
    ):
        return (
            False,
            "Either the branch name or the commit message must start with a JIRA ID.",
        )

    if branch_jira_id and message_jira_id and branch_jira_id != message_jira_id:
        return (
            False,
            "Different JIRA ID in commit message %s and in branch name %s."
            % (message_jira_id, branch_jira_id),
        )

    stripped_message_line = ""
    if first_line:
        stripped_message_line = first_line.strip()
        if message_jira_id:
            stripped_message_line = stripped_message_line[
                len(message_jira_id) + 1 :
            ].strip()

    min_required_length = 10
    if len(stripped_message_line) < min_required_length:
        return (
            False,
            "First line of commit message must be at least %s characters long, "
            "beyond the JIRA ID." % min_required_length,
        )

    return True, ""


def check_commit_msg(filepath: str) -> None:
    """To be used a the Git commit-msg hook.

    Exit with non-0 status if the commit message is deemed invalid.
    """

    (is_valid, error_message) = check_commit_message(filepath)
    if not is_valid:
        print(
            """commit-msg hook: **ERROR** %s
            Message has been saved to %s."""
            % (error_message, filepath)
        )
        sys.exit(1)


def prepare_commit_msg(filepath: str, source: str | None = None) -> None:
    """To be used a the Git prepare-commit-msg hook.

    Will add the JIRA ID found in the branch name in case it is missing from the commit
    message.
    """

    branch_jira_id = ""
    branch_name = get_branch_name()
    if branch_name:
        branch_jira_id = get_jira_id(branch_name)

    if not branch_jira_id:
        return

    if source not in [None, "message", "template"]:
        return

    with open(filepath, "r+", encoding="utf-8") as message_file:
        message_has_jira_id = False
        message_lines = message_file.readlines()
        for line_index, line_content in enumerate(message_lines):
            if not line_content.startswith("#"):
                # test only the first non-comment line
                message_jira_id = get_jira_id(line_content)
                if not message_jira_id:
                    message_lines[line_index] = branch_jira_id + ": " + line_content
                message_has_jira_id = True
                break

        if not message_has_jira_id:
            # message is empty or all lines are comments: insert JIRA ID at the very beginning
            message_lines.insert(0, branch_jira_id + ": ")

        message_file.seek(0, 0)
        message_file.write("".join(message_lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("msg_file", help="the message file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-p", "--prepare", action="store_true", help="prepare the commit message"
    )
    group.add_argument(
        "-c",
        "--check",
        action="store_true",
        help="check if the commit message is valid",
    )
    parser.add_argument("args", nargs=argparse.REMAINDER)

    args = parser.parse_args()
    if args.prepare:
        prepare_commit_msg(args.msg_file, *args.args)
    elif args.check:
        check_commit_msg(args.msg_file)
