import difflib
import random
import re
import statistics
from dataclasses import dataclass

from securityaware.handlers.github import LocalGitFile


def get_sline(pointers, idx, part, signal):
    return pointers[idx].split(' ')[part].split(',')[0].replace(signal, '')


def match_snippet(match, flines):
    if not match:
        print("No match")
        return None

    size = len(match.groups())
    snippet = ''

    for i in range(1, size + 1):
        start, end = match.group(i).split(',')
        snippet += '\n'.join(flines[int(start): int(end)]) + '\n'
#        print(start, end, snippet)

    return snippet


class Changes:
    def __init__(self, patch):
        self.pointers = re.findall("@@ .* @@", patch)
        for pointer in self.pointers:
            patch = patch.replace(pointer, '<##LINE_NUMBERS##>')
        self.patch = patch
        self.deleted_lines = []
        self.added_lines = []
        self.block = None
        self.block_lines = None

    def split_changes(self):
        return self.patch.split('<##LINE_NUMBERS##>')[1::]

    def get_sdel(self, idx):
        return get_sline(self.pointers, idx, 1, '-')

    def get_sadd(self, idx):
        return get_sline(self.pointers, idx, 2, '+')

    def set_block(self, block):
        self.block = block

    def split_block(self):
        self.block_lines = self.block.split('\n')

    def get_deleted_lines(self, start_line: int):
        start_line, line_number = int(start_line)-1, int(start_line)
        for idx, line in enumerate(self.block_lines, start=start_line):
            if idx == start_line:
                continue

            if not re.search(r'^\+', line):
                if re.search(r'^-', line):
                    self.deleted_lines.append(line_number)
                line_number += 1
            else:
                continue

    def get_added_lines(self, start_line: int):
        start_line, line_number = int(start_line)-1, int(start_line)
        for idx, line in enumerate(self.block_lines, start=start_line):
            if idx == start_line:
                continue

            if not re.search(r'^-', line):
                if re.search(r'^\+', line):
                    self.added_lines.append(line_number)
                line_number += 1
            else:
                continue


@dataclass
class Triplet:
    vuln_file: LocalGitFile
    fix_file: LocalGitFile
    non_vuln_file: LocalGitFile
    real: bool = False

    def get_snippet(self):
        vuln_str = None
        fix_str = None
        non_vuln_str = None
        chunks = None

        if self.fix_file and self.fix_file.read() and self.vuln_file and self.vuln_file.read():
            vuln_lines = self.vuln_file.read().splitlines()
            fix_lines = self.fix_file.read().splitlines()

            diff = '\n'.join(difflib.context_diff(vuln_lines, fix_lines, fromfile=str(self.fix_file.short),
                                                tofile=str(self.fix_file.short)))

            match = re.search("\*\*\* (\d+,\d+) \*\*\*\*", diff)
            size_vuln_lines = len(vuln_lines)
            vuln_str = match_snippet(match, vuln_lines)

            match = re.search("--- (\d+,\d+) ----", diff)
            size_fix_lines = len(fix_lines)
            fix_str = match_snippet(match, fix_lines)

            if self.real:
                sequence_matcher = difflib.SequenceMatcher(None, vuln_lines, fix_lines)
                chunks = ['\n'.join(vuln_lines[a: a+size]) for a, b, size in sequence_matcher.get_matching_blocks()]

        elif self.non_vuln_file:
            # here, the row contains only non_vuln_files info
            non_vuln_lines = self.non_vuln_file.read().splitlines()
            size_non_vuln = len(non_vuln_lines)
            size_vuln_lines = random.randint(0, round(size_non_vuln * 0.05))
            size_fix_lines = random.randint(0, round(size_non_vuln * 0.25))

        if self.non_vuln_file and self.non_vuln_file.read():
            # Size of non vuln snippet equal to the mean of the vuln/fix sizes
            size = round(statistics.mean([size_fix_lines, size_vuln_lines]))
            non_vuln_lines = self.non_vuln_file.read().splitlines()
            size_non_vuln = len(non_vuln_lines)

            if size_non_vuln > size:
                start = random.randint(0, size_non_vuln - size)
            elif size_non_vuln > size_vuln_lines:
                start = random.randint(0, size_non_vuln - size_vuln_lines)
            elif size_non_vuln > size_fix_lines:
                start = random.randint(0, size_non_vuln - size_fix_lines)
            else:
                start = random.randint(0, round(size_non_vuln/2))

            non_vuln_str = '\n'.join(non_vuln_lines[start: start+size])

        return vuln_str, fix_str, non_vuln_str, chunks

    def get_contents(self):
        if self.non_vuln_file:
            if not self.vuln_file:
                return None, None, self.non_vuln_file.read()
            return self.vuln_file.read(), self.fix_file.read(), self.non_vuln_file.read()

        return self.vuln_file.read(), self.fix_file.read(), None
