import re


def get_sline(pointers, idx, part, signal):
    return pointers[idx].split(' ')[part].split(',')[0].replace(signal, '')


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
