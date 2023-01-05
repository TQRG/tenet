import re
import io
import os
import tarfile
import json
import numpy as np
import pandas as pd

from binascii import b2a_hex
from os import urandom
from pathlib import Path

from tqdm import tqdm


def load_json_file(file):
    """Load JSON file that stores vulnerability data.
    Args:
        file (file): vulnerability report file name
    Returns:
        data: json data
    """
    with open(file) as jfile:
        return json.load(jfile)

def create_df(data):
    """Create dataframe from vuln data.
    Args:
        data (dict): dictionary with vulnerability data
    Returns:
        dataframe: dataframe with vulnerability data
    """
    return pd.DataFrame(data, index=[0])


def random_id(size: int = 2):
    """
        Generates random id of specified size.
    """
    return b2a_hex(urandom(size)).decode()


def df_init(info):
    return pd.DataFrame(info, index=[0])


def split_next(files):
    return files.split('<_**next**_>')


def str_to_tarfile(data: str, tar_info_name: str) -> Path:
    random = b2a_hex(urandom(2)).decode()
    dest_path = Path('/tmp', random + ".tar")

    info = tarfile.TarInfo(name=tar_info_name)
    info.size = len(data)

    with tarfile.TarFile(str(dest_path), 'w') as tar:
        tar.addfile(info, io.BytesIO(data.encode('utf8')))

    return dest_path


def count_labels(path: Path, kind: str):
    with path.open(mode='r') as df:
        safe = 0
        unsafe = 0
        for line in df.readlines():
            if line.startswith('safe'):
                safe += 1
            else:
                unsafe += 1

    print(f'=== {kind} dataset ===\nsafe:{safe}\nunsafe:{unsafe}')


def split_commits(chain: str):
    """ Normalizes the chain of commits for each CVE
        e.g., from link1,link2 to {link1, link2}.

    Args:
        chain (string): chain of commits
    Returns:
        new_chain: set of commits normalized
    """

    new_chain = set()

    for ref in eval(chain):
        if 'http://' in ref:
            protocol = 'http://'
        else:
            protocol = 'https://'

        count = ref.count(protocol)

        if count > 1:
            if ',' in ref:
                new_chain = set.union(new_chain, set([r for r in ref.split(',')]))
            else:
                new_chain = set.union(new_chain, set([f"{protocol}{r}" for r in ref.split(protocol)]))
        else:
            new_chain = set.union(new_chain, set([ref]))

    return new_chain if new_chain else np.nan


def filter_references(df: pd.DataFrame, refs_col: str = 'refs') -> pd.DataFrame:
    """
        Filter cases without any references to source code hosting websites (GitHub, BitBucket, GitLab, Git).

        :param df: pandas DataFrame with cases
        :param refs_col: name of the column with references

        :returns: pandas DataFrame with extra column ('code_refs') for code references
    """
    # get references to source code hosting websites
    for idx, row in tqdm(df.iterrows()):
        commits = []
        for ref in row[refs_col]:
            found = re.search(r'(github|bitbucket|gitlab|git).*(/commit/|/commits/)', ref)
            if found:
                commits.append(ref)
        if commits:
            df.at[idx, 'code_refs'] = str(set(commits))

    return df.dropna(subset=['code_refs'])


def get_source(refs: str) -> list:
    """
        Get source for each reference.

        :param refs: set with references in string format
    """
    refs, sources = eval(refs), []

    for ref in refs:
        if 'github' in ref:
            sources.append('github')
        elif 'bitbucket' in ref:
            sources.append('bitbucket')
        elif 'gitlab' in ref:
            sources.append('gitlab')
        elif 'git' in ref:
            sources.append('git')
        else:
            sources.append('other')

    return sources


def normalize_commits(df: pd.DataFrame, code_refs_col: str = 'code_refs'):
    """
        Normalize commits references
    """

    # iterate over each row
    for idx, row in df.iterrows():

        # get references and initialize output
        refs, commits = eval(row[code_refs_col]), []

        # iterate over references
        for ref in refs:
            # e.g., https://github.com/{owner}/{repo}/commit/{sha}CONFIRM:
            if "CONFIRM:" in ref:
                commits.append(ref.replace("CONFIRM:", ''))
            # e.g., https://github.com/intelliants/subrion/commits/develop
            # e.g., https://gitlab.gnome.org/GNOME/gthumb/commits/master/extensions/cairo_io/cairo-image-surface-jpeg.c
            # e.g., https://github.com/{owner}/{repo}/commits/{branch}
            elif not re.search(r"\b[0-9a-f]{5,40}\b", ref):
                continue
            elif 'git://' in ref and 'github.com' in ref:
                commits.append(ref.replace('git://', 'https://'))
            # e.g., https://github.com/{owner}/{repo}/commits/master?after={sha}+{no_commits}
            elif '/master?' in ref:
                continue
            # e.g., https://github.com/{owner}/{repo}/commit/{sha}#commitcomment-{id}
            elif '#' in ref and ('#comments' in ref or '#commitcomment' in ref):
                commits.append(ref.split('#')[0])
            # e.g., https://github.com/{owner}/{repo}/commit/{sha}.patch
            elif '.patch' in ref:
                commits.append(ref.replace('.patch', ''))
            # e.g., https://github.com/absolunet/kafe/commit/c644c798bfcdc1b0bbb1f0ca59e2e2664ff3fdd0%23diff
            # -f0f4b5b19ad46588ae9d7dc1889f681252b0698a4ead3a77b7c7d127ee657857
            elif '%23' in ref:
                commits.append(ref.replace('%23', '#'))
            else:
                commits.append(ref)

        # save new set of commits
        if commits:
            df.at[idx, code_refs_col] = set(commits)

    # drop row with no code refs after normalization
    return df.dropna(subset=[code_refs_col])


def filter_commits_by_source(df: pd.DataFrame, source: str, commits_col: str = 'commits',
                             code_refs_col: str = 'code_refs') -> pd.DataFrame:
    """Infer commits source (e.g., git, github, bitbucket, gitlab, etc).
    Args:
        df: pandas DataFrame with commit data
        code_refs_col: name of the column with references for code
        commits_col: name of the column with commits
        source (string): Git, GitHub, BitBucket, GitLab
    """

    # get commits from source
    for idx, row in df.iterrows():
        refs, commits = row[code_refs_col], []

        if isinstance(refs, str):
            refs = eval(refs)

        for ref in refs:
            if source in ref:
                commits.append(ref)

        if commits:
            df.at[idx, commits_col] = str(set(commits))
        else:
            df.at[idx, commits_col] = np.nan

    # drop rows without commits for source
    return df.dropna(subset=[commits_col])


def transform_to_commits(df):
    new_df = pd.DataFrame()

    for _, row in df.iterrows():
        chain = list(row["chain"])
        patch_type = "MULTI" if len(chain) > 1 else "SINGLE"

        for i, _ in enumerate(chain):
            row["commit_href"] = chain[i]
            row["project"] = "/".join(chain[i].split("/")[0:5])
            sha = re.sub(r"http(s)?://github.com/.*/commit(s)?/", "", chain[i])

            # FIXME: one of the source vulns still has the href in the commit_sha column when it reaches here.
            #       For some reason this is not fixed in the normalization phase. Find why!
            sha = sha.split('/')[-1] if 'http' in sha else sha

            row["commit_sha"] = sha
            row["patch"] = patch_type
            new_df = new_df.append(row, ignore_index=True)

    return new_df


def split_github_commits(chain):
    new_chain = set()
    for ref in eval(chain):
        if "http://" in ref:
            protocol = "http://"
        else:
            protocol = "https://"

        count = ref.count(protocol)
        if count > 1:
            if "," in ref:
                new_chain = set.union(
                    new_chain, set([r for r in ref.split(",") if "github.com" in r])
                )
            else:
                new_chain = set.union(
                    new_chain,
                    set(
                        [
                            f"{protocol}{r}"
                            for r in ref.split(protocol)
                            if "github.com" in r
                        ]
                    ),
                )
        else:
            new_chain = set.union(new_chain, set([ref]))
    return new_chain if len(new_chain) > 0 else np.nan


def clean_github_commits(refs):
    chain = set()

    for ref in refs:
        # FIXME: github links to diffs are not considered for now; looking into a solution for it e.g.,
        #  https://github.com/WBCE/WBCE_CMS/commit/0da620016aec17ac2d2f3a22c55ab8c2b55e691e#diff
        #  -7b380285e285160d0070863099baabe0
        if "#diff" in ref:
            continue
        # e.g., https://github.com/SerenityOS/serenity/commit/c841012f569dba4fa72e9eb8989bb847be4535bc
        # :889e1d3db9fe19197a4d22c9bfb2e67b3937a0c5
        if re.match(r".*github.com.*:.*", ref):
            continue
        # clean files and functions lines
        # e.g., https://github.com/liblime/LibLime-Koha/commit/8ea6f7bc37d05a9ec25b5afbea011cf9de5f1e49#C4/Output.pm
        elif "#" in ref:
            chain.add(ref.split("#")[0])
        # clean commits when the commit key is not available
        # e.g., https://github.com/nats-io/nats-server/commits/master
        elif not re.match(re.compile(r".*[0-9a-f]{6,40}.*"), ref):
            continue
        elif "?w=1" in ref:
            chain.add(ref.replace("?w=1", ""))
        elif "?branch=" in ref:
            chain.add(ref.split("?branch=")[0])
        elif "?diff=split" in ref:
            chain.add(ref.replace("?diff=split", ""))
        elif re.match(r".*(,|/)$", ref):
            if "/" in ref:
                chain.add(ref[0:-1])
            else:
                chain.add(ref.replace(",", ""))
        elif ")" in ref:
            chain.add(ref.replace(")", ""))
        else:
            chain.add(ref)

    return chain if chain else np.nan


def project_from_chain(refs):
    proj = None

    for ref in refs:
        proj = re.split("/commit/|/commits/", ref)[0]

        if "pull" in proj:
            proj = re.split("/pull/", proj)[0]

    return proj


def parse_published_date(datetime):
    return datetime.split("T")[0]


def load_file(fin):
    with open(fin) as file:
        return file.read().splitlines()
