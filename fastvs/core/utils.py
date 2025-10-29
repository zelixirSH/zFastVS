
import os, sys
import subprocess as sp
import json
import hashlib
import uuid

def encode_string(s):
    smi_hashid = hashlib.md5(s.encode()).hexdigest()

    return smi_hashid


def load_configs(configs_fpath: str) -> dict:
    """Load configuration file.

    Args:
        configs_fpath (str): config file, json format.

    Returns:
        dict: dictionary, the parameters.
    """
    with open(configs_fpath) as jsonfile:
        params = json.load(jsonfile)

    return params


def make_temp_dpath():

    tmp_dpath = f"/tmp/{str(uuid.uuid4().hex)[:6]}"
    os.makedirs(tmp_dpath, exist_ok=True)

    return tmp_dpath


def run_command(cmd, verbose=False):
    if len(cmd) == 0:
        return None
        
    if type(cmd) == list:
        if verbose:
            print(f"[INFO] Running cmd {''.join(cmd)}")
        job = sp.Popen(cmd)
    else:
        if verbose:
            print(f"[INFO] Running cmd {cmd}")
        job = sp.Popen(cmd, shell=True)

    job.communicate()

    return None


if __name__ == "__main__":

    string = "C1ccc$#331ck"

    print(encode_string(string))
    print(decode_string(encode_string(string)))