#!/usr/bin/env python

import argparse
import os, sys 
import subprocess as sp
import time
import random
import uuid


SLURM_SUBMIT_GAP= 0.5
ZFASTVS_BIN_DPATH = os.path.dirname(__file__)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tasklist", default="tasklist", 
                        help="tasklist file name")
    parser.add_argument("-q", "--queue", default="normal", 
                        help="the slurm queue name. Default is normal.")
    parser.add_argument("-b", "--batch", default=8, type=int, 
                        help="batch number. Default is 8.")
    parser.add_argument("-c", "--cpu", default=16, type=int, 
                        help="per-slurm task CPU number. Default is 16.")
    parser.add_argument("-m", "--max", default=300, type=int,
                        help="Max number of slurm tasks. Default is 300.")    
    parser.add_argument("-p", "--parallel", default=0, type=int,
                        help="Wheather run in all parallel. Default is 0.")
    parser.add_argument("-g", "--gpu", default=0, type=int,
                        help="Wheather run with gpu. Default is 0.")
    parser.add_argument("-r", "--ratio", default=0.5, type=float,
                        help="Ratio for normal queue. Default is 0.5.")
    args = parser.parse_args()

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit()
    
    return args


def get_slurm_tasks(queue="general", taskid="general"):
    '''if queue != "other":
        #cmd = ["squeue", "|", "grep", f"{queue}", "|", "grep", "zheng", "|", "wc", "-l"]
        cmd = "squeue | grep {} | grep zheng | wc -l".format(queue)
    else:
        #cmd = ["squeue", "|", "grep", "zheng", "|", "wc", "-l"] #.format(queue)
        cmd = "squeue | grep zheng | wc -l" #.format(queue)'''
    cmd = f"squeue | grep {taskid} | wc -l"
    print("Running cmd: ", cmd)   

    try:
        ntasks = int(sp.check_output(cmd, shell=True, timeout=5))
    except:
        ntasks = 500
  
    return ntasks


def slurm_submit(cmd, cpu=4, taskid="general", queue="other", use_gpu=0, ratio=0.5):
    if queue == "other":
        r = random.random()
        # print("[INFO]: random number {:.4f}".format(r))
        if r >= ratio:
            queue = "genmsa"
        else:
            queue = "normal"
   
    if use_gpu > 0:
        command = "{}/submit_slurm_gpu.sh \"{}\" {} {}".format(ZFASTVS_BIN_DPATH, cmd, cpu, queue)
    else:
        command = "{}/submit_slurm_taskid.sh \"{}\" {} {} {}".format(ZFASTVS_BIN_DPATH, cmd, cpu, taskid, queue)

    job = sp.Popen(command, shell=True)
    job.communicate()


if __name__ == "__main__":
    args = arguments()

    taskid = "ZLZ" + str(uuid.uuid4().hex)[:4]

    # get tasks list
    with open(args.tasklist) as lines:
        tasklist = ["time " + x.strip("\n") for x in lines]
    random.shuffle(tasklist)

    # prepare cmd
    batch_size = args.batch
    cmd_list = []
    for i, t in enumerate(tasklist):
        cmd_list.append(t)
        if i > 0 and i % batch_size == 0:
            print("[INFO]: example cmd {}".format(cmd_list[0]))
            if args.parallel:
                cmd = " & ".join(cmd_list) + " && wait && date"
            else:
                cmd = " && ".join(cmd_list)

            ntasks = get_slurm_tasks(taskid=taskid)
            while ntasks >= args.max:
                time.sleep(15)
                print("[INFO]: number of slurm tasks {}, sleep 15s".format(ntasks))
                ntasks = get_slurm_tasks(taskid=taskid)

            # submit task now 
            slurm_submit(cmd, taskid=taskid, cpu=args.cpu, queue=args.queue, ratio=args.ratio)
            print("[INFO]: processing task {}".format(i))
            time.sleep(SLURM_SUBMIT_GAP)

            cmd_list = []

    cmd = " && ".join(cmd_list)
    # submit task now 
    slurm_submit(cmd, args.cpu, args.queue)

    # waiting for tasks to complete
    ntasks = get_slurm_tasks(taskid=taskid)
            while ntasks >= args.max:
                time.sleep(15)
                print("[INFO]: number of slurm tasks {}, sleep 15s".format(ntasks))
                ntasks = get_slurm_tasks(taskid=taskid)
