"""This is a command line tool to monitor the status of all your slurm jobs.

It uses rich to display the status of all jobs in a table. It will ssh into all allocated nodes and get CPU and GPU
usage
"""

import getpass
import subprocess
import time
from datetime import datetime as dt
from typing import Optional

import pandas as pd
import paramiko
from rich import get_console, print
from rich.table import Table


def get_slurm_job_info_dict(job_id: str) -> dict:
    """Get the job info from slurm."""
    job_info = subprocess.run(["scontrol", "show", "job", job_id, "-o", "-d"], capture_output=True, text=True)
    job_info = job_info.stdout.split(" ")
    job_info = [x.split("=") for x in job_info if "=" in x]
    job_info = {x[0]: x[1] for x in job_info}
    return job_info


def get_gpu_usage_of_current_node(this_node: paramiko.SSHClient) -> Optional[dict]:
    """Get the GPU usage and memory usage of all GPUs seperated by process ID on the current node, using nvidia-smi.

    Args:
        this_node (paramiko.SSHClient): SSHClient object of the current node.

    Returns:
        dict: Nested dictionary with GPU usage of all GPUs on the current node.
            {PID: {GPU_ID: {mem: memory usage, command: command, sm: compute usage, enc: encoder usage,
            dec: decoder usage}}}
    """
    gpu_usage = this_node.exec_command("nvidia-smi pmon -c 1")[1]
    gpu_usage = gpu_usage.read().decode("utf-8")
    gpu_usage = gpu_usage.split("\n")
    gpu_usage = [x.split() for x in gpu_usage if x != ""]
    gpu_keys = gpu_usage.pop(0)[1:]
    gpu_usage = gpu_usage[1:]
    # if not gpu_usage:
    #     return None
    gpu_df = pd.DataFrame(gpu_usage, columns=gpu_keys)

    gpu_return = {}
    for pid in gpu_df["pid"].unique():
        if pid == "-":
            continue
        gpu_return[pid] = {}
        for i, gpu in enumerate(gpu_df[gpu_df["pid"] == pid]["gpu"].tolist()):
            gpu_return[pid][gpu] = {}
            gpu_return[pid][gpu]["mem"] = gpu_df[gpu_df["pid"] == pid]["mem"].tolist()[i]
            gpu_return[pid][gpu]["command"] = gpu_df[gpu_df["pid"] == pid]["command"].tolist()[i]
            gpu_return[pid][gpu]["sm"] = gpu_df[gpu_df["pid"] == pid]["sm"].tolist()[i]
            gpu_return[pid][gpu]["enc"] = gpu_df[gpu_df["pid"] == pid]["enc"].tolist()[i]
            gpu_return[pid][gpu]["dec"] = gpu_df[gpu_df["pid"] == pid]["dec"].tolist()[i]

    if gpu_return == {}:
        gpu_return = None
    return gpu_return


def get_cpu_usage_of_current_node(user_id: str, this_node: paramiko.SSHClient) -> dict:
    """Get the CPU usage of all CPUs on the current node.

    Args:
        user_id (str): User ID of the user to filter for.

    Returns:
        dict: Nested dictionary with CPU usage of all CPUs on the current node. Outer keys are the CPU IDs, inner keys
        are the CPU usage metrics (%CPU, USER, %MEM).
    """
    # use ps to get the CPU usage of all processes
    cpu_usage = this_node.exec_command(f"ps auxfww | egrep {user_id}")[1]
    cpu_usage = cpu_usage.read().decode("utf-8")

    # get keys:
    cpu_keys = this_node.exec_command("ps auxfww")[1]
    cpu_keys = cpu_keys.read().decode("utf-8")

    cpu_keys = cpu_keys.split("\n")[0]
    cpu_keys = cpu_keys.split()

    cpu_usage = cpu_usage.split("\n")
    cpu_usage = [x.split() for x in cpu_usage if x != ""]

    return_usage = {}
    for usage in cpu_usage:
        return_usage[usage[1]] = {}

        # on the last key combine all remaining values into one
        for key in cpu_keys:
            if key == "COMMAND":
                return_usage[usage[1]][key] = " ".join(usage[cpu_keys.index(key) :]).replace("\\_ ", "").strip()
            else:
                return_usage[usage[1]][key] = usage[cpu_keys.index(key)]

    # filter out processes that have a load of 0.0
    return_usage = {k: v for k, v in return_usage.items() if float(v["%CPU"]) > 0.0}

    # again filter out processes that are not the user's (leftovers could exist due to jobs with names that contain the
    # user's name)
    return_usage = {k: v for k, v in return_usage.items() if v["USER"] == user_id}

    # if command starts with |, then remove the | from the command
    for k, v in return_usage.items():
        if v["COMMAND"].startswith("|"):
            return_usage[k]["COMMAND"] = v["COMMAND"][1:].strip()

    return return_usage


def get_all_jobs_by_user(user_id: str) -> dict:
    """Get all jobs of a user from slurm.

    Args:
        user_id (str): The user id of the user whose jobs you want to get.

    Returns:
        dict: A nested dict containing all jobs of the user. The outer dict has the job id as key and the inner dict
            contains the job info using slurm's format heads as keys.
    """
    all_jobs = subprocess.run(
        ["squeue", "-u", user_id, "--format", "%.40i %.40P %.40j %.40u %.40T %.40M %.40l %.40D %R"],
        capture_output=True,
        text=True,
    )
    all_jobs = all_jobs.stdout.split("\n")
    all_jobs = [x for x in all_jobs if x != ""]
    all_jobs = [x.split() for x in all_jobs if " " in x]

    # in x0 only the names are stored so we use these as dict keys
    dict_keys = all_jobs.pop(0)

    return_jobs = {}
    for job in all_jobs:
        return_jobs[job[0]] = {k: v for k, v in zip(dict_keys, job)}
    return return_jobs


def create_rich_table(all_job_dict):
    """
    Create a rich table Like the following:
    +------------+-----------------------------------------------------+
    | SlurmJobID | This_ID                                             |
    +============+=====================================================+
    | Node | GPU | PID | COMMAND | CPU_USAGE | CPU_MEM | GPU_USAGE | GPU_MEM |
    +------+-----+---------+-----------+---------+-----------+---------+
    |      |     |         |           |         |           |         |
    ...


    Args:
        all_job_dict (dict): The job dict containing all the job info.
    """
    now = dt.now().strftime("%H:%M:%S")
    table = Table(title=f"Slurm Job Usage Monitor - {now}")

    table.add_column("SlurmJobID", justify="left", style="cyan", no_wrap=True)
    table.add_column("Node", justify="center", style="cyan", no_wrap=True)
    table.add_column("GPU", justify="center", style="cyan", no_wrap=True)
    table.add_column("PID", justify="center", style="cyan", no_wrap=True)
    table.add_column("COMMAND", justify="left", style="cyan", no_wrap=False)
    table.add_column("CPU_USAGE", justify="center", style="cyan", no_wrap=True)
    table.add_column("CPU_MEM", justify="center", style="cyan", no_wrap=True)
    table.add_column("GPU_USAGE", justify="center", style="cyan", no_wrap=True)
    table.add_column("GPU_MEM", justify="center", style="cyan", no_wrap=True)

    for job_id, job_dict in all_job_dict.items():
        table.add_section()
        for node in job_dict["Nodes"]:
            node = node.split("(")[0]
            for pid in job_dict[node]["CPU usage"]:
                gpu_usage = "N/A"
                gpu_mem = "N/A"
                gpu_id = "N/A"
                if job_dict[node]["GPU usage"] is not None:
                    if pid in job_dict[node]["GPU usage"]:
                        for gpu_id in job_dict[node]["GPU usage"][pid].keys():
                            gpu_usage = job_dict[node]["GPU usage"][pid][gpu_id]["sm"]
                            gpu_mem = job_dict[node]["GPU usage"][pid][gpu_id]["mem"]

                            table.add_row(
                                job_id,
                                node,
                                gpu_id,
                                pid,
                                job_dict[node]["CPU usage"][pid]["COMMAND"],
                                job_dict[node]["CPU usage"][pid]["%CPU"] + " %",
                                job_dict[node]["CPU usage"][pid]["%MEM"] + " %",
                                gpu_usage + " %",
                                gpu_mem + " %",
                            )
                    else:
                        table.add_row(
                            job_id,
                            node,
                            gpu_id,
                            pid,
                            job_dict[node]["CPU usage"][pid]["COMMAND"],
                            job_dict[node]["CPU usage"][pid]["%CPU"] + " %",
                            job_dict[node]["CPU usage"][pid]["%MEM"] + " %",
                            gpu_usage,
                            gpu_mem,
                        )

    return table


def main():
    """Main function."""
    all_jobs = get_all_jobs_by_user(getpass.getuser())

    # only keep jobs that are running
    all_jobs = {k: v for k, v in all_jobs.items() if v["STATE"] == "RUNNING"}

    # if no jobs are running print a message and exit
    if len(all_jobs) == 0:
        print("No jobs are running.")

    # get the job info from slurm
    slurm_infos = {}
    for job_id in all_jobs.keys():
        slurm_infos[job_id] = get_slurm_job_info_dict(job_id)

    # for each job, get the nodes that it is running on
    for job_id, job_info in all_jobs.items():
        all_jobs[job_id]["Nodes"] = job_info["NODELIST(REASON)"].split(",")

        # for each node get the CPU and GPU usage
        for node in all_jobs[job_id]["Nodes"]:
            all_jobs[job_id][node] = {}

            # ssh into the node
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(node, username=getpass.getuser())

            # get the CPU usage
            cpu_usage = get_cpu_usage_of_current_node(getpass.getuser(), ssh)
            all_jobs[job_id][node]["CPU usage"] = cpu_usage

            # get the GPU usage
            gpu_usage = get_gpu_usage_of_current_node(ssh)
            all_jobs[job_id][node]["GPU usage"] = gpu_usage

            # close the ssh connection
            ssh.close()

    # convert the dataframe to a rich table
    table = create_rich_table(all_jobs)

    console = get_console()
    console.clear()

    console.print(table)
    time.sleep(2)


def cli_entry():
    """Entry point for the CLI."""
    print("Starting")
    while True:
        main()


if __name__ == "__main__":
    print("Starting")
    while True:
        main()
