"""This is a command line tool to monitor the status of all your slurm jobs.

It uses rich to display the status of all jobs in a table. It will ssh into all allocated nodes and get CPU and GPU
usage.
"""

import argparse
import getpass
import json
import subprocess
import time
from datetime import datetime as dt

import numpy as np
import pandas
import pandas as pd
import paramiko
from rich import get_console, print
from rich.table import Table


def get_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--username",
        type=str,
        default=getpass.getuser(),
        help="Username for ssh login. If not specified, the current username will be used.",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Job ID to monitor. If not specified, all jobs will be monitored.",
    )
    parser.add_argument("--refresh-rate", type=int, default=5, help="Refresh rate in seconds.")
    parser.add_argument("--gpu-only", action="store_true", help="Only show jobs that use GPUs.")
    parser.add_argument(
        "--group-by-cmd",
        type=str,
        default=None,
        help="Group jobs per node that have the same command. If not specified all jobs are displayed. "
        "Possible options are `mean`, `max`, `min`.",
    )
    parser.add_argument(
        "--min-cpu-usage",
        type=float,
        default=0.1,
        help="Minimum CPU usage processes have to have to be shown.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save all kinds of dictionaries to files to be inspected.",
    )
    return parser.parse_args()


def get_gpu_usage_of_current_node(this_node: paramiko.SSHClient) -> dict:
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


def get_cpu_usage_of_current_node(user_id: str, this_node: paramiko.SSHClient, min_cpu_usage: float = 0.1) -> dict:
    """Get the CPU usage of all CPUs on the current node.

    Args:
        user_id (str): User ID of the user to filter for.
        this_node (paramiko.SSHClient): SSHClient object of the current node.
        min_cpu_usage (float): Minimum CPU usage processes must have to be shown.

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
    return_usage = {k: v for k, v in return_usage.items() if float(v["%CPU"]) >= min_cpu_usage}

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
        [
            "squeue",
            "-u",
            user_id,
            "--format",
            "%.40i %.40P %.40j %.40u %.40T %.40M %.40l %.40D %R",
        ],
        capture_output=True,
        text=True,
    )
    all_jobs = all_jobs.stdout.split("\n")
    all_jobs = [x for x in all_jobs if x != ""]
    all_jobs = [x.split() for x in all_jobs if " " in x]

    # in x0 only the names are stored, so we use these as dict keys
    dict_keys = all_jobs.pop(0)

    return_jobs = {}
    for job in all_jobs:
        return_jobs[job[0]] = {k: v for k, v in zip(dict_keys, job)}
    return return_jobs


def create_rich_table(all_job_df: pd.DataFrame, display_gpu_only: bool = False) -> Table:
    """
    Create a rich table Like the following:

                                Slurm Job Usage Monitor - 15:33:20
    ┏━━━━━━━━━━━━┳━━━━━━┳━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┓
    ┃ SlurmJobID ┃ Node ┃ GPU ┃  PID   ┃ COMMAND          ┃ CPU_USAGE ┃ CPU_MEM ┃ GPU_USAGE ┃ GPU_MEM ┃
    ┡━━━━━━━━━━━━╇━━━━━━╇━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━┩
    │ 938559     │ g018 │  0  │ 37764  │ /gpfs/work/mach… │  99.4 %   │  0.3 %  │   99 %    │  93 %   │
    │            │      │     │        │ /gpfs/work/mach… │           │         │           │         │
    │ 938559     │ g018 │  1  │ 42223  │ /gpfs/work/mach… │  99.8 %   │  0.3 %  │   99 %    │  95 %   │
    │            │      │     │        │ /gpfs/work/mach… │           │         │           │         │
    │ 938559     │ g018 │ N/A │ 107399 │ bash -c ps       │   1.0 %   │  0.0 %  │    N/A    │   N/A   │
    │            │      │     │        │ auxfww | egrep   │           │         │           │         │
    │            │      │     │        │ machnitz         │           │         │           │         │
    ├────────────┼──────┼─────┼────────┼──────────────────┼───────────┼─────────┼───────────┼─────────┤
    │ 938441     │ g009 │  0  │ 151583 │ /gpfs/work/mach… │  59.3 %   │  0.4 %  │   99 %    │  90 %   │
    │            │      │     │        │ /gpfs/work/mach… │           │         │           │         │
    └────────────┴──────┴─────┴────────┴──────────────────┴───────────┴─────────┴───────────┴─────────┘


    Args:
        all_job_df (pd.Dataframe): The job df containing all the job info.
        display_gpu_only (bool, optional): If True, only jobs with gpu usage will be displayed. Defaults to False.
    """

    # convert all values to string with 0 decimal places
    all_job_df = all_job_df.applymap(lambda x: f"{x:.0f}" if isinstance(x, float) else x)

    now = dt.now().strftime("%H:%M:%S")
    table = Table(title=f"Slurm Job Usage Monitor - {now}")

    table.add_column("SLURM JOB ID", justify="left", style="cyan", no_wrap=True)
    table.add_column("Node", justify="center", style="cyan", no_wrap=True)
    table.add_column("GPU ID", justify="center", style="cyan", no_wrap=True)
    table.add_column("PID", justify="center", style="cyan", no_wrap=True)
    table.add_column("COMMAND", justify="left", style="cyan", no_wrap=False)
    table.add_column("CPU USAGE", justify="center", style="cyan", no_wrap=True)
    table.add_column("CPU MEM", justify="center", style="cyan", no_wrap=True)
    table.add_column("GPU USAGE", justify="center", style="cyan", no_wrap=True)
    table.add_column("GPU MEM", justify="center", style="cyan", no_wrap=True)

    last_node = "this_is_not_a_node"
    for _, row in all_job_df.iterrows():
        if display_gpu_only and row["gpu_usage"] == "N/A":
            continue
        if last_node != row["node"]:
            table.add_section()

        table.add_row(
            row["job_id"],
            row["node"],
            row["gpu_id"],
            row["pid"],
            row["command"],
            row["cpu_usage"] + " %" if row["cpu_usage"] != "N/A" else "N/A",
            row["cpu_mem"] + " %" if row["cpu_mem"] != "N/A" else "N/A",
            row["gpu_usage"] + " %" if row["gpu_usage"] != "N/A" else "N/A",
            row["gpu_mem"] + " %" if row["gpu_usage"] != "N/A" else "N/A",
        )
        last_node = row["node"]

    return table


def group_cpu_usage_by_command(all_job_df: pandas.DataFrame, reduction="mean") -> pandas.DataFrame:
    """Group the CPU usage by command and take the reduction of the CPU usage and CPU memory.

    Return:
        dict: A dictionary containing the grouped CPU usage in the same format as the all_job_dict.
    """

    # split the df into two dfs, one with gpu usage and one without
    gpu_usage_df = all_job_df[all_job_df["gpu_usage"] != "N/A"]
    no_gpu_usage_df = all_job_df[all_job_df["gpu_usage"] == "N/A"]

    grouped_cpu_usage = no_gpu_usage_df.groupby("command").agg(
        {
            "cpu_usage": reduction,
            "cpu_mem": reduction,
            "job_id": "first",
            "node": "first",
            "pid": "first",
        }
    )
    grouped_cpu_usage = grouped_cpu_usage.reset_index()
    grouped_cpu_usage = grouped_cpu_usage.sort_values(by=["job_id", "node"])

    # merge the two dfs again on job_id, node and pid
    grouped_cpu_usage = pd.merge(
        grouped_cpu_usage,
        gpu_usage_df,
        on=["job_id", "node", "command"],
        how="outer",
    )

    # make df nice again:
    output_df = pd.DataFrame(columns=all_job_df.columns)
    for idx, row in grouped_cpu_usage.iterrows():
        out_row = {}
        if not np.isfinite(row["gpu_usage"]):
            out_row["gpu_usage"] = "N/A"
            out_row["gpu_mem"] = "N/A"
            out_row["gpu_id"] = "N/A"
            out_row["cpu_usage"] = row["cpu_usage_x"]
            out_row["cpu_mem"] = row["cpu_mem_x"]
            out_row["pid"] = row["pid_x"]

        else:
            out_row["gpu_usage"] = row["gpu_usage"]
            out_row["gpu_mem"] = row["gpu_mem"]
            out_row["gpu_id"] = row["gpu_id"]
            out_row["cpu_usage"] = row["cpu_usage_y"]
            out_row["cpu_mem"] = row["cpu_mem_y"]
            out_row["pid"] = row["pid_y"]

        out_row["job_id"] = row["job_id"]
        out_row["node"] = row["node"]
        out_row["command"] = row["command"]
        output_df = pd.concat([output_df, pd.DataFrame(out_row, index=[0])])

    output_df = output_df.sort_values(by=["job_id", "node"])

    return output_df


def dump_json(content: dict, filename: str):
    with open(filename, "w") as f:
        json.dump(content, f, indent=4)


def dict_to_df(all_job_dict: dict) -> pandas.DataFrame:
    # create pandas dataframes for CPU and GPU usage
    cpu_usage_df = pd.DataFrame()
    gpu_usage_df = pd.DataFrame()

    for job_id, job_dict in all_job_dict.items():
        for node in job_dict["Nodes"]:
            node = node.split("(")[0]
            for pid in job_dict[node]["CPU usage"]:
                cpu_usage_df = pd.concat(
                    [
                        cpu_usage_df,
                        pd.DataFrame(
                            {
                                "job_id": job_id,
                                "node": node,
                                "pid": pid,
                                "command": job_dict[node]["CPU usage"][pid]["COMMAND"],
                                "cpu_usage": float(job_dict[node]["CPU usage"][pid]["%CPU"]),
                                "cpu_mem": float(job_dict[node]["CPU usage"][pid]["%MEM"]),
                            },
                            index=[0],
                        ),
                    ],
                    ignore_index=True,
                )
                if job_dict[node]["GPU usage"] is not None:
                    if pid in job_dict[node]["GPU usage"]:
                        for gpu_id in job_dict[node]["GPU usage"][pid].keys():
                            gpu_usage_df = pd.concat(
                                [
                                    gpu_usage_df,
                                    pd.DataFrame(
                                        {
                                            "job_id": job_id,
                                            "node": node,
                                            "pid": pid,
                                            "gpu_id": gpu_id,
                                            "gpu_usage": float(job_dict[node]["GPU usage"][pid][gpu_id]["sm"]),
                                            "gpu_mem": float(job_dict[node]["GPU usage"][pid][gpu_id]["mem"]),
                                        },
                                        index=[0],
                                    ),
                                ],
                                ignore_index=True,
                            )

    # merge the two dataframes by job_id, node, pid
    merged_df = cpu_usage_df.merge(gpu_usage_df, on=["job_id", "node", "pid"], how="left")
    sorted_df = merged_df.sort_values(by=["job_id", "node", "pid"])

    # replace NaN values with "N/A"
    sorted_df = sorted_df.fillna("N/A")

    return sorted_df


def main(
    username,
    job_id,
    display_gpu_only,
    group_by_cmd,
    min_cpu_usage,
    refresh_rate,
    debug_mode,
):
    """Main function."""

    all_jobs = get_all_jobs_by_user(username)

    if debug_mode:
        dump_json(all_jobs, "all_jobs.json")

    # if a job id is given, only keep that job
    if job_id is not None:
        all_jobs = {k: v for k, v in all_jobs.items() if k == job_id}
        if len(all_jobs) == 0:
            print(f"No job with id {job_id} found.")
            time.sleep(refresh_rate)

    if debug_mode:
        dump_json(all_jobs, "all_jobs.json")

    # only keep jobs that are running
    all_jobs = {k: v for k, v in all_jobs.items() if v["STATE"] == "RUNNING"}

    if debug_mode:
        dump_json(all_jobs, "all_jobs.json")

    # if no jobs are running print a message and exit
    if len(all_jobs) == 0:
        print("No jobs are running.")
        time.sleep(refresh_rate)
        return

    # for each job, get the nodes that it is running on
    for job_id, job_info in all_jobs.items():
        all_jobs[job_id]["Nodes"] = job_info["NODELIST(REASON)"].split(",")

        # for each node get the CPU and GPU usage
        for node in all_jobs[job_id]["Nodes"]:
            all_jobs[job_id][node] = {}

            # ssh into the node
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(node, username=username)

            # get the CPU usage
            cpu_usage = get_cpu_usage_of_current_node(username, ssh, min_cpu_usage=min_cpu_usage)
            all_jobs[job_id][node]["CPU usage"] = cpu_usage

            # get the GPU usage
            gpu_usage = get_gpu_usage_of_current_node(ssh)
            all_jobs[job_id][node]["GPU usage"] = gpu_usage

            # close the ssh connection
            ssh.close()

    if debug_mode:
        dump_json(all_jobs, "all_jobs.json")

    # convert the dict to a dataframe
    all_jobs = dict_to_df(all_jobs)

    # group the CPU usage by command
    if group_by_cmd is not None:
        all_jobs = group_cpu_usage_by_command(all_jobs, reduction=group_by_cmd)

    if debug_mode:
        dump_json(all_jobs.to_dict(), "all_jobs.json")

    # convert the dataframe to a rich table
    table = create_rich_table(all_jobs, display_gpu_only=display_gpu_only)

    console = get_console()
    console.clear()

    console.print(table)
    time.sleep(refresh_rate)


def cli_entry():
    """Entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Monitor the CPU and GPU usage of your slurm jobs.")
    args = get_args(parser)
    username = args.username
    job_id = args.job_id
    refresh_rate = args.refresh_rate
    display_gpu_only = args.gpu_only
    group_by_cmd = args.group_by_cmd
    min_cpu_usage = args.min_cpu_usage
    debug_mode = args.debug

    trial_counter = 0
    print("Starting")
    while True:
        try:
            main(
                username=username,
                job_id=job_id,
                display_gpu_only=display_gpu_only,
                group_by_cmd=group_by_cmd,
                min_cpu_usage=min_cpu_usage,
                refresh_rate=refresh_rate,
                debug_mode=debug_mode,
            )
        except KeyboardInterrupt:
            print("Exiting")
            break
        except Exception as exc:
            trial_counter += 1
            print(f"Trial {trial_counter} / 10 failed. Retrying in {refresh_rate} seconds.")
            time.sleep(refresh_rate)

            if trial_counter >= 10:
                print("Too many failed trials. Exiting.")
                raise exc

            continue

        trial_counter = 0


if __name__ == "__main__":
    cli_entry()
