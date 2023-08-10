# Slurm Job Resource Monitor

A python command line tool to monitor the resources of your slurm jobs in form of a regularly updated table.

![Demo Image](https://github.com/tmachnitzki/slurm_job_resource_monitor/blob/main/demo.jpg)

## Installation

Install via pip:

```bash
pip install slurm-job-resource-monitor
```

## Usage

On a slurm managed cluster, run the following command to monitor your jobs:

```bash
slurm_job_monitor
```

To get all the possible options run:

```bash
slurm_job_monitor --help
```

The preferred way to use the tool is with the group-by-cmd option to not spam the table when running multi-cpu jobs:

```bash
slurm_job_monitor --group-by-cmd "sum"
```

## Assumptions / Preliminary

The tool assumes that:

- you are running on a slurm managed cluster.
- The tool assumes that you have a slurm account.
- You can ssh in all allocated notes.
- For ssh within the cluster no password is needed (only ssh key).
- You need nvidia-smi to display GPU info of nodes (optional).
- You can run "ps auxfww" to display CPU info of nodes.

## Some Further Info

Only processes are displayed that have a CPU load > 0.0 $ as shown by "ps auxfww". This is to avoid
the spamming of the table with processes that are not doing anything.

The table is grouped by jobs. So if one job runs on multiple GPUs and nodes they will all be displayed
in the same "section" of row. Sections are separated by a horizontal line.

There is a minimal width that the terminal needs to have to display the table correctly. The only column that
allows wrapping is the "Command" column.
