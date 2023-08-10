import json
import unittest

import pandas as pd

from slurm_job_resource_monitor.job_monitor import create_rich_table, dict_to_df, group_cpu_usage_by_command


class TestJobMonitor(unittest.TestCase):
    def test_convert_to_df(self):
        with open("./all_jobs_test_file.json") as f:
            all_jobs = json.load(f)

        df = dict_to_df(all_jobs)
        self.assertIsInstance(df, pd.DataFrame)

    def test_group_by_cmd(self):
        with open("./all_jobs_test_file.json") as f:
            all_jobs = json.load(f)

        all_jobs = dict_to_df(all_jobs)
        grouped_jobs = group_cpu_usage_by_command(all_jobs)
        self.assertEqual(2, len(grouped_jobs))

        # Check that grouped_jobs and all_jobs have the same columns
        self.assertEqual(set(grouped_jobs.columns), set(all_jobs.columns))

    def test_create_table(self):
        with open("./all_jobs_test_file.json") as f:
            all_jobs = json.load(f)

        all_jobs = dict_to_df(all_jobs)
        create_rich_table(
            all_jobs,
        )
