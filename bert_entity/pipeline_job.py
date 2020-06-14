import logging
import os
import urllib
import datetime
from argparse import Namespace
from typing import List, Any, Dict

class Job:

    def run(self, jobs: Dict[str, Any]):
        raise NotImplementedError

    def _get_msg(self, msg):
        return f"{datetime.datetime.now()}  [{self.__class__.__name__}] {msg}"

    def log(self, msg):
        logging.info(self._get_msg(msg))

    def error(self, msg):
        logging.error(self._get_msg(msg))

    def debug(self, msg):
        logging.debug(self._get_msg(msg))

    def _run(self):
        raise NotImplementedError


class PipelineJob(Job):
    def __init__(self, requires, provides, preprocess_jobs, opts: Namespace, rerun_job=False):

        self.requires: List[str] = requires
        self.provides: List[str] = provides
        self.add_provides(preprocess_jobs)
        self.opts = opts
        self.rerun_job = rerun_job

    def run(self, pipeline_jobs: Dict[str, Any]):
        self.log(f"Checking requirements for {self.__class__.__name__}")
        self.check_required_exist(pipeline_jobs)
        self.create_out_directories()
        if not self.provides_exists():
            self.log(f"Start running {self.__class__.__name__}")
            self._run()
            self.log(f"Finished running {self.__class__.__name__}")
        else:
            self.log(f"{self.__class__.__name__} is already finished")

    def _download(self, url, folder):
        if not os.path.exists(f"{folder}/{os.path.basename(url)}"):
            self.log(
                f"Downloading {url}"
            )
            urllib.request.urlretrieve(
                url,
                f"{folder}/{os.path.basename(url)}",
            )
            self.log("Download finished ")
        return f"{folder}/{os.path.basename(url)}"

    def add_provides(self, preprocess_jobs: Dict[str, Job]):
        for file_name in self.provides:
            preprocess_jobs[file_name] = self

    def check_required_exist(self, preprocess_jobs: Dict[str, Job]):
        for file_name in self.requires:
            if not os.path.exists(file_name):
                try:
                    preprocess_jobs[file_name].run(preprocess_jobs)
                except:
                    self.error(
                        f"Cannot find required {file_name} and there is no preprocess job to create it"
                    )
                    raise Exception

    def provides_exists(self,):
        if self.rerun_job:
            return False
        for file_name in self.provides:
            if not os.path.exists(file_name):
                return False
        return True

    def create_out_directories(self,):
        for file_name in self.provides:
            if len(file_name) - len(os.path.dirname(file_name)) in [0, 1]:
                os.makedirs(os.path.dirname(os.path.dirname(file_name)), exist_ok=True)
            else:
                os.makedirs(os.path.dirname(file_name), exist_ok=True)

    @staticmethod
    def run_jobs(job_classes: List, opts):
        jobs_dict = dict()
        job_list = list()
        for job_class in job_classes:
            job_list.append(job_class(jobs_dict, opts))
        for job in job_list:
            job.run(jobs_dict)
