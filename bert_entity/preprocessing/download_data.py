import subprocess
from typing import Dict

from pipeline_job import PipelineJob


class DownloadWikiDump(PipelineJob):
    """
    Download the current Wikipedia dump. Either download one file for a dummy / prototyping version
    (set download_data_only_dummy to True). Or all download files.
    """
    def __init__(self, preprocess_jobs: Dict[str, PipelineJob], opts):
        super().__init__(
            requires=[],
            provides=[
                f"data/versions/{opts.data_version_name}/downloads/{opts.wiki_lang_version}/"
            ],
            preprocess_jobs=preprocess_jobs,
            opts=opts,
        )

    def _run(self):

        self.log(f"Downloading {self.opts.wiki_lang_version}")
        if self.opts.download_data_only_dummy:
            subprocess.check_call(
                [
                    "wget",
                    "-r",
                    "-l1",
                    "-np",
                    "-nd",
                    "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles1.xml-p1p30303.bz2",
                    "-P",
                    f"data/versions/{self.opts.data_version_name}/downloads/{self.opts.wiki_lang_version}/",
                ]
            )
        else:
            subprocess.check_call(
                [
                    "wget",
                    "-r",
                    "-l1",
                    "-np",
                    "-nd",
                    f"https://dumps.wikimedia.org/{self.opts.wiki_lang_version}/latest/",
                    "-A",
                    f"{self.opts.wiki_lang_version}-latest-pages-articles*.xml-*.bz2",
                    "-R",
                    f"{self.opts.wiki_lang_version}-latest-pages-articles-multistream*.xml-*.bz2",
                    "-P",
                    f"data/versions/{self.opts.data_version_name}/downloads/{self.opts.wiki_lang_version}/",
                ]
            )
        self.log("Download finished ")
