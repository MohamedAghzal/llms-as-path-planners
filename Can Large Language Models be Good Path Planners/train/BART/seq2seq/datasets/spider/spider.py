# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Spider: A Large-Scale Human-Labeled Dataset for Text-to-SQL Tasks"""


import json
from typing import List, Generator, Any, Dict, Tuple
from third_party.spider.preprocess.get_tables import dump_db_json_schema
import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{yu2018spider,
  title={Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-sql task},
  author={Yu, Tao and Zhang, Rui and Yang, Kai and Yasunaga, Michihiro and Wang, Dongxu and Li, Zifan and Ma, James and Li, Irene and Yao, Qingning and Roman, Shanelle and others},
  journal={arXiv preprint arXiv:1809.08887},
  year={2018}
}
"""

_DESCRIPTION = """\
Spider is a large-scale complex and cross-domain semantic parsing and text-toSQL dataset annotated by 11 college students
"""

_HOMEPAGE = "https://yale-lily.github.io/spider"

_LICENSE = "CC BY-SA 4.0"

_URL = "https://drive.google.com/uc?export=download&id=1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0"


class Spider(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="spider",
            version=VERSION,
            description="Spider: A Large-Scale Human-Labeled Dataset for Text-to-SQL Tasks",
        ),
    ]

    def __init__(self, *args, writer_batch_size=None, **kwargs) -> None:
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
        self.schema_cache = dict()
        self.include_train_others: bool = kwargs.pop("include_train_others", False)

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features(
            {
                "target": datasets.Value("string"),
                "question": datasets.Value("string")
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        downloaded_filepath = dl_manager.download_and_extract(_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_filepaths": [
                       "./transformers_cache/multigoal_train_updated.json ",
                    ]
                    if self.include_train_others
                    else ["./transformers_cache/multigoal_train_updated.json",]
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_filepaths": ["./transformers_cache/multigoal_dev_updated.json",]
                },
            ),
        ]

    def _generate_examples(
        self, data_filepaths: List[str]
    ) -> Generator[Tuple[int, Dict[str, Any]], None, None]:
        """This function returns the examples in the raw (text) form."""
        for data_filepath in data_filepaths:
            logger.info("generating examples from = %s", data_filepath)
            with open(data_filepath, encoding="utf-8") as f:
                spider = json.load(f)
                for idx, sample in enumerate(spider):
                    yield idx, {
                        "target": sample["target"],
                        "question": sample["question"]
                    }
