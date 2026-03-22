# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import argparse
from pathlib import Path

import datasets

from nemo_skills.dataset.utils import save_jsonl

HF_DATASET = "CohereLabs/m-ArenaHard"
SUPPORTED_LANGUAGES = datasets.get_dataset_config_names(HF_DATASET)


def format_entry(row: dict, language: str) -> dict:
    return {
        "question": row["prompt"],
        "question_id": row["question_id"],
        "category": row["category"],
        "cluster": row["cluster"],
        "language": language,
        "subset_for_metrics": language,
    }


def main(args):
    invalid_langs = set(args.languages) - set(SUPPORTED_LANGUAGES)
    if invalid_langs:
        raise ValueError(f"Unsupported languages: {invalid_langs}. Supported: {SUPPORTED_LANGUAGES}")

    data_dir = Path(__file__).absolute().parent
    output_file = data_dir / "test.jsonl"

    all_entries = []
    for language in args.languages:
        print(f"Processing {language}...")
        ds = datasets.load_dataset(HF_DATASET, name=language, split="test")
        for row in ds:
            all_entries.append(format_entry(row, language))

    save_jsonl(all_entries, output_file)
    print(f"Saved {len(all_entries)} entries to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare m-ArenaHard multilingual benchmark.")
    parser.add_argument(
        "--languages",
        default=SUPPORTED_LANGUAGES,
        nargs="+",
        help=f"Languages to include. Supported: {SUPPORTED_LANGUAGES}",
    )
    args = parser.parse_args()
    main(args)
