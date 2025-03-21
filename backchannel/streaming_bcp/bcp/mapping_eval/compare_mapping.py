"""
# 이 코드는 블록 수에 따른 매핑 결과를 확인하기 위한 테스트용 코드입니다.
# 실제 애플리케이션에 사용되는 코드는 아니며, 테스트 및 검증 목적으로 작성되었습니다.
"""

from pathlib import Path

import pandas as pd
from tabulate import tabulate
from tqdm.auto import tqdm

from bcp.mapping_eval.constant import Constant as C
from bcp.mapping_eval.blocks import BlockBound


class Resouce:
    FOLDER_PATH = Path("bcp/mapping_eval/debug")

    def __init__(self, name):
        self.name = name

        self.full_path = self.FOLDER_PATH / self.name

        assert self.exist_file(), f"File not found: {self.full_path}"

        self._utts = self.read_file()

        print(f"File found: {self.name} ({len(self._utts)} utts)")

    def exist_file(self):
        if self.full_path.exists():
            return True
        else:
            return False

    def read_file(self):
        utts = {}
        with open(self.full_path, "r") as f:
            for line in f:
                key, *bcs = line.strip().split()

                assert key not in utts, f"Key already exists: {key}"

                utts.setdefault(key, bcs)
        return utts

    @property
    def utts(self):
        return self._utts


class Viewer:
    def __init__(self, resources):
        self.resources = resources

    def get_view(self, utt_keys=[], size=None):
        assert len(utt_keys) > 0, "Please provide utt_keys"
        assert all([key in self.resources for key in utt_keys]), "Invalid utt_keys"
        assert "utt2num_samples" in self.resources, "utt2num_samples not found"

        infos = {
            "utt_key": [],
            "num_samples": [],
        }
        infos.update({key: [] for key in utt_keys})

        if isinstance(self.resources["utt2num_samples"], dict):
            for key in self.resources["utt2num_samples"]:
                infos["utt_key"].append(key)
                infos["num_samples"].append(self.resources["utt2num_samples"][key])
                for utt_key in utt_keys:
                    infos[utt_key].append(len(self.resources[utt_key][key]))
        else:
            for key in self.resources["utt2num_samples"].utts:
                infos["utt_key"].append(key)
                infos["num_samples"].append(self.resources["utt2num_samples"].utts[key])
                for utt_key in utt_keys:
                    infos[utt_key].append(len(self.resources[utt_key].utts[key]))

        df_infos = pd.DataFrame(infos)
        if size is not None:
            print(tabulate(df_infos[:size], headers="keys", tablefmt="pretty"))
        else:
            print(tabulate(df_infos, headers="keys", tablefmt="pretty"))


class MappingEvaluation:
    def __init__(self, resources, dataset="bckd"):
        self.resources = resources
        self.dataset = dataset

        self.utt2nums = self.get_utt2num()
        self.block_bounds = self.get_block_bound()

    def get_utt2num(self):
        assert "utt2num_samples" in self.resources, "utt2num_samples not found"

        utt2nums = {}

        for utt_key in self.resources["utt2num_samples"].utts:
            utt2nums.setdefault(
                utt_key, int(self.resources["utt2num_samples"].utts[utt_key][0])
            )

        return utt2nums

    def get_block_bound(self):
        """
        블록 경계 정보를 생성하는 함수이다.

        리소스 키를 기반으로 블록 경계 정보를 생성하여 반환한다.
        각 키는 'bc_label'로 시작하며, 키의 형식은 'bc_label.{n_fft}_{hop_len}_{previous_size}_{current_size}_{lookahead_size}'이다.
        이 함수는 각 키를 파싱하여 block size, previous size, current size, lookahead size, hop size 및 STFT 인자를 포함하는 BlockBound 객체를 생성.

        반환값:
            block_bounds (dict): 블록 경계 정보를 포함한 딕셔너리.
        """

        block_bounds = {}
        for key in self.resources:
            if key.startswith("bc_label"):
                n_fft, hop_len, previous_size, current_size, lookahead_size = (
                    int(x) for x in key.split(".")[1].split("_")
                )

                block_size = previous_size + current_size + lookahead_size
                block_hop_size = current_size

                block_bound = BlockBound(
                    block_size=block_size,
                    previous_size=previous_size,
                    current_size=current_size,
                    lookahead_size=lookahead_size,
                    hop_size=block_hop_size,
                    stft_kwargs={"n_fft": n_fft, "hop_len": hop_len, "center": True},
                )

                block_bounds.setdefault(key, block_bound)

        return block_bounds

    def voting(self, org_labels, mapping_blocks):
        """
        주어진 원본 레이블(org_labels)과 매핑 블록(mapping_blocks)을 사용하여 최종 레이블을 결정하는 함수이다.

        매핑 블록은 base_blocks와 target_blocks 간의 매핑 정보를 포함한다. 각 매핑 블록에 대해 원본 레이블을 참조하여 최종 레이블을 결정한다.
        만약 모든 후보 레이블이 "NOBC"인 경우, 최종 레이블은 "NOBC"로 설정된다.
        그렇지 않은 경우, 후보 레이블의 빈도를 계산하여 가장 빈도가 높은 레이블을 선택한다.
        만약 빈도가 동일한 레이블이 여러 개 있는 경우, 데이터셋(dataset)에 따라 우선순위(priority)를 적용하여 최종 레이블을 결정한다.

        매개변수:
            org_labels (list): 원본 레이블 리스트.
            mapping_blocks (dict): 매핑 블록 정보를 포함한 딕셔너리.
            dataset (str): 데이터셋 이름 ("bckd", "bbkd", "swbd" 중 하나).

        반환값:
            labels (list): 최종 레이블 리스트.
        """
        labels = []

        for m in mapping_blocks.values():
            candidate_labels = [org_labels[i] for i in m]

            if all([v == "NOBC" for v in candidate_labels]):
                label = "NOBC"
            else:
                count = {
                    v: candidate_labels.count(v)
                    for v in set(candidate_labels)
                    if v != "NOBC"
                }

                if len(count) == 1:
                    label = list(count.keys())[0]
                else:
                    # 최대값이 하나만 존재하면 그것을 선택
                    # 아니면 CONTINUER, UNDERSTANDING, EMPATHETIC 순으로 우선순위를 매김
                    max_v = max(count.values())
                    max_k = [k for k, v in count.items() if v == max_v]
                    if len(max_k) == 1:
                        label = max_k[0]
                    else:
                        if self.dataset in ["bckd", "bbkd"]:
                            priority = C.KO_PRIORITY
                        elif self.dataset in ["swbd"]:
                            priority = C.EN_PRIORITY
                        else:
                            raise ValueError(f"Invalid dataset: {self.dataset}")

                        for p in priority:
                            if p in max_k:
                                label = p
                                break
                        else:
                            label = max_k[0]
            labels.append(label)

        return labels

    def mapping_blocks(self, base_blocks, target_blocks):
        """
        base_blocks와 target_blocks 간의 매핑을 생성하는 함수이다.

        base_blocks와 target_blocks는 각각 원본 블록과 대상 블록의 리스트이다. 이 함수는 두 블록 리스트를 비교하여 매핑 정보를 생성한다.
        매핑 정보는 각 base_block의 인덱스를 키로 하고, 해당 base_block에 매핑되는 target_block의 인덱스 리스트를 값으로 하는 딕셔너리 형태로 반환된다.

        매개변수:
            base_blocks (list): 원본 블록 리스트.
            target_blocks (list): 대상 블록 리스트.

        반환값:
            mapping_blocks (dict): 매핑 정보를 포함한 딕셔너리.
        """
        mapping_blocks = {idx: [] for idx in range(len(base_blocks))}

        i, j = 0, 0

        while i < len(base_blocks) and j < len(target_blocks):
            # 블록의 current 시간 중에 끝나는 시간이 같은 경우
            if base_blocks[i].current[1] == target_blocks[j].current[1]:
                mapping_blocks[i].append(j)
                i += 1
                j += 1

            # 블록의 current 시간 중에 끝나는 시간이 base_blocks가 더 느린 경우
            elif base_blocks[i].current[1] > target_blocks[j].current[1]:
                mapping_blocks[i].append(j)
                j += 1

            # 블록의 current 시간 중에 끝나는 시간이 target_blocks가 더 느린 경우
            else:
                mapping_blocks[i].append(j)
                i += 1

        return mapping_blocks

    def get_mapping_labels(self, target_key, base_key):
        """
        주어진 target_key base_key를 사용하여 매핑 레이블을 생성하는 함수이다.

        매핑 레이블은 voting 함수를 사용하여 생성된다.

        매개변수:
            target_key (str): target_key.
            base_key (str): base_key.

        반환값:
            utt_mapping_labels (dict): 매핑 레이블 딕셔너리.
        """
        utt_mapping_labels = {}

        for utt_key in self.resources[target_key].utts:
            org_labels = self.resources[target_key].utts[utt_key]

            base_blocks = self.block_bounds[base_key].blocks(0, self.utt2nums[utt_key])
            target_blocks = self.block_bounds[target_key].blocks(
                0, self.utt2nums[utt_key]
            )

            mapping_blocks = self.mapping_blocks(base_blocks, target_blocks)
            labels = self.voting(org_labels, mapping_blocks)

            utt_mapping_labels.setdefault(utt_key, labels)

        return utt_mapping_labels


if __name__ == "__main__":
    files = [
        "utt2num_samples",
        "bc_label.512_128_8_16_16",
        "bc_label.512_128_8_16_0",
        "bc_label.512_128_16_16_0",
        "bc_label.512_128_24_16_0",
        "bc_label.512_128_32_8_0",
        "bc_label.512_128_36_4_0",
    ]

    resources = {}

    for name in files:
        res = Resouce(name)
        resources.setdefault(name, res)

    mapping_test = Viewer(resources)
    mapping_test.get_view(utt_keys=files[1:], size=20)

    mapping_evaluation = MappingEvaluation(resources, dataset="swbd")

    mapping_target_labels = {
        files[0]: resources[files[0]].utts,
        files[1]: resources[files[1]].utts,
    }

    for target_key in files[2:]:
        mapping_labels = mapping_evaluation.get_mapping_labels(
            target_key=target_key,
            base_key="bc_label.512_128_8_16_16",
        )
        mapping_target_labels.setdefault(target_key, mapping_labels)

    mapping_target_test = Viewer(mapping_target_labels)
    mapping_target_test.get_view(utt_keys=files[1:], size=20)
