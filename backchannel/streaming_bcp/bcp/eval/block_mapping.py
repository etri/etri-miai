from pathlib import Path

from bcp.eval.blocks import BlockBound
from bcp.eval.resource import Resource

##### CONSTANT #####
BLOCKMAP_PATH = Path("bcp/eval/config/mapping.yaml")

####################


class BlockMapper:
    def __init__(self, locator, target_infos):

        # block mapping이 필요한 경우에 bcp/eval/config/mapping.yaml 파일을 읽어서 cfg 저장한다.
        # 없을 경우에는 None으로 설정한다.

        self.cfg = Resource.load_config_file(BLOCKMAP_PATH)

        # cfg priority를 dataset_name에 맞게 설정한다.
        self.priority = self.cfg["priority"][locator.dataset_name]

        # 매핑할 블록 정보를 config 형태로 변환한다.
        self.transform_block_infos(target_infos)

        self.utt2nums = self.get_utt2num(locator.utt2num_path())
        self.ref_block_bounds = self.get_block_bound(name="reference")
        self.tgt_block_bounds = self.get_block_bound(name="target")

        self.stat_viewer()
        self.assert_block_bounds()

    def assert_block_bounds(self):
        # block 정보를 제외한 stft, conv 정보는 동일해야 한다.
        # 2024년 11월 6일 기준으로 지원하지 않음

        assert (
            self.ref_block_bounds.stft == self.tgt_block_bounds.stft
        ), "stft 정보가 다릅니다."
        assert (
            self.ref_block_bounds.conv == self.tgt_block_bounds.conv
        ), "conv 정보가 다릅니다."

    def get_utt2num(self, path):

        utt2nums = Resource.load_file_for_label(path)

        # utt2nums의 value가 list 크기가 1인 형태이므로 int로 변환한다.
        for key in utt2nums:
            utt2nums[key] = int(utt2nums[key][0])

        return utt2nums

    def get_block_bound(self, name="reference"):

        block_bound = BlockBound(
            block_size=self.cfg[name]["block"]["block_size"],
            previous_size=self.cfg[name]["block"]["previous_size"],
            current_size=self.cfg[name]["block"]["current_size"],
            lookahead_size=self.cfg[name]["block"]["lookahead_size"],
            hop_size=self.cfg[name]["block"]["hop_size"],
            stft_kwargs=self.cfg[name]["stft"],
            conv_kwargs=self.cfg[name]["conv"],
        )

        return block_bound

    def transform_block_infos(self, block_infos):
        const_conv_info = {"name": "conv2d", "kernel": [3, 3], "stride": [2, 2]}

        self.cfg.setdefault("target", {})
        self.cfg["target"]["name"] = "{}_{}_{}_{}_{}".format(
            block_infos["stft"]["win_len"],
            block_infos["stft"]["hop_len"],
            block_infos["block"]["previous_size"],
            block_infos["block"]["current_size"],
            block_infos["block"]["lookahead_size"],
        )

        block_infos["conv"] = const_conv_info

        self.cfg["target"].update(block_infos)

        pass

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
        new_labels = []

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
                    # 아니면 priority 순으로 우선순위를 매김
                    max_v = max(count.values())
                    max_k = [k for k, v in count.items() if v == max_v]
                    if len(max_k) == 1:
                        label = max_k[0]
                    else:
                        for p in self.priority:
                            if p in max_k:
                                label = p
                                break
                        else:
                            label = max_k[0]
            new_labels.append(label)

        return new_labels

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

    def get_mapping_labels(self, labels):
        """
        block_bounds를 사용하여 매핑 레이블을 생성하는 함수이다.

        매핑 레이블은 voting 함수를 사용하여 생성된다.

        매개변수:
            labels (str): target_key.

        반환값:
            utt_mapping_labels (dict): 매핑 레이블 딕셔너리.
        """

        assert len(labels) > 0, "labels의 길이가 0입니다."

        utt_mapping_labels = {}

        for utt_key in labels:
            org_labels = labels[utt_key]

            ref_blocks = self.ref_block_bounds.blocks(0, self.utt2nums[utt_key])
            tgt_blocks = self.tgt_block_bounds.blocks(0, self.utt2nums[utt_key])

            mapping_blocks = self.mapping_blocks(ref_blocks, tgt_blocks)
            new_labels = self.voting(org_labels, mapping_blocks)

            utt_mapping_labels.setdefault(utt_key, new_labels)

        return utt_mapping_labels

    def stat_viewer(self):
        print(f"*** Reference Block -> Target Block ***")

        for key in [
            "block_size",
            "previous_size",
            "current_size",
            "lookahead_size",
            "hop_size",
        ]:
            ref_key = self.cfg["reference"]["block"][key]
            tgt_key = self.cfg["target"]["block"][key]
            print(f"  * {key}: {ref_key} -> {tgt_key}")

        print(f"****************************************")
