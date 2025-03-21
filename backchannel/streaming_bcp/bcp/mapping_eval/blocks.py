"""
training data의 block 관련 처리 코드
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Block:
    """
    Utterance가 모델에 입력되었을 때, block 단위로 처리됨.
    하나의 block 은 previous, current, lookahead로 나뉨
    그 각각의 영역 경계가 되는 raw sample index를 담은 dataclass
    """

    total: tuple
    previous: tuple
    current: tuple
    lookahead: tuple


# class AbsBound:
#     """
#     moving window 를 가정,
#     어떤 win length를 가지고, hop length를 가지고 데이터를 처리할 때,
#     src_sequence ==> dst_sequence
#     dst_sequence의 index를 주면, src_squence에서 시작 지점/종료 지점을 알려줌
#     """

#     def start(self, index):
#         raise NotImplementedError

#     def end(self, index):
#         raise NotImplementedError

#     def length(self, sample_length):
#         raise NotImplementedError


# class STFTFBound(AbsBound):
#     """
#     center=True 일때 librosa stft
#     """

#     def __init__(self, n_fft=512, hop_len=128):
#         self.n_fft = n_fft
#         self.hop_len = hop_len

#         self.lpad = n_fft // 2
#         self.rpad = n_fft - self.lpad

#     def start(self, idx):
#         """
#         stft 결과 반환된 sequence의 index를 주면, 그에 해당하는 원본 signal의 시작 index를 줌
#         e.g., 0 => -256
#         """
#         return (idx * self.hop_len) - self.lpad

#     def end(self, idx):
#         """
#         stft 결과 반환된 sequence의 index를 주면, 그에 해당하는 원본 signal의 마지막 index를 줌
#         e.g., 0 => 255
#         """
#         return self.start(idx) + self.n_fft - 1

#     def length(self, sample_len):
#         """
#         stft 결과 반환될 sequence의 길이
#         """
#         return (sample_len // self.hop_len) + 1


# class ConvBound(AbsBound):
#     """
#     torch.Conv2D의 conv 과정
#     """

#     def __init__(self, kernel=3, stride=2, dilation=1):
#         self.kernel = kernel
#         self.stride = stride
#         self.dilation = dilation  # default option in torch.Conv2D

#     def start(self, idx):
#         return idx * self.stride

#     def end(self, idx):
#         return self.start(idx) + self.kernel - 1

#     def length(self, sample_len):
#         """torch.Conv2D 참조"""
#         return ((sample_len - (1 * (self.kernel - 1)) - 1) // self.stride) + 1


# class BlockBound:
#     def __init__(
#         self,
#         block_size=40,
#         previous_size=8,
#         current_size=16,
#         lookahead_size=16,
#         hop_size=8,
#         stft_kwargs={"n_fft": 512, "hop_len": 128},
#         conv_kwargs={"kernel": [3, 3], "stride": [2, 2]},
#     ):
#         self.block = block_size
#         self.previous = previous_size
#         self.current = current_size
#         self.lookahead = lookahead_size
#         self.hop = hop_size

#         self.stftb = STFTFBound(**stft_kwargs)
#         self.convbs = [
#             ConvBound(kernel=k, stride=s)
#             for k, s in zip(conv_kwargs["kernel"], conv_kwargs["stride"])
#         ]

#     def start(self, idx):
#         """
#         convolution 결과 반환된 sequence의 index를 주면, 해당하는 raw audio sample의 시작 index 반환
#         """
#         i = idx
#         for bound in [*list(reversed(self.convbs)), self.stftb]:
#             i = bound.start(i)
#         return i

#     def end(self, idx):
#         """
#         convolution 결과 반환된 sequence의 index를 주면, 해당하는 raw audio sample의 마지막 index 반환 (포함관계임)
#         """
#         i = idx
#         for bound in [*list(reversed(self.convbs)), self.stftb]:
#             i = bound.end(i)
#         return i

#     def blocks(self, start, end):
#         # block 갯수 구하기
#         # *Bound 로 처리 결과 bound를 구하는 class들은 모두 0 시작을 기준으로 작성됨.
#         # 따라서, 결과를 구한 후, 주어진 start를 더해서 시작 지점을 보정해주어야 함.

#         n = end - start
#         for bound in [self.stftb, *self.convbs]:
#             n = bound.length(n)

#         # 주어진 conv 출력 갯수를 바탕으로, block 구성
#         blocks = list()
#         for i in range(0, n, self.hop):
#             block = Block(
#                 total=(i, i + self.block - 1),
#                 previous=(i, i + self.previous - 1),
#                 current=(
#                     i + self.previous,
#                     i + self.previous + self.current - 1,
#                 ),
#                 lookahead=(
#                     i + self.previous + self.current,
#                     i + self.block - 1,
#                 ),
#             )
#             # feature가 1개라도 있으면, block을 생성함.
#             if block.total[1] + 1 >= n + self.hop:
#                 break
#             blocks.append(block)

#         # 원본 오디오 signal의 sample index로 변환
#         # start index를 더해서 오디오 파일의 구성과 일치시킴
#         for block in blocks:
#             block.total = (
#                 start + self.start(block.total[0]),
#                 start + self.end(block.total[1]),
#             )
#             block.previous = (
#                 start + self.start(block.previous[0]),
#                 start + self.end(block.previous[1]),
#             )
#             block.current = (
#                 start + self.start(block.current[0]),
#                 start + self.end(block.current[1]),
#             )
#             block.lookahead = (
#                 start + self.start(block.lookahead[0]),
#                 start + self.end(block.lookahead[1]),
#             )

#         #

#         return blocks


class BlockBound:
    def __init__(
        self,
        block_size=40,
        previous_size=8,
        current_size=16,
        lookahead_size=16,
        hop_size=8,
        stft_kwargs={"n_fft": 512, "hop_len": 128, "center": True},
        conv_kwargs={"kernel": [3, 3], "stride": [2, 2], "name": "conv2d"},
    ):
        assert (
            current_size == hop_size
        ), f"현재 Streaming TF 모델은 hop size와 current size가 같아야 합니다. {current_size} != {hop_size}"
        assert block_size == (
            previous_size + current_size + lookahead_size
        ), f"block size는 previous + current + lookahead의 합과 같아야 합니다. {block_size} != {previous_size} + {current_size} + {lookahead_size}"

        self.block = block_size
        self.previous = previous_size
        self.current = current_size
        self.lookahead = lookahead_size
        self.hop = hop_size

        self.stft = stft_kwargs
        self.conv = conv_kwargs

        self._subsample = self.get_subsample(self.conv["name"])
        self._sample_per_after_downsampling = (
            self.stft["n_fft"] - self.stft["hop_len"]
        ) + (self.stft["hop_len"] * ((self._subsample * 2) - 1))
        self._hop_sample_per_after_downsampling = self.stft["hop_len"] * self._subsample

        self._residual_part = (
            self._sample_per_after_downsampling
            - self._hop_sample_per_after_downsampling
        )

        # 각 block의 start sample 계산
        self._start_current = self._hop_sample_per_after_downsampling * self.previous
        self._start_lookahead = self._hop_sample_per_after_downsampling * (
            self.previous + self.current
        )

        # 각 block의 sample 계산
        self._sample_per_block = self._residual_part + (
            self._hop_sample_per_after_downsampling * self.block
        )
        self._previous_sample_per_block = self._residual_part + (
            self._hop_sample_per_after_downsampling * self.previous
        )
        self._current_sample_per_block = self._residual_part + (
            self._hop_sample_per_after_downsampling * self.current
        )
        self._lookahead_sample_per_block = self._residual_part + (
            self._hop_sample_per_after_downsampling * self.lookahead
        )
        self._current_sample_per_first_block = self._residual_part + (
            self._hop_sample_per_after_downsampling * (self.previous + self.current)
        )
        self._hop_sample_per_block = self._hop_sample_per_after_downsampling * self.hop

        self._pad_start = int(-1 * stft_kwargs["n_fft"] // 2)

    def get_subsample(self, name):
        subsample_map = {
            "linear": 1,
            "conv2d": 4,
            "conv2d6": 6,
            "conv2d8": 8,
            "embed": 1,
            None: 1,
        }
        if name not in subsample_map:
            raise ValueError(f"unknown input_layer: {name}")
        return subsample_map[name]

    def get_num_samples(self, n):
        if self.stft["center"]:
            return n + (self.stft["n_fft"] // 2)
        return n

    def blocks(self, start, end):
        n = end - start
        n_pad = self.get_num_samples(n)
        blocks = []

        if n_pad <= self._sample_per_block:
            blocks.append(
                Block(
                    total=(0, n - 1),
                    previous=(0, 0),
                    current=(0, n - 1),
                    lookahead=(n, n),
                )
            )
        else:
            for block_num, start_sample in enumerate(
                range(self._pad_start, n_pad, self._hop_sample_per_block),
                start=1,
            ):
                end_sample = min(n, start_sample + self._sample_per_block)
                is_first_block = block_num == 1
                is_last_block = (
                    end_sample >= n_pad
                    or end_sample == n
                    or n - end_sample < self._sample_per_after_downsampling
                )

                if (
                    is_first_block
                    and n - end_sample < self._sample_per_after_downsampling
                ):
                    blocks.append(
                        Block(
                            total=(0, n - 1),
                            previous=(0, 0),
                            current=(0, n - 1),
                            lookahead=(n, n),
                        )
                    )
                    break
                elif is_first_block:
                    start_lookahead = start_sample + self._start_lookahead
                    blocks.append(
                        Block(
                            total=(0, end_sample - 1),
                            previous=(0, 0),
                            current=(
                                0,
                                start_sample + self._current_sample_per_first_block - 1,
                            ),
                            lookahead=(
                                (
                                    start_lookahead
                                    if self.lookahead > 0
                                    else start_lookahead
                                    + self._lookahead_sample_per_block
                                    - 1
                                ),
                                start_lookahead + self._lookahead_sample_per_block - 1,
                            ),
                        )
                    )
                elif is_last_block:
                    blocks.append(
                        Block(
                            total=(start_sample, n - 1),
                            previous=(
                                start_sample,
                                start_sample + self._previous_sample_per_block - 1,
                            ),
                            current=(start_sample + self._start_current, n - 1),
                            lookahead=(n, n),
                        )
                    )
                    break
                else:
                    start_current = start_sample + self._start_current
                    start_lookahead = start_sample + self._start_lookahead
                    blocks.append(
                        Block(
                            total=(start_sample, end_sample - 1),
                            previous=(
                                start_sample,
                                start_sample + self._previous_sample_per_block - 1,
                            ),
                            current=(
                                start_current,
                                start_current + self._current_sample_per_block - 1,
                            ),
                            lookahead=(
                                (
                                    start_lookahead
                                    if self.lookahead > 0
                                    else start_lookahead
                                    + self._lookahead_sample_per_block
                                    - 1
                                ),
                                start_lookahead + self._lookahead_sample_per_block - 1,
                            ),
                        )
                    )

        # 원본 오디오 signal의 sample index로 변환
        # start index를 더해서 오디오 파일의 구성과 일치시킴
        for block in blocks:
            block.total = (
                start + block.total[0],
                start + block.total[1],
            )
            block.previous = (
                start + block.previous[0],
                start + block.previous[1],
            )
            block.current = (
                start + block.current[0],
                start + block.current[1],
            )
            block.lookahead = (
                start + block.lookahead[0],
                start + block.lookahead[1],
            )

        #

        return blocks


if __name__ == "__main__":
    b = BlockBound(
        block_size=40,
        previous_size=36,
        current_size=4,
        lookahead_size=0,
        hop_size=4,
        stft_kwargs={"n_fft": 320, "hop_len": 160, "center": True},
        conv_kwargs={"kernel": [3, 3], "stride": [2, 2], "name": "conv2d"},
    )

    print(b.blocks(0, 48000))
