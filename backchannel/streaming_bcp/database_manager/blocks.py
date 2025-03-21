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
                                start_lookahead,
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
                                start_lookahead,
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
