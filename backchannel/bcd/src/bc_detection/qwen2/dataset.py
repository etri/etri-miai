"""
Qwen2-Audio용 데이터셋

기존 BackchannelDataset을 수정하지 않고, Qwen2-Audio 추론/학습에 맞는
별도 데이터셋 클래스를 제공한다.

기존 매니페스트 파일(.label, .speaker.text, .listener.text)을 읽어
mixing_audios/{sample_id}.wav를 로드하는 구조이다.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, TypedDict, Union

import librosa
import numpy as np

from .constants import AUDIO_SAMPLE_RATE

logger = logging.getLogger(__name__)


class Qwen2AudioSample(TypedDict):
    """Qwen2-Audio용 샘플 구조"""

    sample_id: str
    mixed_audio_path: str
    speaker_audio_path: str
    listener_audio_path: str
    label: str
    speaker_text: str
    listener_text: str


class Qwen2AudioDataset:
    """
    Qwen2-Audio용 데이터셋

    기존 매니페스트 파일들을 읽어 mixing_audios/ 기반 샘플을 제공한다.
    기존 BackchannelDataset/BackchannelDataModule은 수정하지 않는다.
    """

    DEFAULT_LABEL_MAPPING = {
        "ALONE": "NOBC",
        "BARGIN": "NOBC",
        "CONTINUER": "CONTINUER",
        "UNDERSTANDING": "UNDERSTANDING",
        "ASSESSMENT": "EMPATHY",
        "AFFIRMATIVE": "EMPATHY",
        "NEGATIVE_SURPRISE": "EMPATHY",
        "POSITIVE_SURPRISE": "EMPATHY",
        "REQUEST_CONFIRMATION": "EMPATHY",
    }

    def __init__(
        self,
        manifest_path: Union[str, Path],
        audio_root: Union[str, Path],
        label_mapping: Optional[Dict[str, str]] = None,
        mixing_audio_dir: str = "mixing_audios",
        speaker_audio_dir: str = "speaker_audios",
        listener_audio_dir: str = "listener_audios",
        audio_input_mode: str = "mixed",
        validate_audio: bool = False,
        crop_speaker_sec: Optional[float] = None,
        crop_listener_sec: Optional[float] = None,
        silence_speaker: bool = False,
        silence_listener: bool = False,
    ):
        """
        Args:
            manifest_path: 매니페스트 base 경로 (예: "splits/train")
            audio_root: 데이터 루트 디렉토리
            label_mapping: 9→4 클래스 매핑 (None이면 기본값 사용)
            mixing_audio_dir: 믹싱 오디오 디렉토리 이름
            speaker_audio_dir: 화자 오디오 디렉토리 이름
            listener_audio_dir: 청자 오디오 디렉토리 이름
            audio_input_mode: "mixed" 또는 "separated"
            validate_audio: True면 초기화 시 오디오 파일 존재 확인
            crop_speaker_sec: speaker 오디오 끝에서 N초만 사용 (None이면 전체)
            crop_listener_sec: listener 오디오 끝에서 N초만 사용 (None이면 전체)
            silence_speaker: True면 speaker audio를 0으로 치환 (control 실험용)
            silence_listener: True면 listener audio를 0으로 치환 (control 실험용)
        """
        if audio_input_mode not in ("mixed", "separated"):
            raise ValueError(
                f"audio_input_mode은 'mixed' 또는 'separated'여야 합니다: {audio_input_mode}"
            )

        self.manifest_path = Path(manifest_path)
        self.audio_root = Path(audio_root)
        self.label_mapping = label_mapping or self.DEFAULT_LABEL_MAPPING
        self.mixing_audio_dir = mixing_audio_dir
        self.speaker_audio_dir = speaker_audio_dir
        self.listener_audio_dir = listener_audio_dir
        self.audio_input_mode = audio_input_mode
        self.crop_speaker_sec = crop_speaker_sec
        self.crop_listener_sec = crop_listener_sec
        self.silence_speaker = silence_speaker
        self.silence_listener = silence_listener

        self.samples: List[Qwen2AudioSample] = self._load_manifest()

        if validate_audio:
            self._validate_audio_files()

        logger.info(f"Qwen2AudioDataset 로드 완료: {len(self.samples)}개 샘플")

    def _load_manifest(self) -> List[Qwen2AudioSample]:
        """매니페스트 파일들을 로드하여 샘플 리스트를 생성한다."""
        label_file = Path(f"{self.manifest_path}.label")
        speaker_text_file = Path(f"{self.manifest_path}.speaker.text")
        listener_text_file = Path(f"{self.manifest_path}.listener.text")

        for f in [label_file, speaker_text_file, listener_text_file]:
            if not f.exists():
                raise FileNotFoundError(f"매니페스트 파일을 찾을 수 없습니다: {f}")

        labels = self._load_file_data(label_file)
        speaker_texts = self._load_file_data(speaker_text_file)
        listener_texts = self._load_file_data(listener_text_file)

        common_ids = set(labels.keys()) & set(speaker_texts.keys()) & set(listener_texts.keys())
        if not common_ids:
            raise ValueError("공통 샘플 ID를 찾을 수 없습니다")

        samples = []
        skipped = 0
        for sample_id in sorted(common_ids):
            raw_label = labels[sample_id].strip()

            if raw_label not in self.label_mapping:
                logger.warning(f"알 수 없는 레이블 건너뜀: {sample_id} → {raw_label}")
                skipped += 1
                continue

            mixed_audio_path = str(
                self.audio_root / self.mixing_audio_dir / f"{sample_id}.wav"
            )
            speaker_audio_path = str(
                self.audio_root / self.speaker_audio_dir / f"{sample_id}.wav"
            )
            listener_audio_path = str(
                self.audio_root / self.listener_audio_dir / f"{sample_id}.wav"
            )

            samples.append(
                Qwen2AudioSample(
                    sample_id=sample_id,
                    mixed_audio_path=mixed_audio_path,
                    speaker_audio_path=speaker_audio_path,
                    listener_audio_path=listener_audio_path,
                    label=self.label_mapping[raw_label],
                    speaker_text=speaker_texts[sample_id].strip(),
                    listener_text=listener_texts[sample_id].strip(),
                )
            )

        if skipped > 0:
            logger.warning(f"알 수 없는 레이블로 {skipped}개 샘플 건너뜀")

        return samples

    def _load_file_data(self, file_path: Path) -> Dict[str, str]:
        """단일 매니페스트 파일을 로드한다. 형식: sample_id value"""
        data = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    data[parts[0]] = parts[1]
                elif len(parts) == 1:
                    data[parts[0]] = ""
        return data

    def _validate_audio_files(self):
        """모든 샘플의 오디오 파일 존재 여부를 확인한다."""
        missing = []
        for sample in self.samples:
            for path in self.get_audio_paths(sample):
                if not Path(path).exists():
                    missing.append(f"{sample['sample_id']}:{Path(path).parent.name}")

        if missing:
            logger.warning(
                f"오디오 파일 누락: {len(missing)}개 (처음 5개: {missing[:5]})"
            )

    def get_audio_paths(self, sample: Qwen2AudioSample) -> List[str]:
        """현재 audio_input_mode에 필요한 오디오 경로를 반환한다."""
        if self.audio_input_mode == "separated":
            return [sample["speaker_audio_path"], sample["listener_audio_path"]]
        return [sample["mixed_audio_path"]]

    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        오디오 파일을 로드한다.

        Args:
            audio_path: wav 파일 경로

        Returns:
            numpy float32 배열 (16kHz)
        """
        audio, _ = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE)
        return audio

    def _crop_tail(self, audio: np.ndarray, target_sec: float) -> np.ndarray:
        target_n = int(round(target_sec * AUDIO_SAMPLE_RATE))
        if target_n <= 0:
            return np.zeros(0, dtype=np.float32)
        if len(audio) >= target_n:
            return audio[-target_n:]
        pad = target_n - len(audio)
        return np.concatenate([np.zeros(pad, dtype=audio.dtype), audio])

    def load_audio_for_role(self, audio_path: str, role: str) -> np.ndarray:
        """role-aware loader: applies crop/silence flags based on speaker/listener/mixed.

        role ∈ {"speaker","listener","mixed"}
        """
        if role == "speaker" and self.silence_speaker:
            n_sec = self.crop_speaker_sec if self.crop_speaker_sec else 1.5
            return np.zeros(int(round(n_sec * AUDIO_SAMPLE_RATE)), dtype=np.float32)
        if role == "listener" and self.silence_listener:
            n_sec = self.crop_listener_sec if self.crop_listener_sec else 1.5
            return np.zeros(int(round(n_sec * AUDIO_SAMPLE_RATE)), dtype=np.float32)

        audio = self.load_audio(audio_path)
        if role == "speaker" and self.crop_speaker_sec is not None:
            audio = self._crop_tail(audio, self.crop_speaker_sec)
        elif role == "listener" and self.crop_listener_sec is not None:
            audio = self._crop_tail(audio, self.crop_listener_sec)
        return audio

    def get_audio_paths_with_roles(self, sample: Qwen2AudioSample):
        """Returns list of (audio_path, role) for current audio_input_mode."""
        if self.audio_input_mode == "separated":
            return [
                (sample["speaker_audio_path"], "speaker"),
                (sample["listener_audio_path"], "listener"),
            ]
        return [(sample["mixed_audio_path"], "mixed")]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Qwen2AudioSample:
        return self.samples[idx]

    def get_class_distribution(self) -> Dict[str, int]:
        """클래스 분포를 반환한다."""
        dist: Dict[str, int] = {}
        for sample in self.samples:
            label = sample["label"]
            dist[label] = dist.get(label, 0) + 1
        return dict(sorted(dist.items()))

    def get_class_labels(self) -> List[str]:
        """정렬된 클래스 레이블 목록을 반환한다."""
        return sorted(set(s["label"] for s in self.samples))
