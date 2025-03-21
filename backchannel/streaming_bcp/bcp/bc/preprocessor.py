from abc import ABC, abstractmethod
from pathlib import Path
from typing import Collection, Dict, Iterable, List, Union

import numpy as np
from typeguard import check_argument_types, check_return_type

from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.cleaner import TextCleaner
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.text.whisper_token_id_converter import OpenAIWhisperTokenIDConverter
from espnet2.train.preprocessor import CommonPreprocessor

import bcp.const.const as C


class ClassificationPreprocessor(CommonPreprocessor):

    def __init__(
        self,
        train: bool,
        token_type: str = None,
        token_list: Union[Path, str, Iterable[str]] = None,
        bpemodel: Union[Path, str, Iterable[str]] = None,
        text_cleaner: Collection[str] = None,
        g2p_type: str = None,
        unk_symbol: str = "<unk>",
        space_symbol: str = "<space>",
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        delimiter: str = None,
        rir_scp: str = None,
        rir_apply_prob: float = 1.0,
        noise_scp: str = None,
        noise_apply_prob: float = 1.0,
        noise_db_range: str = "3_10",
        short_noise_thres: float = 0.5,
        aux_task_names: Collection[str] = None,
        speech_volume_normalize: float = None,
        speech_name: str = "speech",
        text_name: str = "text",
        fs: int = 0,
        nonsplit_symbol: Iterable[str] = None,
        classification_name: str = "classification",
        classification_list: Union[Path, str, Iterable[str]] = None,
        category: str = "merge",
    ):
        super().__init__(train)
        self.train = train
        self.speech_name = speech_name
        self.text_name = text_name
        self.speech_volume_normalize = speech_volume_normalize
        self.rir_apply_prob = rir_apply_prob
        self.noise_apply_prob = noise_apply_prob
        self.short_noise_thres = short_noise_thres
        self.aux_task_names = aux_task_names
        self.classification_name = classification_name
        self.classification_list = classification_list

        if token_type is not None and token_list is not None:
            # if token_list is None:
            #     raise ValueError("token_list is required if token_type is not None")
            self.text_cleaner = TextCleaner(text_cleaner)

            self.tokenizer = build_tokenizer(
                token_type=token_type,
                bpemodel=bpemodel,
                delimiter=delimiter,
                space_symbol=space_symbol,
                non_linguistic_symbols=non_linguistic_symbols,
                g2p_type=g2p_type,
                nonsplit_symbol=nonsplit_symbol,
            )
            if bpemodel not in ["whisper_en", "whisper_multilingual"]:
                self.token_id_converter = TokenIDConverter(
                    token_list=token_list,
                    unk_symbol=unk_symbol,
                )
            else:
                self.token_id_converter = OpenAIWhisperTokenIDConverter(
                    model_type=bpemodel
                )
        else:
            self.text_cleaner = None
            self.tokenizer = None
            self.token_id_converter = None

        if train and rir_scp is not None:
            self.rirs = []
            with open(rir_scp, "r", encoding="utf-8") as f:
                for line in f:
                    sps = line.strip().split(None, 1)
                    if len(sps) == 1:
                        self.rirs.append(sps[0])
                    else:
                        self.rirs.append(sps[1])
        else:
            self.rirs = None

        if train and noise_scp is not None:
            self.noises = []
            with open(noise_scp, "r", encoding="utf-8") as f:
                for line in f:
                    sps = line.strip().split(None, 1)
                    if len(sps) == 1:
                        self.noises.append(sps[0])
                    else:
                        self.noises.append(sps[1])
            sps = noise_db_range.split("_")
            if len(sps) == 1:
                self.noise_db_low = self.noise_db_high = float(sps[0])
            elif len(sps) == 2:
                self.noise_db_low, self.noise_db_high = float(sps[0]), float(sps[1])
            else:
                raise ValueError(
                    "Format error: '{noise_db_range}' e.g. -3_4 -> [-3db,4db]"
                )
        else:
            self.noises = None

        if self.classification_list is not None:
            self.classification_id_converter = TokenIDConverter(
                token_list=classification_list, unk_symbol=unk_symbol
            )

        self.ebc2bc = {}
        for x, y in C.SWBD_BC_CATEGORIES.items():
            if category == "merge":
                self.ebc2bc.setdefault(x, y[0])
            elif category == "binary":
                self.ebc2bc.setdefault(x, y[2])
            else:
                self.ebc2bc.setdefault(x, y[1])

    def _label_process(
        self, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        if self.classification_name in data:
            labels = data[self.classification_name]
            if isinstance(labels, np.ndarray):
                return data

            label_words = [self.ebc2bc[x] for x in labels.split()]
            label_ints = self.classification_id_converter.tokens2ids(label_words)
            label_np_ints = np.array(label_ints, dtype=np.int64)
            if "PAD" in labels:
                label_np_ints = (
                    label_np_ints - 3
                )  # <blank>, <unk> deleted and PAD must be -1 as ignored index.
            else:
                label_np_ints = label_np_ints - 2  # <blank>, <unk> deleted.
            data[self.classification_name] = label_np_ints

        if self.text_name in data:
            data = self._text_process(data)

        assert check_return_type(data)
        return data

    def __call__(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        assert check_argument_types()

        data = self._speech_process(data)
        data = self._label_process(data)

        if self.tokenizer is not None:
            data = self._text_process(data)

        return data


class KoreanClassificationPreprocessor(CommonPreprocessor):
    def __init__(
        self,
        train: bool,
        token_type: str = None,
        token_list: Union[Path, str, Iterable[str]] = None,
        bpemodel: Union[Path, str, Iterable[str]] = None,
        text_cleaner: Collection[str] = None,
        g2p_type: str = None,
        unk_symbol: str = "<unk>",
        space_symbol: str = "<space>",
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        delimiter: str = None,
        rir_scp: str = None,
        rir_apply_prob: float = 1.0,
        noise_scp: str = None,
        noise_apply_prob: float = 1.0,
        noise_db_range: str = "3_10",
        short_noise_thres: float = 0.5,
        aux_task_names: Collection[str] = None,
        speech_volume_normalize: float = None,
        speech_name: str = "speech",
        text_name: str = "text",
        fs: int = 0,
        nonsplit_symbol: Iterable[str] = None,
        classification_name: str = "classification",
        classification_list: Union[Path, str, Iterable[str]] = None,
        category: str = "merge",
    ):
        super().__init__(train)
        self.train = train
        self.speech_name = speech_name
        self.text_name = text_name
        self.speech_volume_normalize = speech_volume_normalize
        self.rir_apply_prob = rir_apply_prob
        self.noise_apply_prob = noise_apply_prob
        self.short_noise_thres = short_noise_thres
        self.aux_task_names = aux_task_names
        self.classification_name = classification_name
        self.classification_list = classification_list

        if token_type is not None and token_list is not None:
            # if token_list is None:
            #     raise ValueError("token_list is required if token_type is not None")
            self.text_cleaner = TextCleaner(text_cleaner)

            self.tokenizer = build_tokenizer(
                token_type=token_type,
                bpemodel=bpemodel,
                delimiter=delimiter,
                space_symbol=space_symbol,
                non_linguistic_symbols=non_linguistic_symbols,
                g2p_type=g2p_type,
                nonsplit_symbol=nonsplit_symbol,
            )
            if bpemodel not in ["whisper_en", "whisper_multilingual"]:
                self.token_id_converter = TokenIDConverter(
                    token_list=token_list,
                    unk_symbol=unk_symbol,
                )
            else:
                self.token_id_converter = OpenAIWhisperTokenIDConverter(
                    model_type=bpemodel
                )
        else:
            self.text_cleaner = None
            self.tokenizer = None
            self.token_id_converter = None

        if train and rir_scp is not None:
            self.rirs = []
            with open(rir_scp, "r", encoding="utf-8") as f:
                for line in f:
                    sps = line.strip().split(None, 1)
                    if len(sps) == 1:
                        self.rirs.append(sps[0])
                    else:
                        self.rirs.append(sps[1])
        else:
            self.rirs = None

        if train and noise_scp is not None:
            self.noises = []
            with open(noise_scp, "r", encoding="utf-8") as f:
                for line in f:
                    sps = line.strip().split(None, 1)
                    if len(sps) == 1:
                        self.noises.append(sps[0])
                    else:
                        self.noises.append(sps[1])
            sps = noise_db_range.split("_")
            if len(sps) == 1:
                self.noise_db_low = self.noise_db_high = float(sps[0])
            elif len(sps) == 2:
                self.noise_db_low, self.noise_db_high = float(sps[0]), float(sps[1])
            else:
                raise ValueError(
                    "Format error: '{noise_db_range}' e.g. -3_4 -> [-3db,4db]"
                )
        else:
            self.noises = None

        if self.classification_list is not None:
            self.classification_id_converter = TokenIDConverter(
                token_list=classification_list, unk_symbol=unk_symbol
            )

        self.ebc2bc = {}

        for x, y in C.BC_CATEGORIES.items():
            if category == "merge":
                self.ebc2bc.setdefault(x, y[0])
            elif category == "binary":
                self.ebc2bc.setdefault(x, y[2])
            else:
                self.ebc2bc.setdefault(x, y[1])

    def _label_process(
        self, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        if self.classification_name in data:
            labels = data[self.classification_name]
            if isinstance(labels, np.ndarray):
                return data

            label_words = [self.ebc2bc[x] for x in labels.split()]
            label_ints = self.classification_id_converter.tokens2ids(label_words)
            label_np_ints = np.array(label_ints, dtype=np.int64)
            if "PAD" in labels:
                label_np_ints = (
                    label_np_ints - 3
                )  # <blank>, <unk> deleted and PAD must be -1 as ignored index.
            else:
                label_np_ints = label_np_ints - 2  # <blank>, <unk> deleted.
            data[self.classification_name] = label_np_ints

        if self.text_name in data:
            data = self._text_process(data)

        assert check_return_type(data)
        return data

    def __call__(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        assert check_argument_types()

        data = self._speech_process(data)
        data = self._label_process(data)

        if self.tokenizer is not None:
            data = self._text_process(data)

        return data
