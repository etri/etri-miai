from pathlib import Path


class Locator:
    exp_tag = None
    dataset_tag = None
    dataset_name = None
    test_name = None

    current_bc_tag = None

    @classmethod
    def set_name(cls, exp_tag: str, dataset_name: str, dataset_tag: str, test_name):
        cls.exp_tag = exp_tag
        cls.dataset_name = dataset_name
        cls.dataset_tag = dataset_tag
        cls.test_name = test_name

    @classmethod
    def set_current_name(cls, bc_tag: str):
        cls.current_bc_tag = bc_tag

    @classmethod
    def exp_root(cls) -> Path:
        return Path(cls.exp_tag) / f"{cls.dataset_tag}.exp"

    @classmethod
    def tag_root(cls, tag: str) -> Path:
        return cls.exp_root() / tag

    @classmethod
    def current_tag(cls) -> Path:
        return cls.exp_root() / cls.current_bc_tag

    @classmethod
    def config_name(cls) -> Path:
        return cls.current_tag() / "config.yaml"

    @classmethod
    def predict_path(cls, inference_tag: str) -> Path:
        return cls.current_tag() / inference_tag / cls.test_name / "bc"

    @classmethod
    def result_path(cls, result: str = "result.json") -> Path:
        return cls.current_tag() / result

    @classmethod
    def utt2num_path(cls) -> Path:
        return (
            Path(cls.exp_tag)
            / f"{cls.dataset_tag}.dump"
            / "raw"
            / cls.test_name
            / "utt2num_samples"
        )

    @classmethod
    def gold_path(cls) -> Path:
        return (
            Path(cls.exp_tag)
            / f"{cls.dataset_tag}.dump"
            / "raw"
            / cls.test_name
            / "bc_label"
        )

    @classmethod
    def confusion_matrix_path(cls, inference_tag: str, data_type: str) -> Path:
        return (
            cls.current_tag()
            / inference_tag
            / cls.test_name
            / f"{data_type}_confusion_matrix.png"
        )
