import matplotlib.colors as colors
import numpy as np

from typing import Tuple

from bcp.eval.pretty_confusion_matrix import pp_matrix_from_data


class ConfusionMatrix:
    def __init__(
        self,
        white=(0.56, 0.7372, 0.56),
        regn_green=(0.596, 0.984, 0.596),
        dark_green=(0, 0.2, 0),
        verbose=False,
    ):
        """_summary_

        Args:
            white (tuple, optional): 축약된 RGB 표기법은 255로 나누기를 적용. Defaults to (0.56, 0.7372, 0.56).
            regn_green (tuple, optional): 일반 RGB: (빨강, 녹색, 파랑) = (0, 1, 0). Defaults to (0.596, 0.984, 0.596).
            dark_green (tuple, optional): dark RGB: (빨강, 녹색, 파랑) = (0, 0.2, 0). Defaults to (0, 0.2, 0).
        """
        self._cdict = {
            "red": (
                (0.0, white[0], white[0]),
                (0.25, regn_green[0], regn_green[0]),
                (1.0, dark_green[0], dark_green[0]),
            ),
            "green": (
                (0.0, white[1], white[1]),
                (0.25, regn_green[1], regn_green[1]),
                (1.0, dark_green[1], dark_green[1]),
            ),
            "blue": (
                (0.0, white[2], white[2]),
                (0.25, regn_green[2], regn_green[2]),
                (1.0, dark_green[2], dark_green[2]),
            ),
        }

        self.mycmap = colors.LinearSegmentedColormap("myGreen", self._cdict)
        self.verbose = verbose

    def save_confusion_matrix(
        self, golds: np.ndarray, preds: np.ndarray, columns: Tuple, save_path: str
    ):
        """_summary_

        Args:
            golds (np.ndarry): [description]
            preds (np.ndarray): [description]
            save_path (str): [description]
        """
        pp_matrix_from_data(
            golds, preds, columns=columns, cmap=self.mycmap, save=save_path
        )

        if self.verbose:
            print(f"Confusion matrix is saved at {save_path}")
