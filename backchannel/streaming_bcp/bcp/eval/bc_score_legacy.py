import argparse
import os

import numpy as np

from collections import OrderedDict
from datetime import datetime
from itertools import chain
from sklearn import metrics

from sklearn.metrics import confusion_matrix
import pandas as pd

import matplotlib.colors as colors
import matplotlib.pyplot as plt

import bcp.const.const as C
from bcp.eval.pretty_confusion_matrix import pp_matrix, pp_matrix_from_data

# confusion matrix set the color map
white = (0.56, 0.7372, 0.56)  # Abbreviated RGB notation applis a division by 255
regn_green = (0.596, 0.984, 0.596)  # regular green: (red, green, blue) = (0, 1, 0)
dark_green = (0, 0.2, 0)  # dark green: (red, green, blue) = (0, 0.2, 0)

cdict = {
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

mycmap = colors.LinearSegmentedColormap("myGreen", cdict)


def read_file(path):
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            key, *line_lbs = line.split()
            labels.append(line_lbs)

    return labels


def mapping_nobc_bc(labels, label_dict, label2idx):
    res_labels = []

    for instance in labels:
        mapping_label = [
            label2idx[x] if "EMPATHETIC" == x or "BC" == x else label2idx[label_dict[x]]
            for x in instance
        ]
        res_labels.append(mapping_label)

    return res_labels


def extract_counsel(gold_path, pred_labels, gold_labels, is_extract=False):
    # TODO: Hop size에 따른 2초 전 위치가 달라질 수가 있음
    labels = OrderedDict()
    with open(gold_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            line_key = line.split()[0]
            line_lbs = line.split()[1:]

            utt_key = "-".join(line_key.split("-")[:2])

            labels.setdefault(utt_key, [])
            labels[utt_key].extend(line_lbs)

    csor_loc_labels = []
    csee_loc_labels = []
    prev_size = 0
    for utt in labels.keys():
        if "상담자" in utt:
            loc_labels = csee_loc_labels
        else:
            loc_labels = csor_loc_labels

        utt_labels = labels[utt]
        loc_label = []

        for i in range(len(utt_labels)):
            if is_extract:
                if utt_labels[i] != "NoBC":
                    loc_label.append(i)

                    if i - 4 >= 0:
                        if utt_labels[i - 4] != "NoBC":
                            if utt_labels[i - 3] != "NoBC":
                                # print("ERROR-1")
                                continue
                            else:
                                loc_label.append(i - 3)
                        else:
                            loc_label.append(i - 4)
                    else:
                        if utt_labels[0] != "NoBC":
                            # print("ERROR-2")
                            continue
                        else:
                            loc_label.append(0)
            else:
                loc_label.append(i)

        loc_labels.extend([x + prev_size for x in loc_label])
        prev_size = len(utt_labels) + prev_size

    csor_pred_labels = pred_labels[csor_loc_labels]
    csor_gold_labels = gold_labels[csor_loc_labels]
    csee_pred_labels = pred_labels[csee_loc_labels]
    csee_gold_labels = gold_labels[csee_loc_labels]

    return csor_pred_labels, csor_gold_labels, csee_pred_labels, csee_gold_labels


def extract_nobc(gold_path, pred_labels, gold_labels):
    labels = OrderedDict()
    with open(gold_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            line_key = line.split()[0]
            line_lbs = line.split()[1:]

            utt_key = "-".join(line_key.split("-")[:2])

            labels.setdefault(utt_key, [])
            labels[utt_key].extend(line_lbs)

    loc_labels = []
    prev_size = 0
    for utt in labels.keys():
        utt_labels = labels[utt]
        loc_label = []

        for i in range(len(utt_labels)):
            if utt_labels[i] != "NOBC":
                loc_label.append(i)

                if i - 4 >= 0:
                    if utt_labels[i - 4] != "NOBC":
                        if utt_labels[i - 3] != "NOBC":
                            # print("ERROR-1")
                            continue
                        else:
                            loc_label.append(i - 3)
                    else:
                        loc_label.append(i - 4)
                else:
                    if utt_labels[0] != "NOBC":
                        # print("ERROR-2")
                        continue
                    else:
                        loc_label.append(0)

        loc_labels.extend([x + prev_size for x in loc_label])
        prev_size = len(utt_labels) + prev_size

    ext_pred_labels = pred_labels[loc_labels]
    ext_gold_labels = gold_labels[loc_labels]

    return ext_pred_labels, ext_gold_labels


def added_cal_f1(
    preds, golds, row, name="total", label_dict={}, exp_dir="exp", inference_tag=""
):
    f1_score = metrics.f1_score(golds, preds, average=None)
    weighted_f1_score = metrics.f1_score(golds, preds, average="weighted")
    # acc = metrics.accuracy_score(full_gold_labels, full_pred_labels)
    row.append(round(weighted_f1_score * 100.0, 2))  # f1
    # row.append(round(f1_score[1] * 100.0, 2))  # continuer
    # row.append(round(f1_score[2] * 100.0, 2))  # understanding
    # if len(f1_score) == 4:
    #     row.append(round(f1_score[3] * 100.0, 2))  # empathetic
    for score in f1_score[1:]:
        row.append(round(score * 100.0, 2))
    if 4 - len(f1_score[1:]) > 0:
        size = 4 - len(f1_score[1:])
        for _ in range(1, size):
            row.append("-")
    row.append(round(f1_score[0] * 100.0, 2))  # noBC

    columns = [x for x in sorted(label_dict.items(), key=lambda x: x[1], reverse=False)]

    img_path = os.path.join(exp_dir, row[1], "images")

    if not os.path.exists(img_path):
        print(f"create folder: {img_path}")
        os.makedirs(img_path, exist_ok=True)

    if inference_tag == "":
        save_path = f"{img_path}/{name}.png"
    else:
        save_path = f"{img_path}/{name}_{inference_tag}.png"
    pp_matrix_from_data(golds, preds, columns, cmap=mycmap, save=save_path)
    print("save confusion matrix: ", save_path)


def main(args):
    exp_folder = args.exp_folder
    output_file = args.output_file
    inference_tag = args.inference_tag
    tmp_dataset = args.dataset

    tmp_dataset = tmp_dataset.strip()

    assert len(tmp_dataset.split("/")) <= 2, f"dataset: {tmp_dataset} is not valid"

    if len(tmp_dataset.split("/")) == 2:
        parent_dataset, dataset = tmp_dataset.split("/")
    else:
        parent_dataset = "."
        dataset = tmp_dataset

    gold_path = os.path.join(args.gold_path, "bc_label")

    excel_file_path = os.path.join(exp_folder, output_file)  # "exp/ckbd/results.xlsx"

    if parent_dataset == "swbd":
        label_dict = {}
        for x, y in C.SWBD_BC_CATEGORIES.items():
            if args.expanded_labels == "merge":
                label_dict.setdefault(x, y[0])
            elif args.expanded_labels == "binary":
                label_dict.setdefault(x, y[2])
            else:
                label_dict.setdefault(x, y[1])
    else:
        label_dict = {}
        for x, y in C.BC_CATEGORIES.items():
            if args.expanded_labels == "merge":
                label_dict.setdefault(x, y[0])
            elif args.expanded_labels == "binary":
                label_dict.setdefault(x, y[2])
            else:
                label_dict.setdefault(x, y[1])

    print("label_dict: ", label_dict)

    label2idx = {}
    for y in label_dict.values():
        if y not in label2idx:
            label2idx[y] = len(label2idx)

    print("label2idx: ", label2idx)

    if os.path.exists(excel_file_path):
        print("loading existing results excel file")
        res_df = pd.read_excel(excel_file_path, sheet_name="results")
    else:
        print("creating new results excel file")
        if parent_dataset == "swbd":
            res_df = pd.DataFrame(
                columns=[
                    "exp_time",
                    "model",
                    "tag",
                    "testset",
                    "total_f1",
                    "total_continuer",
                    "total_assessment",
                    "total_no_bc",
                    "ext_total_f1",
                    "ext_total_continuer",
                    "ext_total_assessment",
                    "ext_total_no_bc",
                ]
            )
        else:
            res_df = pd.DataFrame(
                columns=[
                    "exp_time",
                    "model",
                    "tag",
                    "testset",
                    "total_f1",
                    "total_continuer",
                    "total_understanding",
                    "total_empathetic",
                    "total_no_bc",
                    "ext_total_f1",
                    "ext_total_continuer",
                    "ext_total_understanding",
                    "ext_total_empathetic",
                    "ext_total_no_bc",
                ]
            )

    print(
        f"searching new models (ex. {exp_folder}/MODELNAME/{inference_tag}/{parent_dataset}/{dataset}/bc)"
    )

    tmp_keys = [
        f"{x}/{y}/{z}"
        for x, y, z in zip(res_df["model"], res_df["tag"], res_df["testset"])
    ]

    model_folders = [
        (f.name, datetime.fromtimestamp(f.stat().st_mtime))
        for f in os.scandir(exp_folder)
        if f.is_dir()
        and "stats" not in f.name
        and f"{f.name}/{inference_tag}/{parent_dataset.upper()}" not in tmp_keys
        and os.path.exists(
            os.path.join(
                exp_folder, f.name, inference_tag, parent_dataset, dataset, "bc"
            )
        )
    ]
    print(f"found {len(model_folders)} new models")

    for model_folder, model_modified in model_folders:
        print(f"processing {model_folder}")
        row = [
            model_modified.strftime("%Y-%m-%d_%H:%M:%S"),
            model_folder,
            inference_tag,
            parent_dataset.upper(),
        ]
        pred_path = os.path.join(
            exp_folder, model_folder, inference_tag, parent_dataset, dataset, "bc"
        )

        pred_labels = read_file(pred_path)
        gold_labels = read_file(gold_path)

        if len(pred_labels) != len(gold_labels):
            print(
                f"{model_folder}: pred len({len(pred_labels)}) != gold len({len(gold_labels)})"
            )
            continue

        pred_cut_labels = []
        errors = []

        # assert len(pred_labels) == len(
        #     gold_labels
        # ), f"pred len: {len(pred_labels)} != gold len: {len(gold_labels)}"

        for line_idx, (p_label, g_label) in enumerate(zip(pred_labels, gold_labels)):
            if len(p_label) != len(g_label):
                errors.append(f"{line_idx}: {len(p_label)} != {len(g_label)}")
                tmp = p_label[: len(g_label)]
                pred_cut_labels.append(tmp)
            else:
                tmp = p_label
                pred_cut_labels.append(tmp)

            assert len(tmp) == len(g_label), f"{len(tmp)} != {len(g_label)}"

        assert len(pred_cut_labels) == len(
            gold_labels
        ), f"{len(pred_cut_labels)} != {len(gold_labels)}"

        pred_cut_labels = mapping_nobc_bc(pred_cut_labels, label_dict, label2idx)
        gold_cut_labels = mapping_nobc_bc(gold_labels, label_dict, label2idx)

        pred_cut_labels = list(chain(*pred_cut_labels))
        gold_cut_labels = list(chain(*gold_cut_labels))

        pred_cut_labels = np.array(pred_cut_labels, dtype=np.int64)
        gold_cut_labels = np.array(gold_cut_labels, dtype=np.int64)

        added_cal_f1(
            pred_cut_labels,
            gold_cut_labels,
            row,
            name="total",
            label_dict=label2idx,
            exp_dir=exp_folder,
            inference_tag=inference_tag,
        )  # total

        # (
        #     csor_pred_labels,
        #     csor_gold_labels,
        #     csee_pred_labels,
        #     csee_gold_labels,
        # ) = extract_counsel(
        #     gold_path, pred_cut_labels, gold_cut_labels, is_extract=False
        # )
        # added_cal_f1(
        #     csor_pred_labels,
        #     csor_gold_labels,
        #     row,
        #     name="total-counselor",
        #     label_dict=label_dict,
        # )  # counselor
        # added_cal_f1(
        #     csee_pred_labels,
        #     csee_gold_labels,
        #     row,
        #     name="total-counselee",
        #     label_dict=label_dict,
        # )  # counselee

        ext_pred_labels, ext_gold_labels = extract_nobc(
            gold_path, pred_cut_labels, gold_cut_labels
        )

        added_cal_f1(
            ext_pred_labels,
            ext_gold_labels,
            row,
            name="extract",
            label_dict=label2idx,
            exp_dir=exp_folder,
            inference_tag=inference_tag,
        )  # extract

        # (
        #     ext_csor_pred_labels,
        #     ext_csor_gold_labels,
        #     ext_csee_pred_labels,
        #     ext_csee_gold_labels,
        # ) = extract_counsel(
        #     gold_path, pred_cut_labels, gold_cut_labels, is_extract=True
        # )
        # added_cal_f1(
        #     ext_csor_pred_labels, ext_csor_gold_labels, row, name="extract-counselor"
        # )  # ext_counselor
        # added_cal_f1(
        #     ext_csee_pred_labels, ext_csee_gold_labels, row, name="extract-counselee"
        # )  # ext_counselee
        res_df.loc[len(res_df)] = row

    res_df = res_df.sort_values(by="model", ascending=True)

    # Excel 파일로 쓰기
    res_df.to_excel(
        excel_file_path, index=False, sheet_name="results"
    )  # 수정된 DataFrame을 Excel 파일로 저장 (인덱스 열 제외)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The collection of results from backchannel predictions"
    )
    parser.add_argument("--exp_folder", type=str, default="exp/ckbd")
    parser.add_argument("--output_file", type=str, default="results.xlsx")
    parser.add_argument("--expanded_labels", type=str, default="merge")
    parser.add_argument(
        "--inference_tag",
        type=str,
        default="inference_bcp_model_valid.macro_f1_scores.ave",
    )
    parser.add_argument("--dataset", type=str, default="processed_test")
    parser.add_argument("--gold_path", type=str, default="dump/raw/ckbd/processed_test")

    args = parser.parse_args()

    main(args)
