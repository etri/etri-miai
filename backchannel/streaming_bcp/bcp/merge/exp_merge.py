import argparse
import json
import pandas as pd

from pathlib import Path
from tqdm.auto import tqdm

# key-column name mapping
key_column_mapping = {
    "evaluation_name": "evaluation type",
    "dataset_name": "dataset name",
    "bc_tag": "bc tag",
    "inference_tag": "inference tag",
    "test_name": "test name",
    "macroprecision": "macro precision",
    "macrorecall": "macro recall",
    "macrof1": "macro f1",
    "weightedprecision": "weighted precision",
    "weightedrecall": "weighted recall",
    "weightedf1": "weighted f1",
    "accuracy": "accuracy",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge the results of the BC scores")
    parser.add_argument("root", type=str, help="Root folder path")
    args = parser.parse_args()

    root = Path(args.root)

    paths = [p for p in root.glob("*_dump_and_exp/**/result.json")]

    dfs = {}

    for path in tqdm(paths):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for _, evaluation_types in data.items():
            for k, non_dict in evaluation_types.items():
                # dict type이 아닌 것들을 가져온다.
                label_names = eval(non_dict["label_names"])
                dataset_name = non_dict["dataset_name"]

                if len(label_names) == 2:
                    category_label = "binary"
                else:
                    category_label = "multi"

                new_dict = {}
                for key, value in non_dict.items():
                    if key in key_column_mapping:
                        if key == "test_name":
                            new_dict[key_column_mapping[key]] = value.split("/")[-1]
                        else:
                            new_dict[key_column_mapping[key]] = value
                    elif key.startswith("each"):
                        for label_name, each_value in zip(label_names, eval(value)):
                            new_key = key.replace("each", "")
                            new_dict[f"{label_name} {new_key}"] = each_value
                    else:
                        continue

                df = pd.DataFrame([new_dict])

                df_key = f"{dataset_name}_{category_label}"

                if df_key in dfs:
                    dfs[df_key] = pd.concat([dfs[df_key], df], ignore_index=True)
                else:
                    dfs[df_key] = df

    # 하나의 엑셀 파일에 key를 sheet로 생성해서 df를 저장
    output_path = root / "merged_results.xlsx"
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        for key, df in dfs.items():
            df.to_excel(writer, sheet_name=key, index=False)

            # writer 객체에서 workbook과 worksheet 객체 가져오기
            worksheet = writer.sheets[key]

            # 컬럼 너비 자동 조절
            for col_num, value in enumerate(df.columns.values):
                # 컬럼 헤더의 글자 수를 기반으로 열 너비 계산
                column_len = max(df[value].astype(str).apply(len).max(), len(value)) + 2
                worksheet.set_column(col_num, col_num, column_len)
