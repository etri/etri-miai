import argparse  # 커맨드 라인 파싱

from pathlib import Path

from typing import Dict
from tqdm.auto import tqdm

####### ETC #######
from bcp.eval.draw_confusion_matrix import ConfusionMatrix
from bcp.eval.locator import Locator
from bcp.eval.metrics import Metric
from bcp.eval.resource import Resource
from bcp.eval.reporter import Reporter
from bcp.eval.utterance import Utterance

###################


def process(
    data_type: str,
    args: Dict,
    bc_tag: str,
    inference_tag: str,
    golds: Dict[str, str],
    preds: Dict[str, str],
):
    metric = Metric(
        dataset=args.dataset_name,
        category_label=args.category_label,
        y_true=golds,
        y_pred=preds,
    )

    # Save confusion matrix
    img_path = Locator.confusion_matrix_path(
        inference_tag=inference_tag, data_type=data_type
    )

    results = dict(
        evaluation_name=data_type,
        dataset_name=args.dataset_name,
        bc_tag=bc_tag,
        inference_tag=inference_tag,
        test_name=args.test_name,
        save_img=img_path,
    )

    # Get the metrics
    res_metrics = metric.get_metrics()

    # Save the confusion matrix
    cm.save_confusion_matrix(
        golds=metric.get_golds(),
        preds=metric.get_preds(),
        columns=metric.get_labels(),
        save_path=img_path,
    )

    # Update the results
    results.update(res_metrics)

    reporter = Reporter(**results)
    print(reporter)

    return reporter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate BC Score")
    parser.add_argument(
        "--exp_root", type=str, required=True, help="Experiment folder path"
    )
    parser.add_argument("--dataset_name", type=str, default="ckbd", help="Dataset name")
    parser.add_argument("--dataset_tag", type=str, required=True, help="Dataset tag")
    parser.add_argument("--bc_tag", type=str, default=None, help="BC model tag")
    parser.add_argument("--inference_tag", type=str, default=None, help="Inference tag")
    parser.add_argument("--category_label", type=str, default="merge")
    parser.add_argument("--test_name", type=str, default="XXX.proc_test")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("--result", type=str, default="result.json", help="Result file")
    parser.add_argument(
        "--config", type=str, default="", help="specific config file (Optional)"
    )
    parser.add_argument(
        "--block_mapping", action="store_true", help="need the block mapping"
    )
    args = parser.parse_args()

    # Set experiment folder
    Locator.set_name(
        exp_tag=args.exp_root,
        dataset_tag=args.dataset_tag,
        dataset_name=args.dataset_name,
        test_name=args.test_name,
    )

    # confusion matrix를 위한 클래스 생성
    cm = ConfusionMatrix(verbose=args.verbose)

    # bc_tag가 None일 경우 exp_folder를 순회하면서 stats 폴더를 제외한 모든 폴더를 가져온다. 아닐 경우에는 해당 bc_tag 모델에 대한 결과만 구한다.
    if args.bc_tag is None:
        bc_tags = [
            f.name
            for f in (Locator.exp_root()).iterdir()
            if f.is_dir() and "stats" not in f.name
        ]
    elif (Locator.tag_root(args.bc_tag)).exists():
        bc_tags = [args.bc_tag]
    else:
        raise FileNotFoundError(
            f"{Locator.exp_root().name} 폴더에 {args.bc_tag}가 존재하지 않습니다."
        )

    for bc_tag in (bt_bar := tqdm(bc_tags, leave=False)):
        bt_bar.set_description(f"BC Tag: {bc_tag}")

        reporters = {}

        Locator.set_current_name(bc_tag=bc_tag)

        # Load config file
        if args.config != "":
            config = Resource.load_config_file(Path(args.config))
        else:
            config = Resource.load_config_file(Locator.config_name())

        # inference_tag가 None일 경우 exp_folder를 순회하면서 bc 파일을 전부 찾아서 결과를 각각 구한다.
        # None이 아닐 경우에는 해당 inference_tag에 대한 결과만 구한다.
        if args.inference_tag is None:
            inference_tags = [
                f.name
                for f in Locator.current_tag().iterdir()
                if f.is_dir() and f.name.startswith("decode")
            ]
        else:
            inference_tags = [args.inference_tag]

        for inference_tag in (it_bar := tqdm(inference_tags, leave=False)):
            it_bar.set_description(f"Inference Tag: {inference_tag}")

            # Load prediction and ground truth
            pred_path = Locator.predict_path(inference_tag=inference_tag)
            gold_path = Locator.gold_path()

            try:
                preds = Resource.load_file_for_label(pred_path)
                golds = Resource.load_file_for_label(gold_path)
            except FileNotFoundError as e:
                print("File is not found. Skip the inference tag.")
                continue

            # preds와 golds의 key가 같은지 확인 및 value의 크기가 같은지 확인
            utt = Utterance(
                cfg=config,
                preds=preds,
                golds=golds,
                block_mapping=args.block_mapping,
                locator=Locator,
            )

            # straming process
            streaming_preds = utt.get_preds()
            streaming_golds = utt.get_golds()
            streaming_reporter = process(
                data_type="streaming",
                args=args,
                bc_tag=bc_tag,
                inference_tag=inference_tag,
                golds=streaming_golds,
                preds=streaming_preds,
            )
            reporters.setdefault(streaming_reporter.name, {})
            reporters[streaming_reporter.name].setdefault(
                "streaming", streaming_reporter.get_dict()
            )

            ##############################

            # balanced process
            balanced_preds = utt.get_balanced_preds()
            balanced_golds = utt.get_balanced_golds()
            balanced_reporter = process(
                data_type="balanced",
                args=args,
                bc_tag=bc_tag,
                inference_tag=inference_tag,
                golds=balanced_golds,
                preds=balanced_preds,
            )
            reporters.setdefault(balanced_reporter.name, {})
            reporters[balanced_reporter.name].setdefault(
                "balanced", balanced_reporter.get_dict()
            )
            ##############################

            # block mapping process
            # block_mappingdl True일 경우에만 수행
            if args.block_mapping:
                mapping_preds = utt.get_mapping_preds()
                mapping_golds = utt.get_mapping_golds()
                mapping_reporter = process(
                    data_type="mapping",
                    args=args,
                    bc_tag=bc_tag,
                    inference_tag=inference_tag,
                    golds=mapping_golds,
                    preds=mapping_preds,
                )

                reporters[streaming_reporter.name].setdefault(
                    "streaming_mapper", mapping_reporter.get_dict()
                )

        # Save the results
        if len(reporters) > 0:
            Resource.save_result_file(Locator.result_path(args.result), reporters)
