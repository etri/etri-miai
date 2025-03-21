class Reporter:

    def __init__(
        self,
        **kwargs,
    ):
        """
        Supported keys
        evaluation_name: 평가 유형 (ex: streaming, balanaced)
        dataset_name: 데이터 도메인 (ex: ckbd, bkbd, swbd)
        bc_tag: 모델 유형
        inference_tag: 추론 모델
        test_name: 평가 세트 이름 (ex: valid, test)
        save_img: confusion matrix 이미지 저장 경로

        label_names: 백채널 카테고리
        eachprecision: 각 백채널 카테고리 별 precision
        eachrecall: 각 백채널 카테고리 별 recall
        eachf1: 각 백채널 카테고리 별 f1
        macroprecision: Macro Precision
        macrorecall: Macro Recall
        macrof1: Macro F1
        weightedprecision: Weighted Precision
        weightedrecall: Weighted Recall
        weightedf1: Weighted F1
        accuracy: Accuracy
        """

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def name(self):
        return (
            f"{self.dataset_name}/{self.bc_tag}/{self.inference_tag}/{self.test_name}"
        )

    # dictionary 형태로 출력
    def get_dict(self) -> dict:
        return self.__dict__

    def __repr__(self) -> str:
        fmt = "=" * 50 + "\n"
        fmt += f"Evaluation Name: {self.evaluation_name}\n"
        fmt += f"Dataset Name: {self.dataset_name}\n"
        fmt += f"BC Tag: {self.bc_tag}\n"
        fmt += f"Inference Tag: {self.inference_tag}\n"
        fmt += f"Test Name: {self.test_name}\n"
        fmt += f"Save Image: {self.save_img}\n"
        fmt += "Label Dictionary: "
        for i, k in enumerate(self.label_names):
            if i == len(self.label_names) - 1:
                fmt += f"{k}\n"
            else:
                fmt += f"{k}, "
        fmt += f"Each Precision Score:\n"
        for k, precision in zip(self.label_names, self.eachprecision):
            fmt += f"\t{k}: {precision}\n"
        fmt += f"Each Recall Score:\n"
        for k, recall in zip(self.label_names, self.eachrecall):
            fmt += f"\t{k}: {recall}\n"
        fmt += f"Each F1 Score:\n"
        for k, f1 in zip(self.label_names, self.eachf1):
            fmt += f"\t{k}: {f1}\n"
        fmt += f"Macro Precision Score: {self.macroprecision}\n"
        fmt += f"Macro Recall Score: {self.macrorecall}\n"
        fmt += f"Macro F1 Score: {self.macrof1}\n"
        fmt += f"Weighted Precision Score: {self.weightedprecision}\n"
        fmt += f"Weighted Recall Score: {self.weightedrecall}\n"
        fmt += f"Weighted F1 Score: {self.weightedf1}\n"
        fmt += f"Accuracy: {self.accuracy}\n"
        fmt += "=" * 50

        return fmt
