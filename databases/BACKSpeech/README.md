
# BACKSpeech: Backchannel Annotation Corpus in Korean Speech

## Introduction

We introduce the Backchannel Annotation Corpus in Korean Speech. This corpus has been collected from a wide range of YouTube videos containing dialogues of two or three participants and is intended for researchers and developers working on Backchannel Prediction (BCP) and related fields. 
This corpus is the first Backchannel Prediction corpus available in Korean. We hope that this corpus will contribute to the field of conversational AI by enabling Backchannel Prediction research.


## Dataset Description

The dataset comprises 59 audio files and corresponding transcriptions in CSV format.
Each audio file is stored in the following format: 16000Hz, 16bit, mono, RIFF WAV header.
The dataset contains approximately 19,230 segments, with a total audio duration of about 22 hours.


### Data Format

Each data entry in the dataset consists of:
1. Video List
2. Annotations for ASR containing transcription and segments information
3. Annotations for BCP containing BC annotations 


### Backchannel Categories

1. Normal
- Continuer
- Understanding
2. Empathetic
- Empathetic Response
  - Assessment
    - Negative Surprise
    - Positive Surprise
  - Request Confirmation
- Affirmative
3. Non Backchannel


## Usage Guidelines

To ensure the responsible use of this corpus, please adhere to the following guidelines:
1. **Attribution**: Please credit the creators of the corpus by linking back to this repository in your research publications or project documentation.
2. **Privacy**: Do not use the corpus to identify or infer sensitive information about individuals featured in the videos.
3. **Non-commercial use**: This corpus is made available for academic and research purposes only. Commercial use of the corpus is prohibited.


## Download

You can download the video on [list/list.txt](https://github.com/etri/etri-miai/blob/main/databases/BACKSpeech/list/list.txt) by using the following guideline.
[[:link:]](https://github.com/etri/kmsav/blob/main/HOWTO.md#data-prepare)


## License

This dataset is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Citation

If you use this dataset in your research, please cite it as follows:

```less
@misc{BACKSpeech,
    title={Backchannel Annotation Corpus in Korean Speech},
    author={Seung Hi, Kim, Jeong-Uk Bang, Yong-Seok Choi, Seung Yun},
    year={2023},
    howpublished={\url{https://github.com/etri/etri-miai}},
}
```

## Contact

For any questions or concerns related to this dataset, please reach out to us by opening an issue on this repository or by contacting us at [yseokchoi@etri.re.kr](mailto:yseokchoi@etri.re.kr).

