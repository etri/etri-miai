# BACKCast: Backchannel Annotation Dataset for Korean Broadcast Conversational Speech
[Korean](./README.md) [English](./README.en.md)
---

## Overview

BACKCast is a dataset constructed by extracting **two-party dialogues** from the AIHub
**Broadcast Content Conversational Speech Recognition Dataset** and adding **backchannel** annotations.

The original AIHub data does not include backchannel transcriptions. This dataset
identifies and organizes backchannel occurrences as a separate annotation layer.

> Source data: [AIHub Broadcast Content Conversational Speech Recognition Dataset](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&searchKeyword=%EB%B0%A9%EC%86%A1%20%EC%BD%98%ED%85%90%EC%B8%A0%20%EB%8C%80%ED%99%94%EC%B2%B4%20%EC%9D%8C%EC%84%B1%EC%9D%B8%EC%8B%9D%20%EB%8D%B0%EC%9D%B4%ED%84%B0&aihubDataSe=data&dataSetSn=463)

---

## What is a Backchannel?

In this dataset, a backchannel refers to a **short reactive expression produced by the
listener while the speaker is talking** — for example, `네` (yeah), `응` (uh-huh),
`맞아` (right), or `아이고` (oh my).

Backchannels typically share the following characteristics:

- Brief in duration
- Produced without interrupting or taking over the speaker's turn
- Serving functions such as attention, understanding, empathy, agreement, or surprise

Note that not every short response qualifies as a backchannel. Responses that function
as **answers, opinions, or genuine requests for clarification** are excluded.

---

## Dataset Structure

The dataset is organized into **per-dialogue folders**, each containing two TSV files:

- `word.tsv` — word-level transcription data
- `backchannel.tsv` — backchannel-annotated entries

> Word-level segmentation was performed using [WhisperX](https://github.com/m-bain/whisperx).

### Folder Layout

```text
dataset/
  dialogue_000001/
    word.tsv
    backchannel.tsv
  dialogue_000002/
    word.tsv
    backchannel.tsv
  ...
````

---

## File Descriptions

### word.tsv

Contains word-level transcription information. Each row corresponds to a single word (eojeol).

|Column|Description|
|---|---|
|`uid`|Utterance ID|
|`sid`|Segment ID|
|`wid`|Word ID|
|`speaker`|Speaker ID|
|`start`|Word start time (seconds)|
|`end`|Word end time (seconds)|
|`text`|Word text|
|`backchannels`|List of backchannel IDs that overlap with this word|

### backchannel.tsv

Contains only the backchannel-annotated entries. Each row corresponds to a single backchannel instance.

|Column|Description|
|---|---|
|`bid`|Backchannel ID|
|`speaker`|Speaker ID of the backchannel producer|
|`start`|Backchannel start time (seconds)|
|`end`|Backchannel end time (seconds)|
|`category`|Backchannel category|
|`text`|Backchannel text|
|`respondee_uid`|Utterance ID of the target utterance being responded to|
|`respondee_sid`|Segment ID of the target utterance|
|`respondee_wid`|Word ID of the target utterance|

---

## Backchannel Categories

Backchannels are divided into two broad types — **Neutral** and **Empathetic** — and further classified into the following seven categories.

### Neutral

|Category|Description|
|---|---|
|**CONTINUER**|Signals that the listener is actively attending and following the conversation|
|**UNDERSTANDING**|Indicates that the listener has understood or is in the process of understanding|

### Empathetic

|Category|Description|
|---|---|
|**ASSESSMENT**|Evaluative reactions such as surprise, admiration, or sympathy|
|**POSITIVE_SURPRISE**|Exclamatory reactions to good news or positive events|
|**NEGATIVE_SURPRISE**|Reactions to unfavorable events or negative situations|
|**REQUEST_CONFIRMATION**|Formally resembles a clarification request but primarily functions as an emotional reaction (surprise, admiration, concern, etc.)|
|**AFFIRMATIVE**|Expresses agreement, approval, or support|

---

## Annotation Notes

- Category assignment is not based solely on surface form; **context** must be considered.
- The same expression may or may not constitute a backchannel depending on the situation.
- An utterance that resembles a question (e.g., `REQUEST_CONFIRMATION`) may still be classified as a backchannel if its primary function is emotional rather than information-seeking.
- Conversely, a response produced when the listener is genuinely expected to answer or give an opinion is not considered a backchannel.

---

## Audio Files

This repository contains only the annotation data. Audio files can be downloaded separately from the [AIHub source data page](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&searchKeyword=%EB%B0%A9%EC%86%A1%20%EC%BD%98%ED%85%90%EC%B8%A0%20%EB%8C%80%ED%99%94%EC%B2%B4%20%EC%9D%8C%EC%84%B1%EC%9D%B8%EC%8B%9D%20%EB%8D%B0%EC%9D%B4%ED%84%B0&aihubDataSe=data&dataSetSn=463).

---

## License

This dataset is released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

---

## Citation

```bibtex
@misc{BACKCast,
    title={BACKCast: Backchannel Annotation Dataset for Korean Broadcast Conversational Speech},
    author={Seung Hi Kim, Yong-Seok Choi and Sung Yup Lee},
    year={2026},
    howpublished={\url{https://github.com/etri/etri-miai}},
}
```

---

## Contact

For questions or inquiries, please open an issue in this repository or reach out via email.

📧 [yseokchoi@etri.re.kr](mailto:yseokchoi@etri.re.kr)