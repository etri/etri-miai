# BACKCast: Backchannel Annotation Dataset for Korean Broadcast Conversational Speech
[Korean](./README.md) [English](./README.en.md)
---

## 소개
본 데이터셋은 AIHub **방송 콘텐츠 대화체 음성인식 데이터** 중 **2인 대화**로 구성된 부분을 추출하여, **백채널(backchannel)** 정보를 추가로 레이블링한 데이터셋입니다.

AIHub 원천 데이터에는 백채널에 대한 별도 전사가 포함되어 있지 않으며, 본 데이터셋에서는 이를 별도로 식별하고 정리하였습니다.

> 원천 데이터:  [AIHub 방송 콘텐츠 대화체 음성인식 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&searchKeyword=%EB%B0%A9%EC%86%A1%20%EC%BD%98%ED%85%90%EC%B8%A0%20%EB%8C%80%ED%99%94%EC%B2%B4%20%EC%9D%8C%EC%84%B1%EC%9D%B8%EC%8B%9D%20%EB%8D%B0%EC%9D%B4%ED%84%B0&aihubDataSe=data&dataSetSn=463)

## 백채널 정의
본 데이터셋에서 백채널은 **화자의 발화 도중 청자가 보이는 짧은 반응 표현**을 의미합니다.
`네`, `예`, `응`, `음`, `음흠`, `맞아`, `오`, `아이고` 등이 대표적인 예시입니다.

백채널은 일반적으로 다음과 같은 특성을 가집니다.

- 짧은 길이의 청자 반응
- 상대방의 발화를 이어받거나 끊지 않으면서 나타나는 반응
- 집중, 이해, 공감, 동의, 놀람 등의 기능 수행

다만, 모든 짧은 반응이 백채널에 해당하는 것은 아니며, 문맥상 **답변, 의견 제시,
재확인 요청**에 가까운 경우는 제외될 수 있습니다.

## 데이터 구성
데이터는 **대화 단위 폴더**로 구성되어 있으며, 각 폴더에는 아래 두 개의 TSV 파일이
포함되어 있습니다.

- `word.tsv` — 어절 단위 전사 정보
- `backchannel.tsv` — 백채널로 레이블링된 항목 정보

> 어절 단위 분할에는 [WhisperX](https://github.com/m-bain/whisperx)를 활용하였습니다.

### 폴더 구조

```text
dataset/
  [program_000001]/
    dialogue_000001/
      word.tsv
      backchannel.tsv
    dialogue_000002/
      word.tsv
      backchannel.tsv
  [program_000002]/
    dialogue_000101/
      word.tsv
      backchannel.tsv
  ...
```

## **파일 설명**
### word.tsv

어절 단위 전사 정보를 담고 있는 파일로, 각 행은 하나의 어절에 대응합니다.

| 열 이름 | 설명 |
|---|---|
| `uid` | 발화(utterance) ID |
| `sid` | 세그먼트(segment) ID |
| `wid` | 어절(word) ID |
| `speaker` | 화자 ID |
| `start` | 어절 시작 시각 (초) |
| `end` | 어절 종료 시각 (초) |
| `text` | 어절 텍스트 |
| `backchannels` | 해당 어절이 포함된 백채널 ID 목록 |

### backchannel.tsv

백채널로 레이블링된 항목만 모아둔 파일로, 각 행은 하나의 백채널 단위를 나타냅니다.

| 열 이름 | 설명 |
|---|---|
| `bid` | 백채널 ID |
| `speaker` | 백채널을 발화한 화자 ID |
| `start` | 백채널 시작 시각 (초) |
| `end` | 백채널 종료 시각 (초) |
| `category` | 백채널 범주 |
| `text` | 백채널 텍스트 |
| `respondee_uid` | 반응 대상 발화의 utterance ID |
| `respondee_sid` | 반응 대상 발화의 segment ID |
| `respondee_wid` | 반응 대상 발화의 word ID |

## **백채널 범주**

백채널을 **Neutral**과 **Empathetic** 두 가지 성격으로 구분하고,
이를 아래 7개 범주로 세분화하였습니다.

### Neutral

| 범주 | 설명 |
|---|---|
| **CONTINUER** | 청자가 계속 듣고 있으며 대화에 집중하고 있음을 드러내는 반응 |
| **UNDERSTANDING** | 청자가 이해했거나 이해하고 있음을 드러내는 반응 |

### Empathetic

| 범주 | 설명 |
|---|---|
| **ASSESSMENT** | 놀람, 감탄, 안타까움 등 평가적 반응 |
| **POSITIVE_SURPRISE** | 긍정적인 소식이나 좋은 일에 대한 감탄형 반응 |
| **NEGATIVE_SURPRISE** | 부정적인 사건이나 좋지 않은 일에 대한 반응 |
| **REQUEST_CONFIRMATION** | 형태상 재확인 요청처럼 보이나, 실질적으로는 놀람·감탄·안타까움 등 정서적 반응인 경우 |
| **AFFIRMATIVE** | 긍정, 동의, 지지의 태도를 보이는 반응 |


## 해석 시 유의사항

- 백채널 범주는 표현 형태만으로 결정되지 않으며, **문맥**을 함께 고려해야 합니다.
- 동일한 표현이라도 상황에 따라 백채널에 해당하지 않을 수 있습니다.
- `REQUEST_CONFIRMATION`처럼 표면적으로 질문의 형태를 띠더라도,
  실제 기능이 정서적 반응인 경우 백채널로 분류될 수 있습니다.
- 반대로, 청자가 실제 답변이나 의견을 요구받은 상황에서 보인 반응은
  백채널에서 제외될 수 있습니다.

## 오디오 파일

본 저장소에는 백채널 레이블이 추가된 데이터만 포함되어 있습니다.
오디오 파일은 [AIHub 원천 데이터 페이지](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&searchKeyword=%EB%B0%A9%EC%86%A1%20%EC%BD%98%ED%85%90%EC%B8%A0%20%EB%8C%80%ED%99%94%EC%B2%B4%20%EC%9D%8C%EC%84%B1%EC%9D%B8%EC%8B%9D%20%EB%8D%B0%EC%9D%B4%ED%84%B0&aihubDataSe=data&dataSetSn=463)에서 별도로 다운로드하실 수 있습니다.

## 라이선스
본 데이터셋은 [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/) 라이선스를 따릅니다.


## 인용
```bibtex
@misc{BACKCast,
    title={BACKCast: Backchannel Annotation Dataset for Korean Broadcast Conversational Speech},
    author={Seung Hi Kim, Yong-Seok Choi and Sung Yup Lee},
    year={2026},
    howpublished={\url{https://github.com/etri/etri-miai}},
}
```

## 문의
본 데이터셋과 관련하여 문의 사항이 있으시면, 이슈를 등록하시거나 아래 이메일로 연락해 주시기 바랍니다.

📧 [yseokchoi@etri.re.kr](mailto:yseokchoi@etri.re.kr)