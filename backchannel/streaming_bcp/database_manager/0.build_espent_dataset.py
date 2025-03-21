"""
입수한 DB를 정리해서, 통합 데이터베이스를 생성함.

1. database archiving
2. transcript 정리 => symbol 통합
3. label 형식도 통합 하고
4. 통합 database에 추가하고
"""

"""
BCKO/
    - raw/  # 데이터 베이스 원본
    - audio/        # 오디오 파일
        - {DB이름}.{vendor}.{연도}/
            - {대화이름}.speaker={speaker}.wav  => stereo db
            - {대화이름}.speaker=both.wav       => mono db ?
    
    - label/    # xlsx spread sheet로 관리. 대화 이름은 sheet name으로
        - norm/     # normalized transcription
            - {DB이름}.{vendor}.{연도}.norm.{version}.xlsx => transcript 정규화
        - word/     # word aligned transcription
            - {DB이름}.{vendor}.{연도}.word.{version}.xlsx => word 단위 timestamp
        - boundary/ # boundary가 표시된 transcription
            - {DB이름}.{vendor}.{연도}.boundary.{version}.xlsx => boundary가 표시된 text
        - {DB이름}.{vendor}.{연도}.{version}.xlsx      => 기본 label
        
    - split/
        - {DB이름}.{vendor}.{연도}.yaml
            대화 단위 split 정보
    
    - meta/ # 현재는 툴에서는 참조하지 않는 정보임.
        - {DB이름}.{vendor}.{연도}.yaml

    - espent/
        - {config 이름} 이하 디렉토리에 espnet 데이터 생성


* configuration 파일은 스크립트 파일의 위치를 기준으로 찾아감.

"""
from tqdm.auto import tqdm
import argparse
import itertools

from config import Config
from locator import Resource
from transcript import Transcript
from utterance import Utterance, LabelWriter
from transcript_filter import Filter
from audio import AudioTrimmer


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, default="baseline")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    # config 이름을 읽어와서, 설정.
    Config.load(args.name, verbose=True)

    # espnet 포함 하위 디렉토리 데이터 삭제
    if args.clean:
        Resource.clean()

    names = [Resource.name("Example", None, None)]
    mono = False

    # DB 순회
    for name in (pbar := tqdm(names)):
        pbar.set_description(name)

        results = list()

        # 다이얼로그 순회
        for dialogue in (pbar2 := tqdm(Resource.dialogues(name), leave=False)):
            pbar2.set_description(dialogue)

            # norm.xlsx
            norm = Resource.norm(name, dialogue)
            # apply filter to label
            for filter_policy in Config.policy["filter"]:
                norm = Filter.get(**filter_policy).filter(norm)

            # wrap
            transcript = Transcript(
                norm,
                # word=Resource.word(name, dialogue),
                # boundary=Resource.boundary(name, dialogue),
            )

            # front-channel utterance 순회
            for i, utt in transcript.frontchannels().iterrows():
                pbar.set_postfix(
                    dict(dialogue=dialogue, id=f"{utt.speaker}-{utt.segment_idx}")
                )

                # training utterance 생성
                training_utt = Utterance(
                    name,
                    dialogue,
                    frontchannel=utt,
                    # 해당 front-channel utterance 뒤에 따라오는 front-channel utterance 수집
                    following=transcript.following(utt),  # 없으면 None
                    # 해당 front-channel utterance에 대한 BC 반응 utterance 수집
                    backchannels=transcript.bc_responses(utt),  # 없으면 None
                    # 해당 front-channel utterance에 대한 word align 정보 수집
                    # words=transcript.words(utt),  # 없으면 None
                    # 해당 front-channel utterance에 대한 boundary text 수집
                    # boundary=transcript.boundary(utt),  # 없으면 None
                )

                # audio trimming
                AudioTrimmer.trim(
                    name,
                    dialogue,
                    training_utt.speaker,
                    training_utt.uid,
                    training_utt.start,
                    training_utt.end,
                )

                results.append(
                    [training_utt.uid, training_utt.text, training_utt.labels]
                )

        # sort by uid
        results = sorted(results, key=lambda x: x[0])

        writer = LabelWriter()

        writer.write(results)
