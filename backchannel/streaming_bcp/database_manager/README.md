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
        - {DB이름}.{vendor}.{연도}.{version}.xlsx      => 기본 label
        
    - split/
        - {DB이름}.{vendor}.{연도}.yaml
            대화 단위 split 정보
    
    - meta/ # 현재는 툴에서는 참조하지 않는 정보임.
        - {DB이름}.{vendor}.{연도}.yaml

    - espent/
        - {config 이름} 이하 디렉토리에 espnet 데이터 생성


* configuration 파일은 스크립트 파일의 위치를 기준으로 찾아감.
