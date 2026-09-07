"""
Qwen2-Audio용 프롬프트 빌더

태스크 정의에 맞는 프롬프트를 생성하며,
raw prompt / chat template 분기, base / instruct 모델별 variant를 관리한다.

프롬프트 텍스트는 config(prompt_templates)에서 관리하며,
prompt_version에 따라 자동 선택된다.
"""

import logging
from typing import Dict, List, Optional, Sequence

from .constants import VALID_LABELS, LABEL_CHOICES_STR

logger = logging.getLogger(__name__)

# SWBD label set (영어, 3-class)
SWBD_LABELS = ["NOBC", "CONTINUER", "ASSESSMENT"]
SWBD_LABEL_CHOICES_STR = " / ".join(SWBD_LABELS)

# 기본 프롬프트 템플릿 레지스트리
# config에서 prompt_templates를 지정하지 않으면 이 값을 사용한다.
DEFAULT_PROMPT_TEMPLATES: Dict[str, dict] = {
    "v1": {
        "context": "위 오디오는 대화 중 화자와 청자의 음성을 믹싱한 것이다.",
        "speaker_text_format": '화자 발화: "{speaker_text}"',
        "listener_text_format": '청자 발화/반응: "{listener_text}"',
        "instruction": f"청자의 백채널 유형을 다음 중 하나만 출력하라: {LABEL_CHOICES_STR}",
        "system_prompt": (
            "당신은 음성 대화에서 청자의 백채널 반응 유형을 분류하는 전문가입니다. "
            f"반드시 {LABEL_CHOICES_STR} 중 하나만 출력하십시오."
        ),
    },
    "v2": {
        "context": "다음은 화자와 청자의 대화 음성이다.",
        "speaker_text_format": '화자가 말한 내용: "{speaker_text}"',
        "listener_text_format": '청자가 보인 반응: "{listener_text}"',
        "instruction": (
            "청자의 반응을 아래 4가지 중 하나로 분류하시오. 설명 없이 클래스 이름만 출력하시오.\n"
            f"- {LABEL_CHOICES_STR}"
        ),
        "system_prompt": (
            f"음성 대화 분석 전문가. 반드시 {LABEL_CHOICES_STR} 중 하나만 답하시오."
        ),
    },
    "v_separated": {
        "context": "두 오디오는 같은 시점의 대화 구간에서 분리한 화자/청자 음성이다.",
        "speaker_audio_context": "첫 번째 오디오는 화자의 발화 구간이다.",
        "listener_audio_context": "두 번째 오디오는 같은 시점의 청자 반응 구간이다.",
        "speaker_text_format": '화자 발화 중 일부: "{speaker_text}"',
        "listener_text_format": '청자 발화/반응 텍스트: "{listener_text}"',
        "instruction": f"두 오디오를 함께 참고하여 청자의 백채널 유형을 다음 중 하나만 출력하라: {LABEL_CHOICES_STR}",
        "system_prompt": (
            "당신은 음성 대화에서 청자의 백채널 반응 유형을 분류하는 전문가입니다. "
            f"반드시 {LABEL_CHOICES_STR} 중 하나만 출력하십시오."
        ),
    },
    # English templates for SWBD (3-class: NOBC / CONTINUER / ASSESSMENT)
    "v_separated_en": {
        "context": "Two audio clips come from the same moment of a conversation — one for the speaker, one for the listener.",
        "speaker_audio_context": "The first audio is the speaker's speech.",
        "listener_audio_context": "The second audio is the listener's response at the same moment.",
        "speaker_text_format": 'Part of the speaker\'s utterance: "{speaker_text}"',
        "listener_text_format": 'Listener\'s words/response text: "{listener_text}"',
        "instruction": (
            "Using both audios, classify the listener's backchannel type. "
            f"Output exactly one of: {SWBD_LABEL_CHOICES_STR}."
        ),
        "system_prompt": (
            "You are an expert at classifying a listener's backchannel response in spoken dialogue. "
            f"Always reply with exactly one of {SWBD_LABEL_CHOICES_STR}."
        ),
    },
}

FEWSHOT_EXAMPLES = [
    {
        "audio_path": "/home/yseokchoi/projs/miai/processed_data_v2/BCKO/batch/batch_detection/mixing_audios/Counseling.SelectStar.2023.5ST.내담자.0185.000002.wav",
        "speaker_text": "너무 어려워요.",
        "label": "NOBC"
    },
    {
        "audio_path": "/home/yseokchoi/projs/miai/processed_data_v2/BCKO/batch/batch_detection/mixing_audios/Counseling.SelectStar.2023.5ST.내담자.0187.000004.wav",
        "speaker_text": "많이 아기가 많이 울어가지고",
        "label": "CONTINUER"
    },
    {
        "audio_path": "/home/yseokchoi/projs/miai/processed_data_v2/BCKO/batch/batch_detection/mixing_audios/Counseling.SelectStar.2023.5ST.내담자.0190.000028.wav",
        "speaker_text": "그런 활동을 거기 가서 하니까",
        "label": "UNDERSTANDING"
    },
    {
        "audio_path": "/home/yseokchoi/projs/miai/processed_data_v2/BCKO/batch/batch_detection/mixing_audios/Counseling.SelectStar.2023.5ST.내담자.0392.000029.wav",
        "speaker_text": "그 말씀을 드린건데, 근데 둘째",
        "label": "EMPATHY"
    }
]

class QwenPromptBuilder:
    """
    Qwen2-Audio용 프롬프트 생성기

    - raw prompt 모드: <|audio_bos|><|AUDIO|><|audio_eos|> + 텍스트
    - chat template 모드: conversation dict 리스트 반환
    - use_speaker_text 옵션으로 화자 텍스트 포함 여부 제어
    - prompt_version에 따라 템플릿 자동 선택
    """

    def __init__(
        self,
        use_chat_template: bool = True,
        use_speaker_text: bool = False,
        use_listener_text: bool = False,
        model_type: str = "instruct",
        prompt_version: str = "v1",
        prompt_templates: Optional[Dict[str, dict]] = None,
        fewshot: int = 0,
        audio_input_mode: str = "mixed",
    ):
        """
        Args:
            use_chat_template: True면 chat template용 conversation dict 반환,
                               False면 raw prompt 문자열 반환
            use_speaker_text: True면 화자 발화 텍스트를 프롬프트에 포함
            use_listener_text: True면 청자 발화/반응 텍스트도 프롬프트에 포함
            model_type: "base" 또는 "instruct"
            prompt_version: 사용할 프롬프트 템플릿 버전 키
            prompt_templates: 프롬프트 템플릿 dict (None이면 DEFAULT_PROMPT_TEMPLATES 사용)
            audio_input_mode: "mixed" 또는 "separated"
        """
        if model_type not in ("base", "instruct"):
            raise ValueError(f"model_type은 'base' 또는 'instruct'여야 합니다: {model_type}")
        if audio_input_mode not in ("mixed", "separated"):
            raise ValueError(
                f"audio_input_mode은 'mixed' 또는 'separated'여야 합니다: {audio_input_mode}"
            )

        self.use_chat_template = use_chat_template
        self.use_speaker_text = use_speaker_text
        self.use_listener_text = use_listener_text
        self.model_type = model_type
        self.prompt_version = prompt_version
        self.fewshot = fewshot
        self.audio_input_mode = audio_input_mode

        # 템플릿 레지스트리 구성
        self.templates = prompt_templates or DEFAULT_PROMPT_TEMPLATES

        if prompt_version not in self.templates:
            available = list(self.templates.keys())
            raise ValueError(
                f"prompt_version '{prompt_version}'이 템플릿에 없습니다. "
                f"사용 가능: {available}"
            )

        self.current_template = self.templates[prompt_version]

        logger.info(
            f"QwenPromptBuilder 초기화: chat_template={use_chat_template}, "
            f"speaker_text={use_speaker_text}, listener_text={use_listener_text}, "
            f"model={model_type}, "
            f"version={prompt_version}, fewshot={fewshot}, "
            f"audio_input_mode={audio_input_mode}",
        )

    def build(
        self,
        audio_path: Optional[str] = None,
        audio_paths: Optional[Sequence[str]] = None,
        speaker_text: Optional[str] = None,
        listener_text: Optional[str] = None,
        support_examples: Optional[Sequence[dict]] = None,
    ):
        """
        단일 샘플에 대한 프롬프트를 생성한다.

        Args:
            audio_path: 믹싱 오디오 파일 경로 (mixed mode)
            audio_paths: 오디오 파일 경로 리스트
                         - mixed: 길이 1
                         - separated: [speaker_audio_path, listener_audio_path]
            speaker_text: 화자 발화 텍스트 (use_speaker_text=True일 때 사용)
            listener_text: 청자 발화/반응 텍스트 (use_listener_text=True일 때 사용)
            support_examples: dynamic few-shot 예시 리스트. 각 예시는
                              {audio_paths, speaker_text, listener_text, label}을 포함한다.

        Returns:
            use_chat_template=True: conversation dict 리스트
            use_chat_template=False: raw prompt 문자열
        """
        if support_examples and self.fewshot > 0:
            raise ValueError("static fewshot과 dynamic support_examples는 동시에 사용할 수 없습니다")

        instruction = self._build_instruction(speaker_text, listener_text)
        resolved_audio_paths = self._resolve_audio_paths(audio_path, audio_paths)

        if self.use_chat_template:
            return self._build_chat_template(
                resolved_audio_paths,
                instruction,
                speaker_text=speaker_text,
                listener_text=listener_text,
                support_examples=support_examples,
            )
        else:
            if support_examples:
                raise ValueError("dynamic few-shot은 chat template 모드에서만 지원합니다")
            return self._build_raw_prompt(instruction)

    def _resolve_audio_paths(
        self,
        audio_path: Optional[str],
        audio_paths: Optional[Sequence[str]],
    ) -> List[str]:
        """audio_input_mode에 맞는 오디오 경로 리스트를 검증한다."""
        if audio_paths is None:
            if audio_path is None:
                raise ValueError("audio_path 또는 audio_paths 중 하나는 필요합니다")
            audio_paths = [audio_path]

        resolved = [str(path) for path in audio_paths]

        if self.audio_input_mode == "mixed":
            if len(resolved) != 1:
                raise ValueError(f"mixed mode는 audio 1개가 필요합니다: {len(resolved)}")
            return resolved

        if len(resolved) != 2:
            raise ValueError(
                f"separated mode는 [speaker, listener] audio 2개가 필요합니다: {len(resolved)}"
            )
        return resolved

    def _build_instruction(
        self,
        speaker_text: Optional[str] = None,
        listener_text: Optional[str] = None,
    ) -> str:
        """현재 템플릿으로 태스크 지시문을 생성한다.

        speaker_text_position / listener_text_position 이 "after_*_audio" 인 경우
        해당 텍스트는 instruction에 포함하지 않고 audio_content path에서 처리한다.
        """
        t = self.current_template
        parts = [t["context"]]

        st_pos = t.get("speaker_text_position", "instruction")
        if self.use_speaker_text and speaker_text and st_pos == "instruction":
            parts.append(
                t["speaker_text_format"].format(speaker_text=speaker_text)
            )

        lt_pos = t.get("listener_text_position", "instruction")
        if self.use_listener_text and listener_text and lt_pos == "instruction":
            fmt = t.get("listener_text_format")
            if fmt is None:
                fmt = 'Listener text: "{listener_text}"'
            parts.append(fmt.format(listener_text=listener_text))

        parts.append(t["instruction"])

        return "\n".join(parts)

    def _format_speaker_text_for_audio(self, speaker_text: Optional[str]) -> Optional[str]:
        """ST가 speaker audio 직후 별도 text item으로 들어갈 때 텍스트 생성."""
        t = self.current_template
        if (
            self.use_speaker_text
            and speaker_text
            and t.get("speaker_text_position") == "after_speaker_audio"
        ):
            return t["speaker_text_format"].format(speaker_text=speaker_text)
        return None

    def _format_listener_text_for_audio(self, listener_text: Optional[str]) -> Optional[str]:
        """LT가 listener audio 직후 별도 text item으로 들어갈 때 텍스트 생성."""
        t = self.current_template
        if (
            self.use_listener_text
            and listener_text
            and t.get("listener_text_position") == "after_listener_audio"
        ):
            fmt = t.get("listener_text_format") or 'Listener text: "{listener_text}"'
            return fmt.format(listener_text=listener_text)
        return None

    def _build_audio_content(
        self,
        audio_paths: Sequence[str],
        instruction: str,
        speaker_text: Optional[str] = None,
        listener_text: Optional[str] = None,
    ) -> list:
        """현재 audio_input_mode에 맞는 user content list를 생성한다.

        ST / LT 의 *_text_position 가 "after_*_audio"면 각 audio 직후에 별도 text
        item으로 삽입한다 (early-fusion 의도).
        """
        if self.audio_input_mode == "separated":
            t = self.current_template
            content = [
                {
                    "type": "text",
                    "text": t.get("speaker_audio_context", "첫 번째 오디오(화자 발화):"),
                },
                {"type": "audio", "audio_url": audio_paths[0]},
            ]
            st_after = self._format_speaker_text_for_audio(speaker_text)
            if st_after:
                content.append({"type": "text", "text": st_after})

            content.append({
                "type": "text",
                "text": t.get("listener_audio_context", "두 번째 오디오(청자 반응):"),
            })
            content.append({"type": "audio", "audio_url": audio_paths[1]})

            lt_after = self._format_listener_text_for_audio(listener_text)
            if lt_after:
                content.append({"type": "text", "text": lt_after})

            content.append({"type": "text", "text": instruction})
            return content

        return [
            {"type": "audio", "audio_url": audio_paths[0]},
            {"type": "text", "text": instruction},
        ]

    def _build_chat_template(
        self,
        audio_paths: Sequence[str],
        instruction: str,
        speaker_text: Optional[str] = None,
        listener_text: Optional[str] = None,
        support_examples: Optional[Sequence[dict]] = None,
    ) -> list:
        """chat template용 conversation dict를 생성한다."""
        if self.audio_input_mode == "separated":
            if self.fewshot > 0 and not support_examples:
                raise ValueError("separated mode에서는 fewshot 예제를 아직 지원하지 않습니다")

        conversation = [
            {
                "role": "user",
                "content": self._build_audio_content(
                    audio_paths,
                    instruction,
                    speaker_text=speaker_text,
                    listener_text=listener_text,
                ),
            }
        ]

        if support_examples:
            example_conversation = []
            for support in support_examples:
                support_paths = self._resolve_audio_paths(
                    support.get("audio_path"),
                    support.get("audio_paths"),
                )
                support_instruction = self._build_instruction(
                    speaker_text=support.get("speaker_text"),
                    listener_text=support.get("listener_text"),
                )
                example_conversation.extend([
                    {
                        "role": "user",
                        "content": self._build_audio_content(
                            support_paths,
                            support_instruction,
                            speaker_text=support.get("speaker_text"),
                            listener_text=support.get("listener_text"),
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": support["label"],
                    },
                ])

            conversation = example_conversation + conversation
        
        if self.fewshot > 0:
            example_conversation = [
            ]
            
            for example in FEWSHOT_EXAMPLES[:self.fewshot]:
                example = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio_url": example["audio_path"]},
                            {"type": "text", "text": self._build_instruction(speaker_text=example["speaker_text"])},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": example["label"],
                    }
                ]
                example_conversation.extend(example)

            conversation = example_conversation + conversation


        if self.model_type == "instruct":
            conversation.insert(
                0,
                {
                    "role": "system",
                    "content": self.current_template["system_prompt"],
                },
            )

        return conversation

    def _build_raw_prompt(self, instruction: str) -> str:
        """raw prompt 문자열을 생성한다."""
        audio_tokens = "<|audio_bos|><|AUDIO|><|audio_eos|>"
        if self.audio_input_mode == "separated":
            t = self.current_template
            audio_tokens = (
                f"{t.get('speaker_audio_context', '첫 번째 오디오(화자 발화):')} "
                "<|audio_bos|><|AUDIO|><|audio_eos|>\n"
                f"{t.get('listener_audio_context', '두 번째 오디오(청자 반응):')} "
                "<|audio_bos|><|AUDIO|><|audio_eos|>"
            )
        return f"{audio_tokens}\n{instruction}"

    def get_metadata(self) -> dict:
        """프롬프트 설정 메타데이터를 반환한다 (결과 저장용)."""
        return {
            "prompt_version": self.prompt_version,
            "use_chat_template": self.use_chat_template,
            "use_speaker_text": self.use_speaker_text,
            "use_listener_text": self.use_listener_text,
            "model_type": self.model_type,
            "audio_input_mode": self.audio_input_mode,
            "fewshot": self.fewshot,
            "prompt_template": self.current_template,
        }
