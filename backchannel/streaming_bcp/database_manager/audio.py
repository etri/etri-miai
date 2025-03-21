import subprocess
from locator import Resource


class AudioTrimmer:
    @classmethod
    def trim(self, name, dialogue, speaker, uid, start, end):

        # output audio path mkdir
        Resource.if_not_exist_mkdir(Resource.espnet_dir() / "audios")

        subprocess.run(
            [
                "sox",
                Resource.wav_path(name, dialogue, speaker),
                Resource.espnet_audio_path(uid),
                "trim",
                f"{start}s",
                f"{end-start}s",
            ],
        )
