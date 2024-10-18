import os
from abc import ABC, abstractmethod

import pytest

from transcription_benchmarks.inference.systran_faster_whisper_large_v2.inference import (
    BatchArgs,
    ModelArgs,
    Segment,
    SingleArgs,
    TranscriptionInfo,
)
from transcription_benchmarks.misc.get_test_audio import AudioFilename
from transcription_benchmarks.util.predict import predict_aws, predict_local

LOCAL = "LOCAL"

AWS_ENDPOINT = "AWS_ENDPOINT"


def local_in_env() -> bool:
    return LOCAL in os.environ


def to_str(segments: list[Segment]) -> str:
    return "".join(s.text for s in segments)


def aws_endpoint_in_env() -> bool:
    return AWS_ENDPOINT in os.environ


def get_aws_endpoint() -> str:
    return os.environ[AWS_ENDPOINT]


class BaseTestCase(ABC):
    @abstractmethod
    def predict(
        self, test_file: AudioFilename, params: ModelArgs | None = None
    ) -> tuple[list[Segment], TranscriptionInfo]: ...

    def test_language_detection(self):
        segments, info = self.predict(
            "1min.flac", params=ModelArgs(task="detect_language")
        )
        assert len(segments) == 0
        assert info.language == "en"
        assert info.language_probability == pytest.approx(0.98291015625)

    def test_without_params(self):
        segments, info = self.predict("1min.flac")
        assert to_str(segments).startswith(
            " Ladies and gentlemen, thank you for being here and for your written representations. You know what the purpose of this select committee is. We are exploring the risks, the ways in which deliberate online falsehoods are spread, and we are primarily restricting ourselves to deliberate online falsehoods."
        )

    def test_single_args(self):
        segments, info = self.predict(
            "1min.flac", params=ModelArgs(single_params=SingleArgs())
        )
        assert to_str(segments).startswith(
            " Ladies and gentlemen, thank you for being here and for your written representations. You know what the purpose of this select committee is. We are exploring the risks, the ways in which deliberate online falsehoods are spread, and we are primarily restricting ourselves to deliberate online falsehoods."
        )

    def test_batch_args(self):
        segments, info = self.predict(
            "1min.flac", params=ModelArgs(batch_params=BatchArgs(batch_size=6))
        )
        assert to_str(segments).startswith(
            " Ladies and gentlemen, thank you for being here and for your written representations. You know what the purpose of this select committee is. We are exploring the risks, the ways in which deliberate online falsehoods are spread, and we are primarily restricting ourselves to deliberate online falsehoods. We call them DOFs. And what we should do about the situation. So I think it's useful to be clear about the approach we are going to take in these discussions."
        )
        assert len(segments)

    def test_noisy_with_default_language_detection(self):
        segments, _ = self.predict(
            "noisy.flac", params=ModelArgs(single_params=SingleArgs(temperature=0))
        )
        assert (
            to_str(segments)
            == " Ya, jadi anda boleh menggunakan bahan-bahan tersebut secara berkata-kata Baiklah, saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa Saya ingin menunjukkan kepada anda bahawa"
        )

    def test_noisy_with_language_forced_to_en(self):
        segments, _ = self.predict(
            "noisy.flac",
            params=ModelArgs(single_params=SingleArgs(language="en", temperature=0)),
        )
        assert (
            to_str(segments)
            == " Yeah, so you can just put that there, very good, now that's good. Okay. 这个给你看那个,那个大厂. 大厂? I don't know. 大厂? 一直来啊,小猪,一直在弄。 现在是? 没有啦,一直在弄。 Oh, okay. Okay, we arranged for her not to worry. That was the thought of her. There's something that happened with you, that's why she kept saying that. 没事啦,没事,没事。 That's what I meant to say. 你爱听,看我都不信。 哦,我明白了。 没在跟你讲话。 我没有讲话。 没在跟你讲话。 I want to look at more of your poh buak. Your poh buak. Your poh buak. No, it's just not worth the money. Got any cream for this? This one actually no cream, because this one is not a cream can solve. This one need to cut. It's actually whatever... 爱果啊? 爱果啊? Yeah, it's a cream won't drip lah. But this one would not cost any... Any what? Very serious. We need to check to see what it is. Now we don't know what it is, you see. So we're gonna have to check on that, not to worry. Okay? Thank you lah. 没问题,bye bye!"
        )

    def test_predicting_same_text_multiple_times_should_give_same_output(self):
        preds = [
            to_str(t[0])
            for t in [
                self.predict(
                    "noisy.flac",
                    params=ModelArgs(single_params=SingleArgs(temperature=0)),
                )
            ]
            * 3
        ]
        assert all(x.startswith(" Ya, jadi anda boleh menggunakan b") for x in preds)
        assert len(set(preds)) == 1

    def test_predicting_same_text_multiple_times_should_give_same_output_batch(self):
        preds = [
            to_str(t[0])
            for t in [
                self.predict(
                    "chinese.flac",
                    params=ModelArgs(batch_params=BatchArgs(temperature=0)),
                )
            ]
            * 3
        ]
        assert all(
            x
            == "你好,我是从国泰医院打来的陈医生,我是肾脏医生,想跟你谈一下爸爸的情况,现在适合跟你讲话吗?知道你爸爸有肾脏衰落问题对吗?这是第五期,在医院里面你经常会看到这个肾脏科医生。那現在爸爸進來的情況就是肺積水然後積水之後就是有明顯非常喘然後氧氣也不夠我們測量的時候那覺得說因為爸爸的情況那個腎功能已經非常不好那可能就是需要就是去開始這個腹膜吐息或者血液吐息來幫忙這個肺積水的問題啦你明白我在講什麼嗎"
            for x in preds
        )
        assert len(set(preds)) == 1


@pytest.mark.skipif(
    not local_in_env(), reason="Only run when there is a local endpoint"
)
class TestLocalEndpoint(BaseTestCase):
    def predict(
        self, test_file: AudioFilename, params: ModelArgs | None = None
    ) -> tuple[list[Segment], TranscriptionInfo]:
        return predict_local(test_file, params)


@pytest.mark.skipif(
    not aws_endpoint_in_env(), reason=f"Only run when {AWS_ENDPOINT} is supplied"
)
class TestAwsEndpoint(BaseTestCase):
    def predict(
        self, test_file: AudioFilename, params: ModelArgs | None = None
    ) -> tuple[list[Segment], TranscriptionInfo]:
        return predict_aws(get_aws_endpoint(), test_file, params)
