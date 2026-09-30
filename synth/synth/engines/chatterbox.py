"""chatterbox — запасной вариант сильной атаки. Пока заглушка.

Нужен, только если XTTS упрётся в лицензию: chatterbox под MIT, а
XTTS-v2 под некоммерческой CPML.

Перед тем как это реализовывать, надо закрыть вопрос из docs/MODELS.md.
В chatterbox встроен нейросетевой водяной знак Perth, который, по
заявлению авторов, переживает сжатие в mp3. Если знак доживает до
выхода, детектор научится ловить его, а не артефакты синтеза, и все
метрики проекта окажутся фикцией.

Порядок действий:
  1. синтезировать фразу, сохранить wav;
  2. прогнать её детектором водяного знака Perth — знак находится?
  3. если да — отключить вотермаркинг и повторить проверку;
  4. только после этого включать движок в прогоны.
"""
from __future__ import annotations

from .base import SynthEngine, Utterance

_NOT_READY = (
    "chatterbox ещё не подключён. Сначала проверка водяного знака Perth — "
    "см. комментарий в начале synth/synth/engines/chatterbox.py и docs/MODELS.md."
)


class ChatterboxEngine(SynthEngine):
    name = "chatterbox"

    def load(self) -> None:
        raise NotImplementedError(_NOT_READY)

    def say(self, text: str) -> Utterance:
        raise NotImplementedError(_NOT_READY)
