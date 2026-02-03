# utils/telegram_bot.py
import time
import requests
import logging

logger = logging.getLogger("TelegramBot")


class TelegramBot:
    """
    [Phalanx Utils] TelegramBot (single implementation, sender + optional receiver)

    - send_message(text): HTML parse_mode
    - send(title, lines): 표준 포맷 전송
    - get_latest_command(): offset 기반으로 중복 수신 방지 (옵션)
    - last_update_id는 엔진 state에 저장/복구해서 재시작 정합성 유지 가능
    """

    def __init__(
        self,
        token: str,
        chat_id: str,
        timeout: int = 30,
        parse_mode: str = "HTML",
        tag: str = "PHALANX",
        dedup_window_sec: float = 15.0,
    ):
        self.token = token or ""
        self.chat_id = chat_id or ""
        self.base_url = f"https://api.telegram.org/bot{self.token}/"
        self.timeout = int(timeout)
        self.parse_mode = str(parse_mode)
        self.tag = str(tag)

        # receiver state
        self.last_update_id = 0

        # dedup
        self._last_text = None
        self._last_ts = 0.0
        self._dedup_window_sec = float(dedup_window_sec)

        self._sess = requests.Session()

    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def set_last_update_id(self, update_id: int):
        try:
            self.last_update_id = int(update_id)
        except Exception:
            pass

    def get_last_update_id(self) -> int:
        try:
            return int(self.last_update_id)
        except Exception:
            return 0

    def _dedup_ok(self, text: str) -> bool:
        try:
            now = time.time()
            if self._last_text == text and (now - self._last_ts) < self._dedup_window_sec:
                return False
            self._last_text = text
            self._last_ts = now
            return True
        except Exception:
            return True

    def send_message(self, text: str):
        if not self.enabled():
            return False

        if text is None:
            return False

        text = str(text)

        if not self._dedup_ok(text):
            return False

        url = self.base_url + "sendMessage"
        data = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": self.parse_mode,
            "disable_web_page_preview": True,
        }

        try:
            resp = self._sess.post(url, data=data, timeout=self.timeout)
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"텔레그램 전송 실패 (무시하고 진행): {e}")
            return False

    def send(self, title: str, lines: list):
        """
        표준 알림 포맷:
        <b>[TAG] TITLE</b>
        key=value
        key=value
        """
        if not self.enabled():
            return False

        try:
            t = str(title or "ALERT")
            body = "\n".join([str(x) for x in (lines or [])])
            msg = f"<b>[{self.tag}] {t}</b>\n{body}" if body else f"<b>[{self.tag}] {t}</b>"
            return self.send_message(msg)
        except Exception:
            return False

    def get_latest_command(self):
        """
        최신 메시지 1개만 폴링.
        - offset = last_update_id + 1 로 중복 수신 방지
        - timeout은 짧게 (엔진 루프를 막지 않게)
        """
        if not self.enabled():
            return None

        try:
            url = self.base_url + "getUpdates"
            params = {
                "offset": self.last_update_id + 1,
                "limit": 1,
                "timeout": 5,
            }
            resp = self._sess.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if data.get("ok") and data.get("result"):
                update = data["result"][0]
                try:
                    self.last_update_id = int(update.get("update_id", self.last_update_id))
                except Exception:
                    pass

                msg = update.get("message") or {}
                text = msg.get("text", "")
                return text

        except Exception:
            return None

        return None
