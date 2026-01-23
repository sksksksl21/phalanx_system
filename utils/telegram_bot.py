import requests
import logging

logger = logging.getLogger("TelegramBot")

class TelegramBot:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}/"
        self.timeout = 30
        self.last_update_id = 0  # ★ 핵심: 마지막으로 읽은 메시지 ID 저장

    def send_message(self, text):
        if not self.token or not self.chat_id:
            return

        url = self.base_url + "sendMessage"
        data = {"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"}

        try:
            response = requests.post(url, data=data, timeout=self.timeout)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"텔레그램 전송 실패 (무시하고 진행): {e}")
            pass 

    def get_latest_command(self):
        try:
            url = self.base_url + "getUpdates"
            
            # ★ 핵심 변경: offset을 '마지막 ID + 1'로 설정하여 처리한 메시지는 다시 안 받음
            params = {
                "offset": self.last_update_id + 1, 
                "limit": 1, 
                "timeout": 5 # 수신 대기는 짧게
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data["ok"] and data["result"]:
                # 가장 최신 메시지 하나 가져오기
                update = data["result"][0]
                
                # ★ 처리한 메시지 ID 갱신 (이제 이 메시지는 다시 안 봄)
                self.last_update_id = update["update_id"]
                
                return update["message"].get("text", "")
                
        except Exception as e:
            # 타임아웃 등 에러나면 그냥 무시 (로그 너무 많이 남기지 않음)
            return None
            
        return None