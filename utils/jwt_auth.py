import jwt
import time
from collections import defaultdict
from utils.config import settings

ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 6

LOGIN_ATTEMPTS: dict[str, list[float]] = defaultdict(list)
MAX_ATTEMPTS = 5
LOCKOUT_SECONDS = 60


def create_jwt_token(user_data: dict) -> str:
    payload = {
        "user_id": user_data["user_id"],
        "username": user_data["username"],
        "role": user_data["role"],
        "iat": int(time.time()),
        "exp": int(time.time()) + JWT_EXPIRY_HOURS * 3600,
    }
    return jwt.encode(payload, settings.APP_SECRET_KEY, algorithm=ALGORITHM)


def validate_jwt_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, settings.APP_SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def check_rate_limit(ip: str) -> bool:
    now = time.time()
    LOGIN_ATTEMPTS[ip] = [t for t in LOGIN_ATTEMPTS[ip] if now - t < LOCKOUT_SECONDS]
    if len(LOGIN_ATTEMPTS[ip]) >= MAX_ATTEMPTS:
        return False
    LOGIN_ATTEMPTS[ip].append(now)
    return True


def get_client_ip() -> str:
    try:
        from streamlit.server.server import Server
        session_id = None
        session_info = Server.get_current()._session_info_by_id
        if session_info:
            for sid, info in session_info.items():
                if info.ws is not None and info.ws.request.remote_ip:
                    return info.ws.request.remote_ip
    except Exception:
        pass
    return "unknown"
