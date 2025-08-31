import bcrypt
from utils.db import get_db_connection

def _is_bcrypt_hash(value: str) -> bool:
    return isinstance(value, str) and value.startswith("$2")

def hash_all_passwords():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT user_id, password_hash FROM users")
    users = cur.fetchall()

    for user_id, pw in users:
        # only hash plain text passwords
        if not _is_bcrypt_hash(pw):
            hashed_pw = bcrypt.hashpw(pw.encode('utf-8'), bcrypt.gensalt())
            cur.execute(
                "UPDATE users SET password_hash=%s WHERE user_id=%s",
                (hashed_pw.decode('utf-8'), user_id)
            )

    conn.commit()
    cur.close()
    conn.close()
    print("✅ Plain passwords hashed successfully! Already-hashed ones left intact.")

if __name__ == "__main__":
    hash_all_passwords()
