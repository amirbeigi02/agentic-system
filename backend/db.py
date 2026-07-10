"""
مدیریت دیتابیس SQLite برای Agent Registry و Memory
"""
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

DB_PATH = Path(__file__).parent / "data" / "system.db"
DB_PATH.parent.mkdir(exist_ok=True)


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT NOT NULL,
                system_prompt TEXT NOT NULL,
                tools TEXT DEFAULT '[]',
                is_builtin INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                agent_name TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact TEXT NOT NULL,
                source_agent TEXT,
                created_at TEXT NOT NULL
            )
        """)

        seed_agents = [
            (
                "general_agent",
                "کارهای عمومی، سوال و جواب، نوشتن متن که به تخصص خاصی نیاز ندارد",
                "تو یک دستیار عمومی و مفید هستی. کوتاه، دقیق و مستقیم پاسخ بده.",
            ),
        ]
        for name, desc, prompt in seed_agents:
            conn.execute(
                """INSERT OR IGNORE INTO agents (name, description, system_prompt, is_builtin, created_at)
                   VALUES (?, ?, ?, 1, ?)""",
                (name, desc, prompt, datetime.utcnow().isoformat()),
            )


def list_agents():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM agents ORDER BY id").fetchall()
        return [dict(r) for r in rows]


def get_agent(name: str):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM agents WHERE name = ?", (name,)).fetchone()
        return dict(row) if row else None


def create_agent(name: str, description: str, system_prompt: str, tools=None):
    with get_db() as conn:
        conn.execute(
            """INSERT INTO agents (name, description, system_prompt, tools, is_builtin, created_at)
               VALUES (?, ?, ?, ?, 0, ?)""",
            (name, description, system_prompt, json.dumps(tools or [], ensure_ascii=False),
             datetime.utcnow().isoformat()),
        )


def update_agent(name: str, description: str = None, system_prompt: str = None):
    with get_db() as conn:
        agent = conn.execute("SELECT * FROM agents WHERE name = ?", (name,)).fetchone()
        if not agent:
            return False
        new_desc = description or agent["description"]
        new_prompt = system_prompt or agent["system_prompt"]
        conn.execute(
            "UPDATE agents SET description = ?, system_prompt = ? WHERE name = ?",
            (new_desc, new_prompt, name),
        )
        return True


def delete_agent(name: str):
    with get_db() as conn:
        row = conn.execute("SELECT is_builtin FROM agents WHERE name = ?", (name,)).fetchone()
        if not row or row["is_builtin"]:
            return False
        conn.execute("DELETE FROM agents WHERE name = ?", (name,))
        return True


def add_message(session_id: str, role: str, content: str, agent_name: str = None):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, agent_name, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, agent_name, datetime.utcnow().isoformat()),
        )


def get_recent_messages(session_id: str, limit: int = 30):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]


def add_fact(fact: str, source_agent: str = None):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO memory_facts (fact, source_agent, created_at) VALUES (?, ?, ?)",
            (fact, source_agent, datetime.utcnow().isoformat()),
        )


def get_all_facts(limit: int = 50):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM memory_facts ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
