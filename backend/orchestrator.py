"""
Orchestrator: تنها نقطه‌ی ورودی سیستم (نسخه‌ی Groq - رایگان).
"""
import os
import json
from groq import Groq

import db

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = MODEL = "llama-3.1-8b-instant"



def _tools_schema():
    return [
        {
            "type": "function",
            "function": {
                "name": "delegate_to_agent",
                "description": "کار را به یکی از ایجنت‌های تخصصی موجود در سیستم واگذار می‌کند و نتیجه را برمی‌گرداند.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_name": {"type": "string", "description": "نام دقیق ایجنت از لیست موجود"},
                        "task": {"type": "string", "description": "شرح کامل و دقیق کاری که ایجنت باید انجام دهد"},
                    },
                    "required": ["agent_name", "task"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_agent",
                "description": "یک ایجنت تخصصی جدید می‌سازد و آن را برای همیشه به سیستم اضافه می‌کند.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "نام یکتای انگلیسی و کوتاه، مثل trader_agent"},
                        "description": {"type": "string", "description": "توضیح یک خطی که چه زمانی باید این ایجنت صدا زده شود"},
                        "system_prompt": {"type": "string", "description": "دستورالعمل کامل و حرفه‌ای برای این ایجنت به فارسی"},
                    },
                    "required": ["name", "description", "system_prompt"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_agent",
                "description": "system_prompt یا توضیح یک ایجنت موجود را ویرایش می‌کند.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "system_prompt": {"type": "string"},
                    },
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delete_agent",
                "description": "یک ایجنت غیرپایه را حذف می‌کند.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "remember_fact",
                "description": "یک واقعیت مهم و بلندمدت درباره‌ی کاربر یا پروژه‌هایش را در حافظه دائمی ذخیره می‌کند.",
                "parameters": {
                    "type": "object",
                    "properties": {"fact": {"type": "string"}},
                    "required": ["fact"],
                },
            },
        },
    ]


def _run_subagent(agent_name: str, task: str) -> str:
    agent = db.get_agent(agent_name)
    if not agent:
        return f"خطا: ایجنتی با نام '{agent_name}' پیدا نشد."
    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=4000,
        messages=[
            {"role": "system", "content": agent["system_prompt"]},
            {"role": "user", "content": task},
        ],
    )
    return resp.choices[0].message.content or ""


def _execute_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "delegate_to_agent":
        return _run_subagent(tool_input["agent_name"], tool_input["task"])

    if tool_name == "create_agent":
        existing = db.get_agent(tool_input["name"])
        if existing:
            return f"ایجنتی با نام '{tool_input['name']}' از قبل وجود دارد. از update_agent استفاده کن."
        db.create_agent(tool_input["name"], tool_input["description"], tool_input["system_prompt"])
        return f"ایجنت '{tool_input['name']}' با موفقیت ساخته و به سیستم اضافه شد."

    if tool_name == "update_agent":
        ok = db.update_agent(
            tool_input["name"],
            tool_input.get("description"),
            tool_input.get("system_prompt"),
        )
        return "به‌روزرسانی شد." if ok else "ایجنت پیدا نشد."

    if tool_name == "delete_agent":
        ok = db.delete_agent(tool_input["name"])
        return "حذف شد." if ok else "ایجنت پیدا نشد یا پایه (builtin) است و قابل حذف نیست."

    if tool_name == "remember_fact":
        db.add_fact(tool_input["fact"])
        return "در حافظه دائمی ذخیره شد."

    return "تول ناشناخته."


def _build_system_prompt():
    agents = db.list_agents()
    agents_list = "\n".join(f"- {a['name']}: {a['description']}" for a in agents)
    facts = db.get_all_facts(limit=30)
    facts_text = "\n".join(f"- {f['fact']}" for f in facts) or "(هنوز چیزی ذخیره نشده)"

    return f"""تو Orchestrator یک سیستم چند-ایجنتی هستی. کاربر فقط با تو صحبت می‌کند.
وظیفه‌ی تو:
۱. اگر کار به یکی از ایجنت‌های تخصصی زیر مربوط است، با تول delegate_to_agent به او واگذار کن.
۲. اگر کاربر صریحاً خواست ایجنت جدید ساخته شود، با create_agent بسازش (یک system_prompt حرفه‌ای و کامل برایش بنویس).
۳. اگر چیزی مهم و ماندگار درباره کاربر/پروژه‌هایش فهمیدی، با remember_fact ذخیره‌اش کن.
۴. همیشه پاسخ نهایی را خودت به فارسی، خلاصه و مفید برای کاربر بنویس.

ایجنت‌های موجود در سیستم:
{agents_list}

حافظه بلندمدت (واقعیت‌های شناخته‌شده درباره کاربر):
{facts_text}
"""


def run_turn(session_id: str, user_message: str) -> str:
    db.add_message(session_id, "user", user_message)
    history = db.get_recent_messages(session_id, limit=20)

    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if m["role"] in ("user", "assistant")
    ]

    system_prompt = _build_system_prompt()
    tools = _tools_schema()

    for _ in range(6):
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=4000,
            tools=tools,
            messages=[{"role": "system", "content": system_prompt}] + messages,
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            final_text = msg.content or ""
            db.add_message(session_id, "assistant", final_text)
            return final_text

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })

        for tc in msg.tool_calls:
            tool_input = json.loads(tc.function.arguments)
            result = _execute_tool(tc.function.name, tool_input)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
            if tc.function.name in ("create_agent", "update_agent", "delete_agent"):
                system_prompt = _build_system_prompt()

    fallback = "متاسفانه پردازش این درخواست پیچیده‌تر از حد مجاز شد. لطفاً ساده‌ترش کن."
    db.add_message(session_id, "assistant", fallback)
    return fallback
