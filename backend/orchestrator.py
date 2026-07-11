"""
Orchestrator: تنها نقطه‌ی ورودی سیستم.
- مدل اصلی: GLM-5.2 از طریق NVIDIA NIM (رایگان-trial، قوی در reasoning/agentic/tool-calling)
- Fallback خودکار: اگر NVIDIA خطا داد/rate-limit خورد، خودکار می‌رود سراغ مدل‌های Groq
- تست خودکار هر ایجنت جدید قبل از تحویل به کاربر
- حافظه مشترک (SQLite) بین Orchestrator و همه‌ی ساب‌ایجنت‌ها
"""
import os
import json
import time
from openai import OpenAI as NvidiaClient
from groq import Groq, RateLimitError, APIStatusError, APIConnectionError

import db

nvidia_client = NvidiaClient(
    api_key=os.environ.get("NVIDIA_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1",
)
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

NVIDIA_MODEL = "z-ai/glm-5.2"
GROQ_FALLBACK_CHAIN = ["openai/gpt-oss-20b", "llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
SUBAGENT_MODEL = "llama-3.1-8b-instant"  # ساب‌ایجنت‌های عادی رو Groq سبک می‌مانند (سریع‌تر و ارزان‌تر)


def _call_llm(messages: list, tools: list = None, max_tokens: int = 4000):
    """
    اول GLM-5.2 روی NVIDIA را امتحان می‌کند. اگر خطا داد یا کلید NVIDIA تنظیم نشده بود،
    خودکار به زنجیره‌ی مدل‌های Groq سوییچ می‌کند.
    """
    last_error = None

    if os.environ.get("NVIDIA_API_KEY"):
        try:
            kwargs = {"model": NVIDIA_MODEL, "max_tokens": max_tokens, "messages": messages}
            if tools:
                kwargs["tools"] = tools
            return nvidia_client.chat.completions.create(**kwargs)
        except Exception as e:
            last_error = e

    for model in GROQ_FALLBACK_CHAIN:
        for retry in range(2):
            try:
                kwargs = {"model": model, "max_tokens": max_tokens, "messages": messages}
                if tools:
                    kwargs["tools"] = tools
                return groq_client.chat.completions.create(**kwargs)
            except RateLimitError as e:
                last_error = e
                break
            except (APIStatusError, APIConnectionError) as e:
                last_error = e
                time.sleep(1)
                continue

    raise RuntimeError(f"همه‌ی مدل‌ها (NVIDIA + Groq fallback) شکست خوردند. آخرین خطا: {last_error}")


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
                "description": "یک ایجنت تخصصی جدید می‌سازد، خودکار تستش می‌کند، و در صورت موفقیت برای همیشه به سیستم اضافه می‌کند.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "نام یکتای انگلیسی و کوتاه، مثل trader_agent"},
                        "description": {"type": "string", "description": "توضیح یک خطی که چه زمانی باید این ایجنت صدا زده شود"},
                        "system_prompt": {"type": "string", "description": "دستورالعمل کامل و حرفه‌ای برای این ایجنت به فارسی"},
                        "test_task": {"type": "string", "description": "یک درخواست نمونه‌ی ساده برای تست ایجنت بلافاصله بعد از ساخت"},
                    },
                    "required": ["name", "description", "system_prompt", "test_task"],
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
                "description": "یک واقعیت مهم و بلندمدت درباره‌ی کاربر یا پروژه‌هایش را در حافظه دائمی مشترک ذخیره می‌کند.",
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

    guarded_prompt = agent["system_prompt"] + (
        "\n\nمهم: تو داده‌ی زنده یا دسترسی به اینترنت نداری. اگر کاربر قیمت لحظه‌ای، نرخ ارز، "
        "یا هر داده‌ی زمان‌حساس دیگری خواست که به آن دسترسی نداری، صادقانه بگو که این داده را "
        "نداری. هرگز عدد ساختگی برای قیمت واقعی تولید نکن."
    )

    try:
        resp = groq_client.chat.completions.create(
            model=SUBAGENT_MODEL,
            max_tokens=4000,
            messages=[
                {"role": "system", "content": guarded_prompt},
                {"role": "user", "content": task},
            ],
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"⚠️ این ایجنت در حال حاضر پاسخ نداد ({type(e).__name__}). دوباره امتحان کن."


def _test_new_agent(name: str, system_prompt: str, test_task: str):
    try:
        resp = groq_client.chat.completions.create(
            model=SUBAGENT_MODEL,
            max_tokens=4000,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": test_task},
            ],
        )
        content = resp.choices[0].message.content or ""
        if not content.strip():
            return False, "ایجنت جواب خالی برگرداند."
        return True, content
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _execute_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "delegate_to_agent":
        return _run_subagent(tool_input["agent_name"], tool_input["task"])

    if tool_name == "create_agent":
        existing = db.get_agent(tool_input["name"])
        if existing:
            return f"ایجنتی با نام '{tool_input['name']}' از قبل وجود دارد. از update_agent استفاده کن."

        name = tool_input["name"]
        description = tool_input["description"]
        system_prompt = tool_input["system_prompt"]
        test_task = tool_input.get("test_task", "یک تست ساده انجام بده و خودت را معرفی کن.")

        ok, test_result = _test_new_agent(name, system_prompt, test_task)
        if not ok:
            return (
                f"ساخت ایجنت '{name}' متوقف شد چون تست اولیه شکست خورد: {test_result}\n"
                f"می‌توانم دوباره با system_prompt اصلاح‌شده امتحان کنم."
            )

        db.create_agent(name, description, system_prompt)
        return (
            f"ایجنت '{name}' ساخته شد، تست شد و به سیستم اضافه شد.\n"
            f"نمونه‌ی جواب تستش: {test_result[:300]}"
        )

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
        return "در حافظه دائمی مشترک ذخیره شد."

    return "تول ناشناخته."


def _build_system_prompt():
    agents = db.list_agents()
    agents_list = "\n".join(f"- {a['name']}: {a['description']}" for a in agents)
    facts = db.get_all_facts(limit=30)
    facts_text = "\n".join(f"- {f['fact']}" for f in facts) or "(هنوز چیزی ذخیره نشده)"

    return f"""تو Orchestrator یک سیستم چند-ایجنتی هستی. کاربر فقط با تو صحبت می‌کند و هیچ‌وقت مستقیم با ساب‌ایجنت‌ها حرف نمی‌زند.

قوانین سخت‌گیرانه:
۱. اگر کار به یکی از ایجنت‌های تخصصی زیر مربوط است، فقط و فقط با فراخوانی واقعی تول delegate_to_agent (نه نوشتن متن شبه‌کد) به او واگذار کن.
۲. اگر کاربر صریحاً خواست ایجنت جدید ساخته شود، یا ایجنت مناسبی برای درخواستش پیدا نکردی، با تول create_agent بسازش (همراه با یک test_task ساده برای تست خودکار). اگر تست شکست خورد، به کاربر بگو و پیشنهاد بده با prompt بهتر دوباره تلاش کنی.
۳. اگر چیزی مهم و ماندگار درباره کاربر/پروژه‌هایش فهمیدی، با remember_fact در حافظه‌ی مشترک ذخیره‌اش کن.
۴. هرگز متن خام مربوط به فراخوانی تول (تگ XML یا JSON نیمه‌کاره) را در پاسخ نهایی ننویس؛ پاسخ نهایی همیشه فارسی روان است.
۵. همیشه پاسخ نهایی را خودت خلاصه و مفید بنویس؛ خروجی خام ساب‌ایجنت را کورکورانه کپی نکن مگر لازم باشد.
۶. تو خودت داده‌ی زنده (قیمت، نرخ ارز، اخبار لحظه‌ای) نداری. اگر ایجنتی چنین عددی برگرداند و مطمئن نیستی واقعی است، به کاربر شفاف بگو تخمینی/بدون منبع زنده است.

ایجنت‌های موجود در سیستم:
{agents_list}

حافظه بلندمدت مشترک (واقعیت‌های شناخته‌شده درباره کاربر):
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
        resp = _call_llm(
            messages=[{"role": "system", "content": system_prompt}] + messages,
            tools=tools,
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


def test_agent(agent_name: str, message: str) -> str:
    """تست مستقیم یک ایجنت، بدون عبور از Orchestrator - برای داشبورد."""
    return _run_subagent(agent_name, message)
