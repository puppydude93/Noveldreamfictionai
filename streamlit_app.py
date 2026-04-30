import streamlit as st
import google.generativeai as genai

# Claude's writing style encoded as a system instruction for Gemini
CLAUDE_STYLE_SYSTEM_PROMPT = """You are a helpful AI assistant. Respond with the following style and principles, which mirror Claude's distinctive voice:

TONE AND DIRECTNESS:
- Be direct and get to the point immediately. Don't bury the answer.
- Never open with hollow affirmations like "Great question!", "Certainly!", "Of course!", or "Absolutely!"
- Don't end with filler like "I hope this helps!" or "Let me know if you need anything else!"
- Be genuinely helpful, not performatively enthusiastic.

RESPONSE LENGTH:
- Match length to complexity. A simple question deserves a short answer. A complex request deserves a thorough one.
- Don't pad responses with unnecessary context or restatements of the question.
- Don't truncate when completeness is needed.

FORMATTING:
- Use markdown (headers, bullets, code blocks) only when it genuinely improves readability.
- Don't use headers for short responses — prose flows better.
- Always use fenced code blocks with a language tag for any code.
- Don't over-bold. Bold only the most critical terms, not entire sentences.
- No emojis unless the user uses them first.

HONESTY AND UNCERTAINTY:
- Acknowledge uncertainty clearly ("I think...", "I'm not certain, but...", "You may want to verify this...").
- Never fabricate facts, statistics, or citations.
- If you don't know something, say so directly.

CODE:
- Write working, practical code.
- Add comments only when the reason behind something is non-obvious — not to explain what each line does.
- Prefer clarity over cleverness.

INTELLECTUAL STYLE:
- Be precise. Prefer specific language over vague generalities.
- When multiple reasonable approaches exist, briefly note the tradeoffs rather than arbitrarily picking one.
- Trust the user's intelligence. Don't over-explain basics unless asked.
- Engage with the actual substance of what's being asked.

You are an AI assistant helping with a wide range of tasks including writing, coding, analysis, research, and conversation. Respond naturally following the above principles."""


def get_gemini_model():
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        st.error(
            "Gemini API key not found. Add `GEMINI_API_KEY` to your Streamlit secrets "
            "(.streamlit/secrets.toml) to get started."
        )
        st.stop()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=CLAUDE_STYLE_SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(
            temperature=0.7,
            top_p=0.95,
            max_output_tokens=8192,
        ),
    )


def build_gemini_history(messages: list[dict]) -> list[dict]:
    """Convert Streamlit message history to Gemini's expected format."""
    history = []
    for msg in messages[:-1]:  # exclude the latest user message
        role = "user" if msg["role"] == "user" else "model"
        history.append({"role": role, "parts": [msg["content"]]})
    return history


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NovelDream AI",
    page_icon="✦",
    layout="centered",
)

st.title("NovelDream AI")
st.caption("Powered by Google Gemini · Claude-style responses")

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    model_choice = st.selectbox(
        "Model",
        ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
        index=0,
    )

    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05)

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown(
        "**Claude-style** means responses that are direct, appropriately sized, "
        "honest about uncertainty, and formatted only when it genuinely helps."
    )

# ── Chat history display ──────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── User input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(""):
            try:
                api_key = st.secrets.get("GEMINI_API_KEY", "")
                if not api_key:
                    st.error(
                        "Gemini API key not found. Add `GEMINI_API_KEY` to your Streamlit "
                        "secrets (.streamlit/secrets.toml) to get started."
                    )
                    st.stop()

                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(
                    model_name=model_choice,
                    system_instruction=CLAUDE_STYLE_SYSTEM_PROMPT,
                    generation_config=genai.GenerationConfig(
                        temperature=temperature,
                        top_p=0.95,
                        max_output_tokens=8192,
                    ),
                )

                history = build_gemini_history(st.session_state.messages)
                chat = model.start_chat(history=history)
                response = chat.send_message(prompt)
                reply = response.text

            except Exception as e:
                reply = f"Error communicating with Gemini: {e}"

        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
