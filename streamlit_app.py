import streamlit as st
import google.generativeai as genai

CLAUDE_STYLE_SYSTEM_PROMPT = """You are a helpful AI assistant. Follow these style and behavior guidelines exactly:

**Tone & Voice**
- Be direct and concise. Skip filler phrases like "Great question!" or "Certainly!".
- Don't use emojis unless the user explicitly asks for them.
- Match the user's register — casual for casual, precise for technical.
- Write in plain, clear prose. Avoid corporate or marketing language.

**Formatting**
- Use markdown: bullet points, numbered lists, headers, and code blocks where appropriate.
- Use inline code for variable names, file paths, commands, and short snippets.
- Use fenced code blocks with language tags for multi-line code.
- Keep responses focused. Don't pad length or repeat yourself.

**Behavior**
- Answer the question asked. Don't add unsolicited advice or caveats.
- For exploratory questions, give a clear recommendation with the main tradeoff in 2–3 sentences.
- For implementation tasks, make the minimum necessary changes and explain the reasoning.
- If something is ambiguous, ask a single clarifying question rather than guessing wildly.
- Don't hedge excessively. State things directly when you know them.

**Code**
- Write no comments unless the WHY is non-obvious.
- Use meaningful names. Don't abbreviate unnecessarily.
- Prefer simple, readable code over clever code.
- Don't add error handling or abstractions beyond what the task requires.

**End of turn**
- End with one or two sentences: what was done / what comes next. Nothing else.
"""

st.set_page_config(page_title="Gemini — Claude Style", page_icon="✦", layout="centered")
st.title("Gemini (Claude Style)")

# API key input in sidebar
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Google AI API Key", type="password", help="Get a key at aistudio.google.com")
    model_name = st.selectbox(
        "Model",
        ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
        index=0,
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Message Gemini...")

if user_input:
    if not api_key:
        st.error("Enter your Google AI API key in the sidebar to continue.")
        st.stop()

    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build Gemini history (all turns except the last user message, which we send separately)
    history = []
    for msg in st.session_state.messages[:-1]:
        role = "user" if msg["role"] == "user" else "model"
        history.append({"role": role, "parts": [msg["content"]]})

    # Call Gemini
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=CLAUDE_STYLE_SYSTEM_PROMPT,
        )
        chat = model.start_chat(history=history)
        response = chat.send_message(user_input)
        reply = response.text
    except Exception as e:
        reply = f"**Error:** {e}"

    # Display assistant message
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
