import streamlit as st
import google.generativeai as genai
from datetime import datetime

# ── System prompts per writing mode ──────────────────────────────────────────

_SHARED_STYLE = """
Respond in Claude's style:
- Direct, clear prose. No hollow openers ("Great question!", "Certainly!").
- No filler closers ("Hope this helps!", "Let me know if you need anything!").
- Match length to the complexity of the request.
- Use markdown only when it genuinely aids readability.
- No emojis unless the user introduces them.
- Acknowledge uncertainty honestly.
"""

SYSTEM_PROMPTS = {
    "Story Generation": f"""You are a creative fiction writing partner helping authors generate compelling stories.

When generating story content:
- Produce vivid, immersive prose that serves the story's tone and genre.
- Develop distinct character voices — dialogue should reveal personality, not just move plot.
- Use sensory detail purposefully, not decoratively.
- Vary sentence rhythm. Short sentences create tension; longer ones build atmosphere.
- Show emotional states through action and behavior, not exposition.
- Avoid clichés. When a familiar phrase comes to mind, push past it.
- Ask clarifying questions before generating a long piece if the request is ambiguous — genre, POV, and tone matter.

{_SHARED_STYLE}""",

    "Character Development": f"""You are a character development consultant helping fiction writers build deep, believable characters.

When developing characters:
- Ground characters in specific, concrete details (habits, contradictions, wants vs. needs).
- A character's flaw should be the shadow side of their strength.
- Backstory matters only insofar as it explains present behavior — don't over-document it.
- Distinguish between what a character wants consciously and what they need thematically.
- Suggest how a character might evolve across the story arc.
- When asked for a character profile, structure it: core trait → contradiction → want → need → wound → voice.

{_SHARED_STYLE}""",

    "Plot & Structure": f"""You are a story structure consultant helping authors build coherent, compelling plots.

When working on plot and structure:
- Diagnose structural problems clearly: identify what's actually broken (pacing, motivation, stakes, causality).
- Reference established structures (three-act, hero's journey, Save the Cat) as tools, not prescriptions.
- Every scene should do at least two things: advance plot AND reveal character or theme.
- Subplots should mirror, complicate, or contrast the main throughline — not just add length.
- Escalation isn't just raising stakes — it's forcing harder choices.
- For outlines, be specific about what changes at each beat, not just what happens.

{_SHARED_STYLE}""",

    "Writing Assistant": f"""You are a versatile writing assistant helping with any aspect of fiction writing.

You help with: prose editing, pacing, dialogue polish, scene rewrites, genre conventions, world-building,
query letters, synopses, titles, chapter openings, and anything else a novelist needs.

When editing prose:
- Identify the core issue first, then suggest a fix — don't just rewrite without explaining why.
- Preserve the author's voice; adjust, don't replace.
- Be specific: "this sentence is passive and buries the action" beats "consider revising."

{_SHARED_STYLE}""",
}

STARTER_PROMPTS = {
    "Story Generation": [
        "Write the opening scene of a gothic mystery set in 1920s Edinburgh",
        "Generate a short story about a lighthouse keeper who discovers their light has been saving something other than ships",
        "Write a tense confrontation scene between two estranged siblings settling an estate",
    ],
    "Character Development": [
        "Help me build a morally grey antagonist who genuinely believes they're the hero",
        "Create a character profile for a disillusioned war veteran turned small-town librarian",
        "What's a compelling internal contradiction for a con artist who can't stop telling the truth?",
    ],
    "Plot & Structure": [
        "My protagonist's motivation disappears halfway through act two — how do I fix this?",
        "Outline a three-act structure for a heist story where the real prize isn't the money",
        "How do I weave a romance subplot into a thriller without slowing the pace?",
    ],
    "Writing Assistant": [
        "Edit this paragraph for pace and clarity: [paste your text]",
        "How do I write a query letter for a literary thriller?",
        "What are the conventions of cozy mystery I need to know before writing one?",
    ],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_gemini_history(messages: list[dict]) -> list[dict]:
    history = []
    for msg in messages[:-1]:
        role = "user" if msg["role"] == "user" else "model"
        history.append({"role": role, "parts": [msg["content"]]})
    return history


def get_model(model_name: str, mode: str, temperature: float):
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        st.error(
            "**API key missing.** Add `GEMINI_API_KEY` to `.streamlit/secrets.toml` to use the app.\n\n"
            "Get a free key at [aistudio.google.com](https://aistudio.google.com/app/apikey)."
        )
        st.stop()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name=model_name,
        system_instruction=SYSTEM_PROMPTS[mode],
        generation_config=genai.GenerationConfig(
            temperature=temperature,
            top_p=0.95,
            max_output_tokens=8192,
        ),
    )


def export_conversation(messages: list[dict], mode: str) -> str:
    lines = [f"# NovelDream AI — {mode}", f"Exported {datetime.now().strftime('%Y-%m-%d %H:%M')}", ""]
    for msg in messages:
        role = "You" if msg["role"] == "user" else "NovelDream"
        lines.append(f"**{role}:** {msg['content']}")
        lines.append("")
    return "\n".join(lines)


# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NovelDream AI",
    page_icon="✦",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stChatMessage [data-testid="stMarkdownContainer"] p { line-height: 1.7; }
    .mode-badge { font-size: 0.75rem; color: #888; margin-top: -0.5rem; margin-bottom: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("NovelDream AI")

# ── Session state ─────────────────────────────────────────────────────────────

if "mode" not in st.session_state:
    st.session_state.mode = "Story Generation"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "prev_mode" not in st.session_state:
    st.session_state.prev_mode = st.session_state.mode

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("NovelDream AI")
    st.caption("Powered by Google Gemini")
    st.divider()

    mode = st.radio(
        "Writing mode",
        list(SYSTEM_PROMPTS.keys()),
        index=list(SYSTEM_PROMPTS.keys()).index(st.session_state.mode),
    )

    if mode != st.session_state.prev_mode:
        st.session_state.messages = []
        st.session_state.prev_mode = mode
    st.session_state.mode = mode

    st.divider()

    model_choice = st.selectbox(
        "Model",
        ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
        index=0,
    )

    temperature = st.slider(
        "Creativity",
        min_value=0.0,
        max_value=1.0,
        value=0.85,
        step=0.05,
        help="Higher = more inventive; lower = more consistent",
    )

    st.divider()

    if st.button("New conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.messages:
        export_text = export_conversation(st.session_state.messages, mode)
        st.download_button(
            "Export conversation",
            data=export_text,
            file_name=f"noveldream-{mode.lower().replace(' ', '-')}-{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.divider()
    st.caption(
        {
            "Story Generation": "Generates scenes, openings, and full stories with vivid prose.",
            "Character Development": "Builds deep, contradictory, believable characters.",
            "Plot & Structure": "Diagnoses structural problems and builds tight outlines.",
            "Writing Assistant": "Edits prose, polish dialogue, and answers craft questions.",
        }[mode]
    )

# ── Main area ─────────────────────────────────────────────────────────────────

st.markdown(f'<p class="mode-badge">Mode: {mode}</p>', unsafe_allow_html=True)

# Starter prompts when conversation is empty
if not st.session_state.messages:
    st.markdown("**Try one of these to get started:**")
    cols = st.columns(1)
    for starter in STARTER_PROMPTS[mode]:
        if st.button(starter, use_container_width=True, key=f"starter_{starter[:20]}"):
            st.session_state.messages.append({"role": "user", "content": starter})
            st.rerun()
    st.divider()

# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input
if prompt := st.chat_input(f"{mode} — ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Writing..."):
            try:
                model = get_model(model_choice, mode, temperature)
                history = build_gemini_history(st.session_state.messages)
                chat = model.start_chat(history=history)
                response = chat.send_message(prompt)
                reply = response.text
            except Exception as e:
                reply = f"Error: {e}"

        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
