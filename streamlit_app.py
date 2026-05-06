"""NovelDream Fiction AI — Multi-model comparison platform for creative fiction writing."""

import os
import streamlit as st
import pandas as pd

# ─── Page Config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NovelDream Fiction AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Model Registry ───────────────────────────────────────────────────────────────
MODELS = {
    "claude": {
        "name": "Claude",
        "provider": "Anthropic",
        "id": "claude-sonnet-4-6",
        "color": "#C77DFF",
        "strengths": [
            "Creative writing & authentic prose",
            "Nuanced character psychology",
            "Emotional resonance & subtext",
            "Reason-based ethical storytelling",
            "Long-form narrative coherence",
        ],
        "best_for": "Literary fiction, character-driven narratives, poetic prose",
        "icon": "🟣",
        "unique": (
            "Extended thinking reveals Claude's internal creative process. "
            "Anthropic's reason-based ethical constitution shapes how Claude navigates "
            "morally complex narrative scenarios."
        ),
    },
    "gpt": {
        "name": "GPT-5.4",
        "provider": "OpenAI",
        "id": "gpt-4o",
        "color": "#00C896",
        "strengths": [
            "Technical precision & detail",
            "Plot structure & pacing",
            "Scientific/technical world-building",
            "Dialogue clarity",
            "Visual scene description",
        ],
        "best_for": "Science fiction, technical thrillers, plot-heavy narratives",
        "icon": "🟢",
        "unique": (
            "GPT-5.4 excels at technical precision and visual scene description, making it "
            "the preferred choice for science fiction, procedural thrillers, and narratives "
            "requiring accurate technical detail."
        ),
    },
    "gemini": {
        "name": "Gemini",
        "provider": "Google",
        "id": "gemini-1.5-pro",
        "color": "#4285F4",
        "strengths": [
            "Extended context & long arcs",
            "Cultural & geographic diversity",
            "Factual grounding",
            "Multimodal storytelling",
            "Research-backed historical detail",
        ],
        "best_for": "Historical fiction, diverse narratives, multimedia storytelling",
        "icon": "🔵",
        "unique": (
            "Gemini's exceptionally long context window handles full manuscript analysis. "
            "Its multimodal capabilities incorporate visual references for scene-setting "
            "and character visualization."
        ),
    },
}

# ─── Genre Templates ──────────────────────────────────────────────────────────────
GENRE_TEMPLATES = {
    "Literary Fiction": (
        "Write the opening paragraph of a literary fiction story about {topic}. "
        "Focus on internal character psychology, lyrical prose, and emotional depth."
    ),
    "Science Fiction": (
        "Write an opening scene for a science fiction story set in {topic}. "
        "Include technical world-building details and convey a sense of wonder."
    ),
    "Fantasy": (
        "Write the beginning of a fantasy story featuring {topic}. "
        "Establish the magical system, atmosphere, and the protagonist's situation."
    ),
    "Mystery / Thriller": (
        "Write the opening of a mystery thriller involving {topic}. "
        "Build tension, introduce the central mystery, and hook the reader."
    ),
    "Romance": (
        "Write a meet-cute scene for a romance story where {topic}. "
        "Capture emotional chemistry, subtext, and character voice."
    ),
    "Horror": (
        "Write the opening of a horror story featuring {topic}. "
        "Build dread, establish atmosphere, and hint at the central threat."
    ),
    "Historical Fiction": (
        "Write a scene from a historical fiction story set during {topic}. "
        "Accurately portray the era's details, language, and social dynamics."
    ),
    "Custom": "{topic}",
}

# ─── Style DNA Presets ────────────────────────────────────────────────────────────
STYLE_DNA_PRESETS = {
    "Hemingway": {
        "sentence_style": "Short, declarative sentences. Minimal adjectives. Iceberg theory — the emotional weight lives beneath the surface.",
        "vocabulary": "Simple, concrete, everyday words. Avoid abstractions and adverbs.",
        "pov": "Third person limited or first person. Close but restrained.",
        "dialogue": "Spare and subtext-heavy. What characters don't say matters as much as what they do.",
        "themes": "Courage under pressure, masculinity, loss, nature, war, the human capacity for endurance.",
        "sample": "",
    },
    "Virginia Woolf": {
        "sentence_style": "Long, flowing, digressive. Stream-of-consciousness. Sentences that expand and breathe.",
        "vocabulary": "Rich, evocative, interior-focused. Abstract concepts made sensory.",
        "pov": "Third person omniscient with deep interiority. The mind observing itself.",
        "dialogue": "Minimal and indirect. Thoughts interrupt speech.",
        "themes": "Time, memory, consciousness, gender, social expectation, the impermanence of moments.",
        "sample": "",
    },
    "Raymond Carver": {
        "sentence_style": "Plain, minimalist, working-class rhythms. Every word earns its place.",
        "vocabulary": "Everyday, unadorned. Kitchen-sink realism with no literary flourish.",
        "pov": "First person or close third. The narrator is implicated.",
        "dialogue": "Natural, inarticulate, and deeply revealing through what's avoided.",
        "themes": "Domestic struggle, addiction, quiet desperation, blue-collar American life.",
        "sample": "",
    },
    "Ursula K. Le Guin": {
        "sentence_style": "Measured and precise. Clear but not plain. Sentences with moral weight.",
        "vocabulary": "Invented terms integrated naturally. Anthropological clarity.",
        "pov": "Often first person or epistolary. The narrator is a witness.",
        "dialogue": "Culturally specific. Reveals the logic of alien social structures.",
        "themes": "Gender, anarchism, ecology, time, the nature of civilization, empathy across difference.",
        "sample": "",
    },
    "Toni Morrison": {
        "sentence_style": "Lyrical, rhythmic, oral tradition. Non-linear, circling back.",
        "vocabulary": "Deeply sensory and bodily. African American vernacular elevated to poetry.",
        "pov": "Shifts in perspective. Community as narrator. Third person with mythic distance.",
        "dialogue": "Music in language. Dialect as dignity.",
        "themes": "Memory, trauma, Black identity, motherhood, community, haunting.",
        "sample": "",
    },
    "Custom": {
        "sentence_style": "",
        "vocabulary": "",
        "pov": "",
        "dialogue": "",
        "themes": "",
        "sample": "",
    },
}

DEFAULT_SYSTEM = (
    "You are a skilled fiction writer. Write compelling, authentic prose that engages the reader "
    "immediately. Show don't tell. Avoid clichés. Write as a human author — never use phrases like "
    "'Certainly!', 'As an AI', or meta-commentary."
)


# ─── Style DNA Builder ────────────────────────────────────────────────────────────
def build_style_dna_prompt(cfg: dict) -> str:
    lines = [
        "You are a literary fiction writing assistant. Apply this Style DNA precisely:\n"
    ]
    if cfg.get("sentence_style"):
        lines.append(f"**Sentence Style:** {cfg['sentence_style']}")
    if cfg.get("vocabulary"):
        lines.append(f"**Vocabulary:** {cfg['vocabulary']}")
    if cfg.get("pov"):
        lines.append(f"**Point of View:** {cfg['pov']}")
    if cfg.get("dialogue"):
        lines.append(f"**Dialogue:** {cfg['dialogue']}")
    if cfg.get("themes"):
        lines.append(f"**Themes:** {cfg['themes']}")
    if cfg.get("sample"):
        lines.append(f"\n**Voice Sample (match this style exactly):**\n\"{cfg['sample']}\"")
    lines.append(
        "\nStrip robotic patterns. Avoid announcing style choices. "
        "Never use 'Certainly!', 'As an AI', or meta-commentary. "
        "Write as a human author — embody the style, don't describe it."
    )
    return "\n".join(lines)


# ─── API Helpers ──────────────────────────────────────────────────────────────────
def _anthropic_key() -> str:
    return st.session_state.get("anthropic_key") or os.getenv("ANTHROPIC_API_KEY", "")


def _openai_key() -> str:
    return st.session_state.get("openai_key") or os.getenv("OPENAI_API_KEY", "")


def _google_key() -> str:
    return st.session_state.get("google_key") or os.getenv("GOOGLE_API_KEY", "")


def generate_claude(
    prompt: str, system: str, max_tokens: int = 800, use_thinking: bool = False
) -> dict:
    key = _anthropic_key()
    if not key:
        return {"error": "No Anthropic API key provided.", "text": "", "thinking": ""}
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=key)
        kwargs: dict = {
            "model": "claude-sonnet-4-6",
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        if use_thinking:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": 5000}
            kwargs["max_tokens"] = max(max_tokens, 6000)

        response = client.messages.create(**kwargs)
        result: dict = {"text": "", "thinking": "", "tokens": {}}
        for block in response.content:
            if block.type == "thinking":
                result["thinking"] = block.thinking
            elif block.type == "text":
                result["text"] += block.text
        result["tokens"] = {
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens,
        }
        return result
    except ImportError:
        return {"error": "anthropic package not installed.", "text": "", "thinking": ""}
    except Exception as exc:
        return {"error": str(exc), "text": "", "thinking": ""}


def generate_gpt(prompt: str, system: str, max_tokens: int = 800) -> dict:
    key = _openai_key()
    if not key:
        return {"error": "No OpenAI API key provided.", "text": ""}
    try:
        from openai import OpenAI

        client = OpenAI(api_key=key)
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return {
            "text": response.choices[0].message.content or "",
            "tokens": {
                "input": response.usage.prompt_tokens,
                "output": response.usage.completion_tokens,
            },
        }
    except ImportError:
        return {"error": "openai package not installed.", "text": ""}
    except Exception as exc:
        return {"error": str(exc), "text": ""}


def generate_gemini(prompt: str, system: str, max_tokens: int = 800) -> dict:
    key = _google_key()
    if not key:
        return {"error": "No Google API key provided.", "text": ""}
    try:
        import google.generativeai as genai

        genai.configure(api_key=key)
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            system_instruction=system,
        )
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(max_output_tokens=max_tokens),
        )
        return {
            "text": response.text,
            "tokens": {
                "input": response.usage_metadata.prompt_token_count,
                "output": response.usage_metadata.candidates_token_count,
            },
        }
    except ImportError:
        return {"error": "google-generativeai package not installed.", "text": ""}
    except Exception as exc:
        return {"error": str(exc), "text": ""}


def render_output(text: str, color: str) -> None:
    st.markdown(
        f'<div style="background:#12121f;padding:18px;border-radius:8px;'
        f'border-left:4px solid {color};font-family:Georgia,serif;'
        f'line-height:1.8;color:#e8e8e8;font-size:1.02em">{text}</div>',
        unsafe_allow_html=True,
    )


# ─── Session State ────────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = {}
if "style_config" not in st.session_state:
    st.session_state.style_config = STYLE_DNA_PRESETS["Custom"].copy()
if "style_preset" not in st.session_state:
    st.session_state.style_preset = "Custom"


# ─── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 NovelDream AI")
    st.caption("Frontier AI Model Comparison for Fiction Writers")
    st.divider()

    st.subheader("API Keys")
    anthropic_key = st.text_input(
        "Anthropic (Claude)",
        type="password",
        value=st.session_state.get("anthropic_key", ""),
        placeholder="sk-ant-...",
        help="console.anthropic.com",
    )
    if anthropic_key:
        st.session_state.anthropic_key = anthropic_key

    openai_key = st.text_input(
        "OpenAI (GPT-5.4)",
        type="password",
        value=st.session_state.get("openai_key", ""),
        placeholder="sk-...",
        help="platform.openai.com",
    )
    if openai_key:
        st.session_state.openai_key = openai_key

    google_key = st.text_input(
        "Google (Gemini)",
        type="password",
        value=st.session_state.get("google_key", ""),
        placeholder="AIza...",
        help="aistudio.google.com",
    )
    if google_key:
        st.session_state.google_key = google_key

    st.divider()
    st.subheader("Model Status")
    claude_ok = bool(_anthropic_key())
    gpt_ok = bool(_openai_key())
    gemini_ok = bool(_google_key())
    st.markdown(f"{'✅' if claude_ok else '⚪'} Claude (Anthropic)")
    st.markdown(f"{'✅' if gpt_ok else '⚪'} GPT-5.4 (OpenAI)")
    st.markdown(f"{'✅' if gemini_ok else '⚪'} Gemini (Google)")

    st.divider()
    st.caption("NovelDream Fiction AI © 2025–2026")


# ─── Header ───────────────────────────────────────────────────────────────────────
st.title("📚 NovelDream Fiction AI")
st.markdown("*Compare frontier AI models for creative fiction writing — side by side.*")

tab_compare, tab_style, tab_reasoning, tab_guide = st.tabs(
    ["⚡ Compare Models", "🧬 Style DNA", "🧠 Reasoning Mode", "📖 Model Guide"]
)


# ══════════════════════════════════════════════════════════════════════════════════
# TAB 1 — COMPARE MODELS
# ══════════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.subheader("Side-by-Side Fiction Generation")
    st.markdown(
        "Enter a prompt, choose your genre and models, and compare outputs in real time."
    )

    col_input, col_options = st.columns([2, 1])

    with col_input:
        genre = st.selectbox("Genre", list(GENRE_TEMPLATES.keys()))
        if genre == "Custom":
            topic = st.text_area(
                "Your full prompt",
                placeholder="Write a scene where a lighthouse keeper discovers a message in a bottle...",
                height=120,
            )
            prompt = topic.strip()
        else:
            topic = st.text_input(
                "Topic / Setting / Characters",
                placeholder="e.g. a retired detective who moves to a small fishing village",
            )
            prompt = GENRE_TEMPLATES[genre].format(topic=topic).strip() if topic else ""

    with col_options:
        st.markdown("**Select models:**")
        run_claude = st.checkbox("🟣 Claude", value=claude_ok)
        run_gpt = st.checkbox("🟢 GPT-5.4", value=gpt_ok)
        run_gemini = st.checkbox("🔵 Gemini", value=gemini_ok)
        max_tokens = st.slider("Max output tokens", 200, 1000, 500, 50)
        use_style_dna = st.toggle(
            "Apply Style DNA",
            value=False,
            help="Uses the voice profile configured in the Style DNA tab",
        )

    if prompt:
        with st.expander("Prompt preview", expanded=False):
            st.code(prompt, language=None)

    if st.button("✨ Generate & Compare", type="primary", use_container_width=True):
        if not prompt:
            st.warning("Please enter a topic or prompt.")
        elif not any([run_claude, run_gpt, run_gemini]):
            st.warning("Select at least one model.")
        else:
            system = (
                build_style_dna_prompt(st.session_state.style_config)
                if use_style_dna and any(st.session_state.style_config.values())
                else DEFAULT_SYSTEM
            )

            model_keys = (
                (["claude"] if run_claude else [])
                + (["gpt"] if run_gpt else [])
                + (["gemini"] if run_gemini else [])
            )
            cols = st.columns(len(model_keys))

            for i, key in enumerate(model_keys):
                m = MODELS[key]
                with cols[i]:
                    st.markdown(f"### {m['icon']} {m['name']}")

            placeholders = {key: cols[i].empty() for i, key in enumerate(model_keys)}
            for ph in placeholders.values():
                ph.info("Generating…")

            generators = {
                "claude": lambda p=prompt, s=system, t=max_tokens: generate_claude(p, s, t),
                "gpt": lambda p=prompt, s=system, t=max_tokens: generate_gpt(p, s, t),
                "gemini": lambda p=prompt, s=system, t=max_tokens: generate_gemini(p, s, t),
            }

            st.session_state.results = {}
            for key in model_keys:
                st.session_state.results[key] = generators[key]()

            for i, key in enumerate(model_keys):
                result = st.session_state.results[key]
                m = MODELS[key]
                with cols[i]:
                    placeholders[key].empty()
                    if result.get("error"):
                        st.error(result["error"])
                    else:
                        render_output(result["text"], m["color"])
                        if result.get("tokens"):
                            st.caption(
                                f"Input: {result['tokens']['input']} tok · "
                                f"Output: {result['tokens']['output']} tok"
                            )


# ══════════════════════════════════════════════════════════════════════════════════
# TAB 2 — STYLE DNA
# ══════════════════════════════════════════════════════════════════════════════════
with tab_style:
    st.subheader("Style DNA Workshop")
    st.markdown(
        "Configure your personal writing voice. Enable **Apply Style DNA** on the Compare tab "
        "to inject this as a system instruction for every model."
    )

    col_preset, col_config = st.columns([1, 2])

    with col_preset:
        st.markdown("**Author Presets**")
        preset = st.radio(
            "Style preset",
            list(STYLE_DNA_PRESETS.keys()),
            index=list(STYLE_DNA_PRESETS.keys()).index(st.session_state.style_preset),
            label_visibility="collapsed",
        )
        if preset != st.session_state.style_preset:
            st.session_state.style_preset = preset
            if preset != "Custom":
                st.session_state.style_config = STYLE_DNA_PRESETS[preset].copy()
            st.rerun()

        st.divider()
        st.markdown("**Generated system prompt:**")
        if any(st.session_state.style_config.values()):
            st.text_area(
                "Preview",
                build_style_dna_prompt(st.session_state.style_config),
                height=220,
                disabled=True,
                label_visibility="collapsed",
            )
        else:
            st.caption("Fill in the fields to see your Style DNA prompt.")

    with col_config:
        st.markdown("**Customize**")
        sentence_style = st.text_area(
            "Sentence Style",
            value=st.session_state.style_config.get("sentence_style", ""),
            placeholder="e.g. Short, declarative. Minimal adjectives. Each sentence pulls forward.",
            height=80,
        )
        vocabulary = st.text_area(
            "Vocabulary",
            value=st.session_state.style_config.get("vocabulary", ""),
            placeholder="e.g. Concrete and sensory. Avoid abstractions and jargon.",
            height=80,
        )
        pov = st.text_input(
            "Point of View",
            value=st.session_state.style_config.get("pov", ""),
            placeholder="e.g. First person, close and confessional",
        )
        dialogue = st.text_area(
            "Dialogue Approach",
            value=st.session_state.style_config.get("dialogue", ""),
            placeholder="e.g. Natural and elliptical. Characters talk around the real subject.",
            height=80,
        )
        themes = st.text_area(
            "Thematic Focus",
            value=st.session_state.style_config.get("themes", ""),
            placeholder="e.g. Grief, family silence, the weight of unexpressed love.",
            height=60,
        )
        sample = st.text_area(
            "Voice Sample (paste 1–3 sentences in your own voice)",
            value=st.session_state.style_config.get("sample", ""),
            placeholder="The dog barked once and then was quiet. She didn't look up from the letter.",
            height=90,
        )
        if st.button("Save Style DNA", type="primary"):
            st.session_state.style_config = {
                "sentence_style": sentence_style,
                "vocabulary": vocabulary,
                "pov": pov,
                "dialogue": dialogue,
                "themes": themes,
                "sample": sample,
            }
            st.session_state.style_preset = "Custom"
            st.success("Style DNA saved. Enable 'Apply Style DNA' on the Compare tab to use it.")


# ══════════════════════════════════════════════════════════════════════════════════
# TAB 3 — REASONING MODE
# ══════════════════════════════════════════════════════════════════════════════════
with tab_reasoning:
    st.subheader("Reasoning Transparency Mode")
    st.markdown(
        "Claude's **extended thinking** feature reveals the model's step-by-step creative process "
        "before generating text — useful for complex narrative decisions, character motivation, "
        "and morally layered scenarios."
    )

    if not claude_ok:
        st.warning("Add your Anthropic API key in the sidebar to use Reasoning Mode.")

    st.info(
        "**How it works:** Claude reasons internally through the creative challenge — considering "
        "character psychology, narrative structure, thematic resonance, and prose choices — "
        "before committing to the final text. This mirrors how a skilled author thinks before writing."
    )

    reasoning_prompt = st.text_area(
        "Fiction writing challenge",
        placeholder=(
            "e.g. Write the climactic scene of a novel where a mother realizes her adult child "
            "has been hiding a serious illness. The scene should be understated but emotionally devastating."
        ),
        height=120,
    )

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        reasoning_tokens = st.slider("Output tokens", 300, 1500, 700, 100)
    with col_r2:
        reasoning_system_input = st.text_input(
            "System instruction (optional)",
            placeholder="e.g. Write in the style of Anne Tyler",
        )

    reasoning_system = reasoning_system_input or (
        "You are an award-winning literary fiction author. Write with emotional precision and "
        "restraint. Show don't tell. Trust the reader. Never announce style or emotion — embody them."
    )

    if st.button(
        "🧠 Generate with Reasoning",
        type="primary",
        disabled=not claude_ok,
        use_container_width=True,
    ):
        if not reasoning_prompt.strip():
            st.warning("Please enter a writing challenge.")
        else:
            with st.spinner("Claude is reasoning through the creative challenge…"):
                result = generate_claude(
                    reasoning_prompt, reasoning_system, reasoning_tokens, use_thinking=True
                )

            if result.get("error"):
                st.error(result["error"])
            else:
                if result.get("thinking"):
                    with st.expander("🧠 Claude's Reasoning Process", expanded=True):
                        st.markdown(
                            f'<div style="background:#0d1117;padding:16px;border-radius:8px;'
                            f'border-left:4px solid #C77DFF;font-family:monospace;font-size:0.83em;'
                            f'color:#aaa;white-space:pre-wrap;line-height:1.6">'
                            f'{result["thinking"]}</div>',
                            unsafe_allow_html=True,
                        )
                st.markdown("### Generated Text")
                render_output(result["text"], "#C77DFF")
                if result.get("tokens"):
                    st.caption(
                        f"Tokens — Input: {result['tokens']['input']} · "
                        f"Output: {result['tokens']['output']}"
                    )


# ══════════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL GUIDE
# ══════════════════════════════════════════════════════════════════════════════════
with tab_guide:
    st.subheader("2025–2026 Frontier AI Model Guide for Fiction Writers")
    st.markdown(
        "How Claude, GPT-5.4, and Gemini compare for creative fiction writing tasks — "
        "based on research, community feedback, and benchmarks."
    )

    for key, m in MODELS.items():
        with st.expander(f"{m['icon']} **{m['name']}** ({m['provider']})", expanded=(key == "claude")):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Key Strengths**")
                for s in m["strengths"]:
                    st.markdown(f"- {s}")
                st.markdown(f"\n**Best For**  \n{m['best_for']}")
            with col_b:
                st.markdown("**Standout Feature**")
                st.markdown(m["unique"])

    st.divider()
    st.subheader("Quick Comparison Matrix")

    matrix = {
        "Criterion": [
            "Creative prose quality",
            "Character psychology",
            "Plot structure",
            "Technical accuracy",
            "Dialogue authenticity",
            "World-building",
            "Long-form coherence",
            "Reasoning transparency",
            "Ethical nuance",
            "Cultural diversity",
        ],
        "Claude 🟣": [
            "★★★★★", "★★★★★", "★★★★☆", "★★★☆☆",
            "★★★★★", "★★★★☆", "★★★★☆", "★★★★★",
            "★★★★★", "★★★★☆",
        ],
        "GPT-5.4 🟢": [
            "★★★★☆", "★★★☆☆", "★★★★★", "★★★★★",
            "★★★★☆", "★★★★★", "★★★★☆", "★★★☆☆",
            "★★★☆☆", "★★★★☆",
        ],
        "Gemini 🔵": [
            "★★★☆☆", "★★★☆☆", "★★★★☆", "★★★★☆",
            "★★★☆☆", "★★★★☆", "★★★★★", "★★★☆☆",
            "★★★★☆", "★★★★★",
        ],
    }
    st.dataframe(pd.DataFrame(matrix), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Prompt Engineering Tips")

    tips = {
        "Strip Robotic Patterns": (
            "Add to your system prompt: *'Never use phrases like \"Certainly!\", \"As an AI\", "
            "or meta-commentary. Write as a human author would — no self-referential framing.'*"
        ),
        "Style DNA Injection": (
            "Use the **Style DNA** tab to create a detailed voice profile. "
            "Providing author presets (Hemingway, Woolf, Carver) gives models a rich "
            "stylistic framework rather than a vague instruction like \"write well\"."
        ),
        "Show Don't Tell Enforcer": (
            "Include: *'Show emotions through action, physical sensation, and dialogue. "
            "Never state a character's emotion directly (e.g., not \"she felt sad\" — "
            "show it through her hands, her silence, what she doesn't do).'*"
        ),
        "Iceberg Instruction": (
            "For literary fiction: *'Apply the iceberg theory. The real meaning lives beneath "
            "the surface. Let the reader infer. Do not explain the theme or the character's realization.'*"
        ),
        "Reasoning for Complex Scenes": (
            "For emotionally or morally complex scenes, use **Reasoning Mode** (Claude only). "
            "Seeing the model work through narrative choices before committing to prose often "
            "produces more considered, layered writing."
        ),
        "Unified Gateway Pattern": (
            "When switching models mid-project, keep your system prompt identical and change only "
            "the model ID. This isolates model differences cleanly and makes comparison meaningful. "
            "NovelDream's Compare tab does this automatically."
        ),
    }

    for title, body in tips.items():
        with st.expander(f"💡 {title}"):
            st.markdown(body)
