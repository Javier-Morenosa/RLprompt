"""
Human-Watch: FastAPI chat server with RAG panel and feedback UI.
Powered by Groq (Llama 3.1 8B).

Run: python -m uvicorn demos.human_watch.server:app --reload --port 8000
Or: rlprompt-serve  (entry point)
Requires: pip install -e ".[human-watch]"
Env:      GROQ_API_KEY (o .env en la raiz del proyecto)
"""

import json
import os
import re
import sys
from pathlib import Path

from demos.human_watch.common import DATA_DIR, PROJECT_ROOT

# Cargar .env desde la raiz del proyecto
def _load_dotenv():
    p = PROJECT_ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s and not s.startswith("#") and "=" in s:
                k, v = s.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v

_load_dotenv()

# Asegurar que prompt_rl es importable (src o instalado)
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

_SERVER_VERBOSE = os.environ.get("SERVER_VERBOSE", "").lower() in ("1", "true", "yes")

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from openai import AsyncOpenAI

from prompt_rl.core.policy_schema import build_actor_system_text, parse_policy

# ── Groq config ─────────────────────────────────────────────────────────────────
_GROQ_BASE = "https://api.groq.com/openai/v1"
_GROQ_MODEL = "llama-3.1-8b-instant"
_GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

if not _GROQ_API_KEY:
    print("[Server] AVISO: GROQ_API_KEY no definida. Define la variable o crea .env")

_llm = AsyncOpenAI(base_url=_GROQ_BASE, api_key=_GROQ_API_KEY or "not-set")

app = FastAPI()

# Archivos de estado en data/
_HISTORY_FILE = DATA_DIR / "reward_history.json"
_POPULATION_FILE = DATA_DIR / "population.json"
_INTERACTIONS_FILE = DATA_DIR / "interactions.md"
_SYSTEM_PROMPT_FILE = DATA_DIR / "system_prompt.md"
_SYSTEM_PROMPT_LEGACY = DATA_DIR / "system_prompt.txt"


def _load_system_prompt() -> str:
    try:
        content = _SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        content = ""
    if not content:
        from prompt_rl.core.policy_schema import MINIMAL_POLICY_JSON
        return MINIMAL_POLICY_JSON.strip()
    if _SERVER_VERBOSE:
        mtime = _SYSTEM_PROMPT_FILE.stat().st_mtime if _SYSTEM_PROMPT_FILE.exists() else 0
        preview = (content[:80] + "...") if len(content) > 80 else content
        short_hash = hash(content) & 0xFFFF
        print(f"[Actor] Hot-reload: system_prompt.md | {len(content)} chars | "
              f"mtime={mtime:.1f} | hash={short_hash} | preview={repr(preview)[:60]}...")
    return content

RAG_INFO = """\
## Información de la Empresa

### Planes y Precios
- Plan Básico: 29 €/mes — 5 usuarios, 10 GB almacenamiento
  * El precio INCLUYE IVA del 21 %
- Plan Pro: 79 €/mes — 25 usuarios, 100 GB, soporte prioritario
  * El precio INCLUYE IVA del 21 %

### Política de Devoluciones
- 30 días de garantía de satisfacción
- Devolución completa sin condiciones

### Horario de Atención
- Lunes a Viernes: 9:00 – 18:00
- Soporte técnico 24/7 únicamente para Plan Pro
"""

HTML_PAGE = """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Business Chatbot – Human Watch</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: Arial, sans-serif; display: flex; height: 100vh; background: #f0f2f5; }

    /* ── Chat panel ── */
    #chat-panel {
      flex: 1; display: flex; flex-direction: column;
      padding: 20px; max-width: 620px;
    }
    #chat-panel h2 { margin-bottom: 12px; color: #333; }
    #chat-window {
      flex: 1; background: #fff; border-radius: 10px; border: 1px solid #ddd;
      padding: 14px; overflow-y: auto; min-height: 380px; max-height: 380px;
    }
    .message { margin-bottom: 10px; }
    .message.user { text-align: right; }
    .bubble {
      display: inline-block; padding: 10px 14px; border-radius: 18px;
      max-width: 80%; word-wrap: break-word;
    }
    .user .bubble { background: #0084ff; color: #fff; }
    .bot  .bubble { background: #e4e6eb; color: #333; }

    #input-area { display: flex; gap: 8px; margin-top: 10px; }
    #user-input {
      flex: 1; padding: 10px 14px; border-radius: 20px;
      border: 1px solid #ccc; font-size: 14px;
    }
    #send-btn {
      padding: 10px 22px; background: #0084ff; color: #fff;
      border: none; border-radius: 20px; cursor: pointer; font-size: 14px;
    }
    #send-btn:hover { background: #006edc; }

    /* ── Feedback area ── */
    #feedback-area {
      margin-top: 14px; padding: 12px; background: #fff;
      border-radius: 10px; border: 1px solid #ddd;
    }
    #feedback-area h3 { margin-bottom: 8px; color: #555; font-size: 14px; }
    .feedback-btns { display: flex; gap: 10px; margin-bottom: 8px; }
    .fb-btn {
      padding: 8px 22px; border: none; border-radius: 8px;
      cursor: pointer; font-size: 14px; font-weight: bold;
    }
    #btn-yes { background: #28a745; color: #fff; }
    #btn-no  { background: #dc3545; color: #fff; }
    #btn-yes:hover { background: #1e7e34; }
    #btn-no:hover  { background: #bd2130; }
    #comment-input {
      width: 100%; padding: 8px; border: 1px solid #ccc;
      border-radius: 8px; font-size: 13px; resize: vertical;
    }

    /* ── RAG panel ── */
    #rag-panel {
      width: 340px; padding: 20px; background: #fff8e1;
      border-left: 2px solid #ffc107; overflow-y: auto;
      transition: background 0.4s, border-left-color 0.4s;
    }
    #rag-panel.highlight {
      background: #fff3cd; border-left: 4px solid #ff9800;
    }
    #rag-panel h2 { color: #856404; margin-bottom: 6px; font-size: 16px; }
    #review-rag-btn {
      display: block; width: 100%; margin-bottom: 12px;
      padding: 7px 14px; background: #fff3cd; color: #856404;
      border: 1px solid #ffc107; border-radius: 8px;
      cursor: pointer; font-size: 13px; font-weight: bold; text-align: center;
      transition: background 0.2s;
    }
    #review-rag-btn:hover  { background: #ffe8a1; }
    #review-rag-btn.active { background: #ffc107; color: #fff; }
    #rag-content {
      font-size: 13px; color: #333; line-height: 1.65; user-select: text;
    }
    #rag-content h3 { font-size: 13px; color: #555; margin: 10px 0 4px; }
    #rag-content .rag-bullet { margin: 2px 0 2px 8px; }
    #rag-content .rag-sub    { margin: 1px 0 1px 20px; color: #666; }
  </style>
</head>
<body>

<div id="chat-panel">
  <h2>Chatbot de Negocio</h2>
  <div id="chat-window">
    <div class="message bot">
      <div class="bubble">Hola, soy tu asistente virtual. ¿En qué puedo ayudarte hoy?</div>
    </div>
  </div>

  <div id="input-area">
    <input id="user-input" type="text" placeholder="Escribe tu mensaje..." />
    <button id="send-btn">Enviar</button>
  </div>

  <div id="feedback-area">
    <h3>¿Fue útil la última respuesta?</h3>
    <div class="feedback-btns">
      <button class="fb-btn" id="btn-yes" data-value="yes">Correcto (Sí)</button>
      <button class="fb-btn" id="btn-no"  data-value="no">Incorrecto (No)</button>
    </div>
    <textarea id="comment-input" rows="2"
              placeholder="Si es Incorrecto, indica la corrección aquí (opcional)"></textarea>
  </div>
</div>

<div id="rag-panel">
  <h2>Información de la Empresa (RAG)</h2>
  <button id="review-rag-btn">Revisando documentación</button>
  <div id="rag-content">Cargando...</div>
</div>

<script>
  const SESSION_ID = Math.random().toString(36).substring(2, 10).toUpperCase();
  let lastBotMessage = '';

  function parseRagToHtml(txt) {
    let html = '';
    let inSection = false;
    txt.split('\\n').forEach(line => {
      if (line.startsWith('### ')) {
        if (inSection) html += '</section>';
        const title = line.replace(/^###\\s*/, '');
        html += `<section data-section="${title}"><h3>${title}</h3>`;
        inSection = true;
      } else if (line.startsWith('## ')) {
        if (inSection) { html += '</section>'; inSection = false; }
        html += `<h3>${line.replace(/^##\\s*/, '')}</h3>`;
      } else if (line.startsWith('  * ') || line.startsWith('  * ')) {
        html += `<p class="rag-sub">${line.trim().replace(/^\\*\\s*/, '')}</p>`;
      } else if (line.startsWith('- ')) {
        html += `<p class="rag-bullet">${line.replace(/^-\\s*/, '')}</p>`;
      }
    });
    if (inSection) html += '</section>';
    return html;
  }

  fetch('/rag-info').then(r => r.text()).then(txt => {
    document.getElementById('rag-content').innerHTML = parseRagToHtml(txt);
  });

  async function sendMessage() {
    const input = document.getElementById('user-input');
    const msg = input.value.trim();
    if (!msg) return;

    appendMessage('user', msg);
    input.value = '';

    if (typeof sendEventToPython !== 'undefined') {
      sendEventToPython(JSON.stringify({ type: 'user_query', text: msg }));
    }

    const resp = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg, session_id: SESSION_ID })
    });
    const data = await resp.json();
    lastBotMessage = data.response;
    appendMessage('bot', data.response);

    if (typeof sendEventToPython !== 'undefined') {
      sendEventToPython(JSON.stringify({ type: 'bot_response', text: data.response }));
    }
  }

  function appendMessage(role, text) {
    const win = document.getElementById('chat-window');
    const div = document.createElement('div');
    div.className = 'message ' + role;
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;
    div.appendChild(bubble);
    win.appendChild(div);
    win.scrollTop = win.scrollHeight;
  }

  document.getElementById('send-btn').addEventListener('click', sendMessage);
  document.getElementById('user-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') sendMessage();
  });

  document.querySelectorAll('.fb-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      if (typeof sendEventToPython !== 'undefined') {
        if (btn.dataset.value === 'no') {
          const comment = document.getElementById('comment-input').value.trim();
          sendEventToPython(JSON.stringify({
            type: 'incorrect_with_comment',
            value: 'no',
            label: btn.innerText.trim(),
            lastBotMessage: lastBotMessage,
            text: comment
          }));
          document.getElementById('comment-input').value = '';
        } else {
          sendEventToPython(JSON.stringify({
            type: 'feedback',
            value: btn.dataset.value,
            label: btn.innerText.trim(),
            lastBotMessage: lastBotMessage
          }));
        }
      }
    });
  });

  document.getElementById('review-rag-btn').addEventListener('click', () => {
    const panel = document.getElementById('rag-panel');
    panel.classList.add('highlight');
    setTimeout(() => panel.classList.remove('highlight'), 1200);
    if (typeof sendEventToPython !== 'undefined') {
      sendEventToPython(JSON.stringify({ type: 'review_rag' }));
    }
  });

  const DWELL_MS = 800;
  const DWELL_COOLDOWN_MS = 5000;
  let _dwellTimer = null;
  let _lastDwellKey = '';
  let _lastDwellTs = 0;

  function _zone(el) {
    if (el.closest('#rag-panel'))     return 'RAG_PANEL';
    if (el.closest('#chat-window'))   return 'CHAT_WINDOW';
    if (el.closest('#feedback-area')) return 'FEEDBACK_AREA';
    return null;
  }

  function _section(el, zone) {
    if (zone === 'RAG_PANEL') {
      const sec = el.closest('[data-section]');
      return sec ? sec.dataset.section : 'general';
    }
    if (zone === 'CHAT_WINDOW') {
      const msg = el.closest('.message');
      if (!msg) return 'general';
      return msg.classList.contains('bot') ? 'bot_message' : 'user_message';
    }
    if (zone === 'FEEDBACK_AREA') return 'feedback_controls';
    return 'general';
  }

  function _preview(el, zone) {
    let src;
    if (zone === 'RAG_PANEL') {
      src = el.closest('[data-section]') || el.closest('#rag-content') || el;
    } else if (zone === 'CHAT_WINDOW') {
      src = el.closest('.bubble') || el;
    } else {
      src = el;
    }
    return (src.innerText || src.textContent || '').trim().replace(/\\s+/g, ' ').substring(0, 120);
  }

  let _cursorX = 0, _cursorY = 0, _cursorEl = null;

  document.addEventListener('mousemove', function(e) {
    _cursorX = e.clientX;
    _cursorY = e.clientY;
    _cursorEl = e.target;

    clearTimeout(_dwellTimer);
    const zone = _zone(e.target);
    if (!zone) return;

    _dwellTimer = setTimeout(function() {
      const section = _section(_cursorEl, zone);
      const key = zone + '|' + section;
      const now = Date.now();
      if (key === _lastDwellKey && (now - _lastDwellTs) < DWELL_COOLDOWN_MS) return;
      _lastDwellKey = key;
      _lastDwellTs = now;
      if (typeof sendEventToPython !== 'undefined') {
        sendEventToPython(JSON.stringify({
          type: 'mouse_dwell',
          zone: zone,
          section: section,
          preview: _preview(_cursorEl, zone),
          x: _cursorX,
          y: _cursorY
        }));
      }
    }, DWELL_MS);
  }, { passive: true });

  const CURSOR_SAMPLE_MS = 800;
  setInterval(function() {
    if (!_cursorEl || typeof sendEventToPython === 'undefined') return;
    const zone = _zone(_cursorEl);
    if (!zone) return;
    sendEventToPython(JSON.stringify({
      type: 'cursor_sample',
      zone: zone,
      section: _section(_cursorEl, zone),
      x: _cursorX,
      y: _cursorY,
      element: _cursorEl.tagName,
      label: (_cursorEl.innerText || _cursorEl.id || '').trim().substring(0, 60)
    }));
  }, CURSOR_SAMPLE_MS);
</script>
</body>
</html>
"""

def _build_system_prompt() -> str:
    raw = _load_system_prompt()
    policy, _ = parse_policy(raw)
    actor_text = build_actor_system_text(policy)
    return f"{actor_text}\n\n---\nCATÁLOGO DE LA EMPRESA:\n{RAG_INFO}"


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


@app.get("/system-prompt")
def get_system_prompt():
    return {"system_prompt": _load_system_prompt()}


@app.get("/rag-info", response_class=PlainTextResponse)
def get_rag_info():
    return RAG_INFO


@app.post("/chat")
async def chat(body: dict):
    user_msg = body.get("message", "").strip()
    if not user_msg:
        return {"response": "Por favor, escribe un mensaje."}
    messages = [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user", "content": user_msg},
    ]
    try:
        resp = await _llm.chat.completions.create(
            model=_GROQ_MODEL, messages=messages, max_tokens=256, temperature=0.1
        )
        bot_reply = resp.choices[0].message.content.strip()
        return {"response": bot_reply}
    except Exception as exc:
        return {"response": f"[Error al conectar con Groq: {exc}]"}


@app.post("/clear-history")
async def clear_history(body: dict):
    return {"status": "cleared"}


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    history = []
    version = 0
    if _HISTORY_FILE.exists():
        try:
            hist_data = json.loads(_HISTORY_FILE.read_text(encoding="utf-8"))
            history = hist_data.get("history", [])
            version = hist_data.get("version", 0)
        except Exception:
            pass

    current_prompt = _load_system_prompt()

    pop_size = 0
    pop_gen = 0
    top3_fitness = []
    if _POPULATION_FILE.exists():
        try:
            pop_data = json.loads(_POPULATION_FILE.read_text(encoding="utf-8"))
            individuals = pop_data.get("individuals", [])
            pop_size = len(individuals)
            pop_gen = pop_data.get("generation", 0)
            sorted_inds = sorted(individuals, key=lambda x: x.get("fitness", 0), reverse=True)
            top3_fitness = [round(ind.get("fitness", 0), 4) for ind in sorted_inds[:3]]
        except Exception:
            pass

    corrections = []
    if _INTERACTIONS_FILE.exists():
        try:
            md = _INTERACTIONS_FILE.read_text(encoding="utf-8")
            corrections = [m.group(1) for m in re.finditer(r'\[CORRECCION\]\s*"([^"]*)"', md)][-5:]
        except Exception:
            pass

    total = len(history)
    correct_count = sum(1 for e in history if e.get("verdict") == "CORRECTO")
    accuracy_pct = round(100 * correct_count / total) if total > 0 else 0

    r_vals = [e["R"] for e in history]
    r_max = max((abs(r) for r in r_vals), default=1.0) or 1.0
    bars_html = ""
    for entry in history:
        R = entry["R"]
        pct = round(50 + 50 * R / r_max)
        color = "#28a745" if R >= 0 else "#dc3545"
        ts = entry.get("ts", "")[:16]
        verdict_tag = entry.get("verdict", "")
        bars_html += (
            f'<div style="display:flex;align-items:center;margin-bottom:4px;font-size:12px">'
            f'<span style="width:140px;color:#666">{ts}</span>'
            f'<div style="flex:1;background:#f0f2f5;border-radius:4px;height:18px">'
            f'<div style="width:{pct}%;background:{color};height:100%;border-radius:4px"></div>'
            f'</div>'
            f'<span style="width:70px;text-align:right;color:{color};font-weight:bold">{R:+.4f}</span>'
            f'<span style="width:90px;text-align:right;color:#666;font-size:11px">{verdict_tag}</span>'
            f'</div>'
        )

    top3_parts = []
    for i, f in enumerate(top3_fitness, 1):
        fc = "#28a745" if f >= 0 else "#dc3545"
        top3_parts.append(f'<div style="color:{fc};margin:2px 0">#{i}: {f:+.4f}</div>')
    top3_html = "".join(top3_parts) or '<div style="color:#999">Sin datos</div>'

    corrections_html = "".join(
        f'<li style="margin-bottom:4px">{c}</li>' for c in corrections
    ) or '<li style="color:#999">Sin correcciones registradas</li>'

    prompt_escaped = (current_prompt
                      .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Dashboard — Human Watch</title>
  <style>
    body {{ font-family: Arial, sans-serif; background: #f0f2f5; padding: 24px; margin: 0; }}
    .card {{ background: #fff; border-radius: 10px; padding: 18px; margin-bottom: 18px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
    h1 {{ color: #333; margin-bottom: 20px; }}
    h2 {{ color: #444; font-size: 16px; margin-bottom: 12px; border-bottom: 1px solid #eee; padding-bottom: 6px; }}
    .grid {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 18px; }}
    .stat {{ text-align: center; padding: 12px; }}
    .stat .val {{ font-size: 32px; font-weight: bold; color: #0084ff; }}
    .stat .lbl {{ font-size: 12px; color: #888; margin-top: 4px; }}
    pre {{ background: #f8f9fa; padding: 10px; border-radius: 6px; font-size: 12px; white-space: pre-wrap; word-break: break-word; max-height: 120px; overflow-y: auto; margin: 0; }}
    ul {{ padding-left: 18px; margin: 0; }}
    a {{ color: #0084ff; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>Dashboard — Human-Watch RL</h1>
  <p style="margin-bottom:16px"><a href="/">Volver al Chat</a></p>
  <div class="grid">
    <div class="card stat"><div class="val">{version}</div><div class="lbl">Versión del Prompt</div></div>
    <div class="card stat"><div class="val">{accuracy_pct}%</div><div class="lbl">Accuracy ({correct_count}/{total})</div></div>
    <div class="card stat"><div class="val">{pop_size}</div><div class="lbl">Individuos (gen {pop_gen})</div></div>
  </div>
  <div class="card">
    <h2>System Prompt Activo</h2>
    <pre>{prompt_escaped}</pre>
  </div>
  <div class="card">
    <h2>Historial de Recompensas ({total} entradas)</h2>
    {bars_html if bars_html else '<p style="color:#999;margin:0">Sin historial registrado</p>'}
  </div>
  <div class="card">
    <h2>Top-3 Fitness de Población</h2>
    {top3_html}
  </div>
  <div class="card">
    <h2>Últimas 5 Correcciones</h2>
    <ul>{corrections_html}</ul>
  </div>
</body>
</html>"""
