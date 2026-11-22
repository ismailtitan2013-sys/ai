from pathlib import Path
import re
import brotli
import subprocess

root = Path('.')
source = root / 'index.html.br'
site = root / '_site'
site.mkdir(exist_ok=True)
page = site / 'index.html'
if not source.is_file():
    raise SystemExit('Missing index.html.br')
html = brotli.decompress(source.read_bytes()).decode('utf-8')
original = html

def replace_once(pattern, replacement, label, flags=re.S):
    global html
    updated, count = re.subn(pattern, lambda _m: replacement, html, count=1, flags=flags)
    if count == 0:
        raise SystemExit(f'Could not patch {label}: source pattern not found.')
    html = updated

# Replace the neural TTS worker, ONNX download, and WebAudio pipeline with
# the built-in browser SpeechSynthesis API. No model download or API call.
browser_engine = r"""
let browserVoices = [];
let currentUtterance = null;
let ttsPlaybackToken = 0;
let ttsCurrentSource = null;
let ttsReady = false;
let voicesChangeBound = false;

function browserVoiceId(voice) {
  return voice.voiceURI || (voice.name + '|' + voice.lang);
}

function getBrowserVoice() {
  const select = document.getElementById('voiceSelect');
  const requested = select?.value || state.browserVoiceURI || 'auto';
  if (requested && requested !== 'auto') {
    const chosen = browserVoices.find(voice => browserVoiceId(voice) === requested);
    if (chosen) return chosen;
  }
  return browserVoices.find(voice => /google/i.test(voice.name)) || browserVoices[0] || null;
}

function updateTTSStatus(message, ready = true, progress = 100) {
  const dot = document.getElementById('ttsDot');
  const status = document.getElementById('ttsHeaderStatus');
  const detail = document.getElementById('ttsDetailStatus');
  const bar = document.getElementById('ttsProgressBar');
  if (dot) dot.classList.toggle('ready', !!ready);
  if (status) status.textContent = ready ? 'Голос браузера готов' : message;
  if (detail) detail.textContent = message;
  if (bar && progress !== null) bar.style.width = Math.max(0, Math.min(100, progress)) + '%';
}

function refreshBrowserVoices() {
  if (!('speechSynthesis' in window)) {
    ttsReady = false;
    updateTTSStatus('В этом браузере нет встроенного чтения вслух. Открой сайт в актуальном Google Chrome.', false, 0);
    return;
  }

  const allVoices = window.speechSynthesis.getVoices() || [];
  browserVoices = allVoices
    .filter(voice => /^ru([_-]|$)/i.test(voice.lang || ''))
    .sort((a, b) => {
      const aGoogle = /google/i.test(a.name) ? 0 : 1;
      const bGoogle = /google/i.test(b.name) ? 0 : 1;
      return aGoogle - bGoogle || a.name.localeCompare(b.name, 'ru');
    });

  const select = document.getElementById('voiceSelect');
  if (select) {
    const saved = state.browserVoiceURI || 'auto';
    select.replaceChildren();
    const autoOption = document.createElement('option');
    autoOption.value = 'auto';
    autoOption.textContent = browserVoices.some(v => /google/i.test(v.name))
      ? 'Автоматически · Google Русский'
      : 'Автоматически · русский голос';
    select.appendChild(autoOption);

    browserVoices.forEach(voice => {
      const option = document.createElement('option');
      option.value = browserVoiceId(voice);
      option.textContent = voice.name + ' · ' + voice.lang;
      select.appendChild(option);
    });

    select.value = [...select.options].some(option => option.value === saved) ? saved : 'auto';
    if (select.value !== saved && saved !== 'auto') {
      state.browserVoiceURI = 'auto';
      saveState();
    }
  }

  const selected = getBrowserVoice();
  ttsReady = !!selected;
  if (selected) {
    updateTTSStatus('Готово: ' + selected.name + ' (' + selected.lang + '). Голос предоставляется Chrome/Windows; скачивать модель не нужно.', true, 100);
  } else {
    updateTTSStatus('Chrome пока не предоставил русский голос. Проверь, установлен ли русский голосовой пакет в Windows, затем перезапусти Chrome.', false, 0);
  }
}

function setupVoices() {
  if (!('speechSynthesis' in window)) {
    refreshBrowserVoices();
    return;
  }
  if (!voicesChangeBound) {
    window.speechSynthesis.addEventListener('voiceschanged', refreshBrowserVoices);
    voicesChangeBound = true;
  }
  refreshBrowserVoices();
  // Chrome may populate its voice list asynchronously after DOMContentLoaded.
  window.setTimeout(refreshBrowserVoices, 250);
  window.setTimeout(refreshBrowserVoices, 1000);
}

function changeVoice() {
  const select = document.getElementById('voiceSelect');
  state.browserVoiceURI = select?.value || 'auto';
  saveState();
  refreshBrowserVoices();
}

function loadLocalTTS() {
  // Kept as a harmless compatibility hook for old saved UI state.
  setupVoices();
  refreshBrowserVoices();
}

function optsRate() {
  return Math.max(0.5, Math.min(2, Number(state.rate) || 1));
}
"""
replace_once(
    r"let ttsWorker = null;[\s\S]*?function optsRate\(\) \{ return state\.rate \|\| 1; \}",
    browser_engine,
    "neural TTS engine"
)

browser_speak = r"""
function speakText(text, opts = {}) {
  return new Promise((resolve, reject) => {
    const synthesis = window.speechSynthesis;
    if (!synthesis || typeof window.SpeechSynthesisUtterance !== 'function') {
      reject(new Error('В Chrome недоступно встроенное чтение вслух.'));
      return;
    }

    const token = ttsPlaybackToken;
    const sourceText = opts.noExpand ? String(text || '') : expandForSpeech(String(text || ''));
    const prepared = sourceText.replace(/\s+/g, ' ').trim();
    if (!prepared) {
      resolve();
      return;
    }

    refreshBrowserVoices();
    const selectedVoice = getBrowserVoice();
    if (!selectedVoice) {
      reject(new Error('Не найден русский голос Chrome/Windows. Открой настройки голоса и установи русский голосовой пакет Windows.'));
      return;
    }

    const chunks = typeof splitSpeechChunks === 'function' ? splitSpeechChunks(prepared) : [prepared];
    let index = 0;

    const speakNext = () => {
      if (token !== ttsPlaybackToken) {
        currentUtterance = null;
        resolve();
        return;
      }
      if (index >= chunks.length) {
        currentUtterance = null;
        resolve();
        return;
      }

      const utterance = new SpeechSynthesisUtterance(chunks[index++]);
      utterance.voice = selectedVoice;
      utterance.lang = selectedVoice.lang || 'ru-RU';
      utterance.rate = optsRate();
      utterance.pitch = Math.max(0.7, Math.min(1.35, Number(state.pitch) || 1));
      utterance.volume = 1;

      utterance.onstart = () => {
        if (token === ttsPlaybackToken) {
          updateReadStatus('Воспроизвожу голосом ' + selectedVoice.name + '…');
        }
      };
      utterance.onend = () => {
        if (currentUtterance === utterance) currentUtterance = null;
        speakNext();
      };
      utterance.onerror = event => {
        if (currentUtterance === utterance) currentUtterance = null;
        if (token !== ttsPlaybackToken || event.error === 'canceled' || event.error === 'interrupted') {
          resolve();
          return;
        }
        reject(new Error('Ошибка голоса браузера: ' + (event.error || 'не удалось воспроизвести текст') + '.'));
      };

      currentUtterance = utterance;
      try {
        synthesis.speak(utterance);
      } catch (error) {
        currentUtterance = null;
        reject(error);
      }
    };

    speakNext();
  });
}

function testVoice() {
  const button = document.getElementById('testTTSBtn');
  const text = 'Привет! Это проверка встроенного русского голоса браузера. Никакая отдельная нейросеть для чтения не загружается.';
  if (button) button.blur();
  startReadingText(text);
}
"""
replace_once(
    r"async function speakText\(text, opts = \{\}\) \{[\s\S]*?\n\}\n\nfunction testVoice\(\) \{[\s\S]*?\n\}",
    browser_speak,
    "browser speakText and voice test"
)

# Replace WebAudio pause/resume with the browser's own speech controls.
toggle_play = r"""
function togglePlay() {
  const synthesis = window.speechSynthesis;
  if (state.isPlaying) {
    if (state.isPaused) {
      synthesis.resume();
      state.isPaused = false;
      updatePlaybackButtons();
      updateReadStatus('Чтение продолжается');
    } else {
      synthesis.pause();
      state.isPaused = true;
      updatePlaybackButtons();
      updateReadStatus('Пауза');
    }
  } else {
    startReading();
  }
}
"""
replace_once(
    r"function togglePlay\(\) \{[\s\S]*?\n\}\n\nfunction unlockAudioFromClick\(\)",
    toggle_play + "\nfunction unlockAudioFromClick()",
    "play/pause controls"
)

unlock = r"""function unlockAudioFromClick() {
  if (!('speechSynthesis' in window)) {
    throw new Error('Встроенное чтение вслух недоступно. Открой сайт в актуальном Google Chrome.');
  }
  setupVoices();
  return window.speechSynthesis;
}"""
replace_once(
    r"function unlockAudioFromClick\(\) \{[\s\S]*?\n\}\n\nfunction startReading\(\)",
    unlock + "\n\nfunction startReading()",
    "browser voice activation"
)

stop = r"""function stopReading() {
  state.isPlaying = false;
  state.isPaused = false;
  ttsPlaybackToken++;
  if ('speechSynthesis' in window) {
    try { window.speechSynthesis.cancel(); } catch (_) {}
  }
  currentUtterance = null;
  ttsCurrentSource = null;
  updatePlaybackButtons();
  clearHighlight();
  updateProgress();
  updateReadStatus('Остановлено');
}"""
replace_once(
    r"function stopReading\(\) \{[\s\S]*?\n\}\nfunction updateProgress\(\)",
    stop + "\nfunction updateProgress()",
    "stop reading"
)

# Replace obsolete model-download text with the selected built-in browser voice.
html, status_count = re.subn(
    r"  updateReadStatus\('Подключаю мужской голос…'\);\s*\n"
    r"  updateTTSStatus\('Готовлю мужскую русскую озвучку\. При первом запуске скачается модель около 51 МБ\.', ttsReady, ttsReady \? 100 : 0\);",
    "  updateReadStatus('Подготавливаю встроенный голос Chrome…');\n"
    "  refreshBrowserVoices();\n"
    "  const activeBrowserVoice = getBrowserVoice();\n"
    "  updateTTSStatus(activeBrowserVoice ? ('Используется голос: ' + activeBrowserVoice.name) : 'Русский голос не найден в Chrome. Проверь настройки Windows.', !!activeBrowserVoice, activeBrowserVoice ? 100 : 0);",
    html,
    count=1
)
if status_count == 0 and "updateReadStatus('Подготавливаю встроенный голос Chrome…')" not in html:
    raise SystemExit('Could not update startReading status.')

# Replace the old forced-neural voice list with real voices exposed by Chrome.
voices_setup = r"""
function setupVoices() {
  if (!('speechSynthesis' in window)) {
    refreshBrowserVoices();
    return;
  }
  if (!voicesChangeBound) {
    window.speechSynthesis.addEventListener('voiceschanged', refreshBrowserVoices);
    voicesChangeBound = true;
  }
  refreshBrowserVoices();
  window.setTimeout(refreshBrowserVoices, 250);
  window.setTimeout(refreshBrowserVoices, 1000);
}
function changeVoice() {
  const select = document.getElementById('voiceSelect');
  state.browserVoiceURI = select?.value || 'auto';
  saveState();
  refreshBrowserVoices();
}
"""
replace_once(
    r"/\* Системные голоса Windows/Google не используются\.[\s\S]*?function changeVoice\(\) \{[^}]*\}",
    voices_setup.rstrip(),
    "browser voice selector"
)

# Save the selected system voice across reloads.
html = html.replace(
    "      voiceName: null,\n",
    "      voiceName: null,\n      browserVoiceURI: state.browserVoiceURI || 'auto',\n",
    1
)
html = html.replace(
    "    state._voiceName = null;\n",
    "    state._voiceName = null;\n    state.browserVoiceURI = data.browserVoiceURI || 'auto';\n",
    1
)

# Replace settings UI: no neural model download button, just voice choice and test.
voice_card = r"""<div class="voice-card">
        <div class="voice-card-heading">Голос рассказчика</div>
        <div class="voice-card-name">Встроенный голос Chrome</div>
        <div class="voice-card-note">Используется русский голос, который уже доступен в Chrome или Windows. Отдельную модель скачивать не нужно. Google-голос будет выбран автоматически, если он установлен.</div>
      </div>
      <div class="setting-row">
        <div class="setting-label">Голос для чтения</div>
        <select id="voiceSelect" onchange="changeVoice()" aria-label="Голос для чтения" style="max-width:58%;min-width:180px">
          <option value="auto">Автоматически · русский голос</option>
        </select>
      </div>"""
replace_once(
    r'<div class="voice-card">[\s\S]*?</div>\s*<div class="setting-row">\s*<div class="setting-label">Скорость:',
    voice_card + """
      <div class="setting-row">
        <div class="setting-label">Скорость:""",
    "settings voice card and selector"
)

replace_once(
    r'<div class="setting-row voice-actions-row">[\s\S]*?<div class="setting-row">\s*<div class="setting-label">Размер текста',
    r"""<div class="setting-row voice-actions-row">
        <div class="setting-label">Проверка голоса</div>
        <div class="voice-action-buttons">
          <button class="tool-btn primary" id="testTTSBtn" onclick="testVoice()">Проверить голос</button>
        </div>
      </div>
      <div class="voice-status-panel" id="ttsDetailStatus">Используется встроенный голос браузера. Загрузка отдельной модели не требуется.</div>
      <div class="setting-row">
        <div class="setting-label">Размер текста""",
    "settings remove neural download"
)

# Update status wording and error log so it no longer claims neural speech.
html = html.replace("console.error('Neural TTS error:', error);", "console.error('Browser speech error:', error);")
html = html.replace("Русская нейросетевая модель · Историк", "Встроенный русский голос Chrome")
html = html.replace("Подключаю локальную голосовую модель…", "Использую встроенный голос браузера…")

# Persist the canonical compressed HTML and syntax-check the inline app script.
page.write_text(html, encoding='utf-8')
source.write_bytes(brotli.compress(html.encode('utf-8'), quality=9))
scripts = re.findall(r'<script\b[^>]*>([\s\S]*?)</script>', html, re.I)
Path('/tmp/history-reader-inline.js').write_text('\n'.join(s for s in scripts if s.strip()), encoding='utf-8')

required = [
    'SpeechSynthesisUtterance',
    'synthesis.pause()',
    'synthesis.resume()',
    'window.speechSynthesis.cancel()',
    'id="voiceSelect"',
    'AI-рассказ'
]
for token in required:
    if token not in html:
        raise SystemExit('Missing expected browser voice feature: ' + token)
if 'Загрузить голос · ≈51 МБ' in html or 'tts-worker.js?v=' in html:
    raise SystemExit('Obsolete neural voice download UI or worker reference remains.')

if html != original:
    subprocess.run(['git', 'config', 'user.name', 'github-actions[bot]'], check=True)
    subprocess.run(['git', 'config', 'user.email', '41898282+github-actions[bot]@users.noreply.github.com'], check=True)
    subprocess.run(['git', 'add', 'index.html.br'], check=True)
    staged = subprocess.run(['git', 'diff', '--cached', '--quiet'])
    if staged.returncode != 0:
        subprocess.run(['git', 'commit', '-m', 'Replace neural voice with built-in Chrome speech [skip ci]'], check=True)
        subprocess.run(['git', 'push', 'origin', 'HEAD:main'], check=True)
        print('Published source now uses browser SpeechSynthesis; no TTS model is downloaded.')
    else:
        print('Browser SpeechSynthesis source already up to date.')
else:
    print('No page changes required.')
