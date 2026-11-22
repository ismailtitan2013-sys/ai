from pathlib import Path
import re
import brotli
import subprocess

repo_root = Path('.').resolve()
source = repo_root / 'index.html.br'
site = repo_root / '_site'
site.mkdir(exist_ok=True)
page = site / 'index.html'
if not source.exists():
    raise SystemExit('Missing index.html.br')
html = brotli.decompress(source.read_bytes()).decode('utf-8')
original = html

# Fix the expired fixed timer, if an older source is encountered.
old_timeout = re.compile(
    r"const initTimer = setTimeout\(\(\) => \{\s*"
    r"reject\(failTTSWorker\(new Error\('Загрузка голоса не завершилась за 4 минуты\. Проверь интернет и попробуй ещё раз\.'\)\)\);\s*"
    r"\}, 240000\);\s*\n\s*"
    r"ttsWorker\.onmessage = \(event\) => \{\s*\n\s*"
    r"const m = event\.data \|\| \{\};"
)
timeout_replacement = """let initTimer = null;
const resetInitTimer = () => {
  clearTimeout(initTimer);
  initTimer = setTimeout(() => {
    reject(failTTSWorker(new Error('Голосовой движок не отвечал 90 секунд. Проверь соединение и повтори запуск.')));
  }, 90000);
};
resetInitTimer();

ttsWorker.onmessage = (event) => {
  const m = event.data || {};
  if (m.type === 'progress' && !ttsReady) resetInitTimer();"""
html, timeout_count = old_timeout.subn(lambda _m: timeout_replacement, html, count=1)
if timeout_count == 0 and 'Голосовой движок не отвечал 90 секунд' not in html:
    raise SystemExit('Voice init timeout source did not match expected versions.')

# Add the inference statistics and explicit progress event returned by the worker.
handler_old = """          pending.resolve({ audio: new Float32Array(m.audio), sampling_rate: m.sampling_rate });
        } else if (m.type === 'error') {"""
handler_new = """          pending.resolve({
            audio: new Float32Array(m.audio),
            sampling_rate: m.sampling_rate,
            duration: Number(m.duration) || 0,
            peak: Number(m.peak) || 0,
            rms: Number(m.rms) || 0
          });
        } else if (m.type === 'synthesis-progress' && ttsPending.has(m.id)) {
          updateReadStatus(m.message || 'Генерирую речь…');
        } else if (m.type === 'error') {"""
if handler_old in html:
    html = html.replace(handler_old, handler_new, 1)
elif "m.type === 'synthesis-progress' && ttsPending.has(m.id)" not in html:
    raise SystemExit('TTS worker response handler was not found.')

# Make PCM output robust: verify signal, normalize very quiet frames, and report WebAudio failures.
play_new = """function playPcm(samples, sampleRate, token) {
  return new Promise(async (resolve, reject) => {
    let source = null;
    let gainNode = null;
    try {
      if (!samples || !samples.length) throw new Error('Голосовая модель не передала аудиоданные.');
      if (token !== ttsPlaybackToken) return resolve();

      const ctx = getSpeechAudioContext();
      if (!state.isPaused && ctx.state !== 'running') await ctx.resume();
      if (token !== ttsPlaybackToken) return resolve();
      if (!state.isPaused && ctx.state !== 'running') {
        throw new Error('Chrome не запустил аудиоустройство. Проверь громкость и выход звука Windows.');
      }

      let peak = 0;
      for (let i = 0; i < samples.length; i++) {
        const value = Math.abs(Number(samples[i]) || 0);
        if (value > peak) peak = value;
      }
      if (peak < 0.00001) throw new Error('Модель вернула тишину вместо речи. Нажми «Проверить звук» и посмотри статус.');

      const scale = peak < 0.12 ? Math.min(8, 0.82 / peak) : 1;
      const normalized = new Float32Array(samples.length);
      for (let i = 0; i < samples.length; i++) {
        normalized[i] = Math.max(-1, Math.min(1, (Number(samples[i]) || 0) * scale));
      }

      const safeRate = Math.max(8000, Math.min(48000, Number(sampleRate) || 16000));
      const buffer = ctx.createBuffer(1, normalized.length, safeRate);
      buffer.copyToChannel(normalized, 0);

      source = ctx.createBufferSource();
      gainNode = ctx.createGain();
      source.buffer = buffer;
      source.playbackRate.value = Math.max(.65, Math.min(1.6, optsRate()));
      source.detune.value = Math.max(-280, Math.min(280, (state.pitch - 1) * 380));
      gainNode.gain.value = 1;
      source.connect(gainNode);
      gainNode.connect(ctx.destination);
      ttsCurrentSource = source;

      updateReadStatus('Воспроизвожу речь…');
      source.onended = () => {
        if (ttsCurrentSource === source) ttsCurrentSource = null;
        try { source.disconnect(); gainNode.disconnect(); } catch (_) {}
        resolve();
      };
      source.start(0);
    } catch (error) {
      try { if (source) source.disconnect(); if (gainNode) gainNode.disconnect(); } catch (_) {}
      reject(error);
    }
  });
}"""
play_pattern = re.compile(r"function playPcm\(samples, sampleRate, token\) \{[\s\S]*?\n\}\n\nfunction optsRate\(\)")
play_match = play_pattern.search(html)
if play_match:
    html = html[:play_match.start()] + play_new + "\n\nfunction optsRate()" + html[play_match.end():]
elif 'Chrome не запустил аудиоустройство' not in html:
    raise SystemExit('PCM playback function was not found.')

# Show whether the model is synthesizing or actually playing each phrase.
speak_old = """    updateReadStatus(`Готовлю фразу ${index + 1} из ${chunks.length}…`);
    const result = await requestTTSAudio(chunks[index]);
    if (token !== ttsPlaybackToken) return;
    await playPcm(result.audio, result.sampling_rate, token);"""
speak_new = """    updateReadStatus(`Генерирую фразу ${index + 1} из ${chunks.length}…`);
    const result = await requestTTSAudio(chunks[index]);
    if (token !== ttsPlaybackToken) return;
    if (!result.audio || !result.audio.length) throw new Error('Нейросеть не вернула аудиофрагмент.');
    updateReadStatus(`Воспроизвожу фразу ${index + 1} из ${chunks.length}…`);
    await playPcm(result.audio, result.sampling_rate, token);"""
if speak_old in html:
    html = html.replace(speak_old, speak_new, 1)
elif 'Генерирую фразу ${index + 1} из ${chunks.length}' not in html:
    raise SystemExit('Speech synthesis loop was not found.')

html = html.replace('./tts-worker.js?v=3.1.1', './tts-worker.js?v=3.2.0')
page.write_text(html, encoding='utf-8')

# Keep the canonical compressed source in the repository in sync with Pages.
compressed = brotli.compress(html.encode('utf-8'), quality=9)
source.write_bytes(compressed)

# Write an extracted inline script to let the workflow run a real syntax check.
scripts = re.findall(r'<script\b[^>]*>([\s\S]*?)</script>', html, re.I)
Path('/tmp/history-reader-inline.js').write_text('\n'.join(s for s in scripts if s.strip()), encoding='utf-8')

if html != original:
    subprocess.run(['git', 'config', 'user.name', 'github-actions[bot]'], check=True)
    subprocess.run(['git', 'config', 'user.email', '41898282+github-actions[bot]@users.noreply.github.com'], check=True)
    subprocess.run(['git', 'add', 'index.html.br'], check=True)
    staged = subprocess.run(['git', 'diff', '--cached', '--quiet'])
    if staged.returncode != 0:
        subprocess.run(['git', 'commit', '-m', 'Fix neural voice playback and expose synthesis status [skip ci]'], check=True)
        subprocess.run(['git', 'push', 'origin', 'HEAD:main'], check=True)
        print('Updated compressed page source.')
    else:
        print('Compressed page source was already up to date.')
else:
    print('No page changes required.')
