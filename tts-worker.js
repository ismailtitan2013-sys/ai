/* Russian male VITS voice, generated locally in the browser. */
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.wasm.min.mjs';

const MODEL_BASE = 'https://huggingface.co/utrobinmv/tts_ru_free_hf_vits_low_multispeaker/resolve/main/';
const MODEL_URL = MODEL_BASE + 'model.onnx';
const VOCAB_URL = MODEL_BASE + 'vocab.json';
const CACHE_NAME = 'history-reader-tts-model-v1';
const SAMPLE_RATE = 16000;
const MALE_SPEAKER_ID = 1;
const BLANK_TOKEN = '_';
const UNKNOWN_TOKEN = '<unk>';
const NETWORK_IDLE_MS = 45000;

let session = null;
let vocab = null;
let loadingPromise = null;
let synthesisQueue = Promise.resolve();

function progress(stage, percent, message) {
  self.postMessage({ type: 'progress', stage, progress: percent, message });
}

async function fetchJson(url) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), NETWORK_IDLE_MS);
  try {
    const response = await fetch(url, { mode: 'cors', cache: 'force-cache', signal: controller.signal });
    if (!response.ok) throw new Error(`Не удалось загрузить словарь голоса (${response.status}).`);
    return await response.json();
  } catch (error) {
    if (controller.signal.aborted) throw new Error('Сервер не ответил при загрузке словаря голоса за 45 секунд.');
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

async function downloadModelAttempt() {
  const abort = new AbortController();
  let idleTimer = null;
  let stalled = false;
  const armIdleWatchdog = () => {
    clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
      stalled = true;
      abort.abort();
    }, NETWORK_IDLE_MS);
  };

  try {
    armIdleWatchdog();
    const response = await fetch(MODEL_URL, { mode: 'cors', cache: 'no-store', signal: abort.signal });
    if (!response.ok) throw new Error(`Сервер модели ответил кодом ${response.status}.`);

    const total = Number(response.headers.get('content-length')) || 0;
    if (!response.body?.tee) {
      progress('download', 0, 'Скачиваю мужскую голосовую модель…');
      const buffer = await response.arrayBuffer();
      if (buffer.byteLength < 40 * 1024 * 1024) {
        throw new Error('Сервер передал неполный файл голосовой модели.');
      }
      return buffer;
    }

    const [cacheBranch, dataBranch] = response.body.tee();
    const cachePromise = ('caches' in self)
      ? caches.open(CACHE_NAME)
          .then(cache => cache.put(MODEL_URL, new Response(cacheBranch, {
            headers: { 'Content-Type': 'application/octet-stream' }
          })))
          .then(() => true)
          .catch(error => {
            console.warn('[TTS] Не удалось сохранить голосовую модель в кэш:', error);
            return false;
          })
      : Promise.resolve(false);

    let received = 0;
    const reader = dataBranch.getReader();
    const chunks = [];
    try {
      while (true) {
        armIdleWatchdog();
        let packet;
        try {
          packet = await reader.read();
        } catch (error) {
          if (stalled) throw new Error('Загрузка остановилась: сервер не передавал данные 45 секунд.');
          throw error;
        }
        if (packet.done) break;
        chunks.push(packet.value);
        received += packet.value.byteLength;
        progress(
          'download',
          total ? Math.min(99, Math.round(received / total * 100)) : null,
          `Скачано ${(received / 1048576).toFixed(1)} МБ${total ? ` из ${(total / 1048576).toFixed(1)} МБ` : ''}`
        );
      }
    } catch (error) {
      abort.abort();
      try { await reader.cancel(); } catch (_) {}
      await cachePromise;
      throw error;
    } finally {
      clearTimeout(idleTimer);
    }

    const bytes = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.byteLength;
    }
    await cachePromise;

    if (bytes.byteLength < 40 * 1024 * 1024) {
      throw new Error(`Файл пришёл не полностью: ${(bytes.byteLength / 1048576).toFixed(1)} МБ.`);
    }
    return bytes.buffer;
  } finally {
    clearTimeout(idleTimer);
  }
}

async function getModelBytes() {
  try {
    if ('caches' in self) {
      const cache = await caches.open(CACHE_NAME);
      const cached = await cache.match(MODEL_URL);
      if (cached) {
        const buffer = await cached.arrayBuffer();
        if (buffer.byteLength >= 40 * 1024 * 1024) {
          progress('cache', 100, 'Мужской голос найден в кэше браузера');
          return buffer;
        }
        await cache.delete(MODEL_URL);
      }
    }
  } catch (error) {
    console.warn('[TTS] Не удалось прочитать кэш модели:', error);
  }

  let lastError = null;
  for (let attempt = 1; attempt <= 2; attempt++) {
    try {
      progress('download', 0, attempt === 1
        ? 'Скачиваю мужскую русскую голосовую модель · около 48,5 МБ'
        : 'Соединение прервалось. Повторяю загрузку…');
      return await downloadModelAttempt();
    } catch (error) {
      lastError = error;
      console.error(`[TTS] Попытка загрузки ${attempt} не удалась:`, error);
      if (attempt < 2) {
        try {
          const cache = await caches.open(CACHE_NAME);
          await cache.delete(MODEL_URL);
        } catch (_) {}
      }
    }
  }
  throw new Error(`Не удалось скачать голосовую модель: ${String(lastError?.message || lastError)}`);
}

async function loadEngine() {
  if (session && vocab) return;
  if (loadingPromise) return loadingPromise;

  loadingPromise = (async () => {
    progress('prepare', 0, 'Подготавливаю русский мужской голос');
    const [loadedVocab, modelBytes] = await Promise.all([fetchJson(VOCAB_URL), getModelBytes()]);
    if (!loadedVocab || typeof loadedVocab !== 'object' ||
        typeof loadedVocab['а'] !== 'number' ||
        typeof loadedVocab[BLANK_TOKEN] !== 'number') {
      throw new Error('Словарь голоса повреждён или имеет неизвестный формат.');
    }
    vocab = loadedVocab;

    progress('engine', null, 'Файл готов. Запускаю голосовой движок — это может занять время.');
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;

    // ONNX may spend several minutes compiling/initializing on slower PCs.
    // Send activity updates so the page's inactivity watchdog does not kill it.
    const heartbeat = setInterval(() => {
      progress('engine', null, 'Голосовой движок ещё запускается. Не закрывай страницу…');
    }, 12000);

    try {
      session = await ort.InferenceSession.create(new Uint8Array(modelBytes), {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
        executionMode: 'sequential',
        enableCpuMemArena: true,
        intraOpNumThreads: 1
      });
    } catch (error) {
      throw new Error(`Файл скачан, но ONNX/WASM не запустился: ${String(error?.message || error)}. Возможно, браузеру не хватает оперативной памяти.`);
    } finally {
      clearInterval(heartbeat);
    }

    const required = ['input_ids', 'attention_mask', 'sid'];
    const missing = required.filter(name => !session.inputNames.includes(name));
    if (missing.length) {
      await session.release();
      session = null;
      throw new Error('У голосовой модели неожиданные входы: ' + missing.join(', '));
    }
  })().catch(error => {
    session = null;
    vocab = null;
    throw error;
  }).finally(() => { loadingPromise = null; });

  return loadingPromise;
}

function tokenize(text) {
  const clean = String(text || '')
    .normalize('NFKC')
    .toLowerCase()
    .replace(/[\r\n\t]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

  const unknownId = typeof vocab[UNKNOWN_TOKEN] === 'number' ? vocab[UNKNOWN_TOKEN] : 41;
  const ids = [];
  for (const char of clean) {
    ids.push(Object.prototype.hasOwnProperty.call(vocab, char) ? vocab[char] : unknownId);
  }
  if (!ids.length) throw new Error('Нет текста для озвучивания.');

  const blankId = BigInt(vocab[BLANK_TOKEN]);
  const result = new BigInt64Array(ids.length * 2 + 1);
  result.fill(blankId);
  for (let i = 0; i < ids.length; i++) result[i * 2 + 1] = BigInt(ids[i]);
  return result;
}

async function synthesize(text, id) {
  if (!session || !vocab) await loadEngine();

  const cleanText = String(text || '').trim();
  if (!cleanText) throw new Error('Нет текста для озвучивания.');

  self.postMessage({ type: 'synthesis-progress', id, message: 'Генерирую речь нейросетью…' });

  const input = tokenize(cleanText);
  const attention = new BigInt64Array(input.length);
  attention.fill(1n);
  const sid = new BigInt64Array([BigInt(MALE_SPEAKER_ID)]);

  const startedAt = performance.now();
  const outputs = await session.run({
    input_ids: new ort.Tensor('int64', input, [1, input.length]),
    attention_mask: new ort.Tensor('int64', attention, [1, attention.length]),
    sid: new ort.Tensor('int64', sid, [1])
  });

  const outputName = session.outputNames[0];
  const waveform = outputs[outputName]?.data;
  if (!waveform?.length) throw new Error('Голосовая модель вернула пустой аудиосигнал.');

  const audio = new Float32Array(waveform.length);
  let peak = 0;
  let sumSquares = 0;
  for (let i = 0; i < waveform.length; i++) {
    const sample = Number(waveform[i]);
    if (!Number.isFinite(sample)) throw new Error('Модель вернула повреждённый аудиосигнал.');
    const clamped = Math.max(-1, Math.min(1, sample));
    audio[i] = clamped;
    const abs = Math.abs(clamped);
    if (abs > peak) peak = abs;
    sumSquares += clamped * clamped;
  }

  const rms = Math.sqrt(sumSquares / audio.length);
  const duration = audio.length / SAMPLE_RATE;
  const inferenceMs = Math.round(performance.now() - startedAt);

  if (peak < 0.00001 || rms < 0.000001) {
    throw new Error('Нейросеть вернула почти нулевой звук. Проверь текст страницы и перезапусти проверку голоса.');
  }
  if (duration < 0.08) {
    throw new Error('Получился слишком короткий аудиофрагмент. Попробуй другую фразу.');
  }

  self.postMessage({
    type: 'audio',
    id,
    audio,
    sampling_rate: SAMPLE_RATE,
    duration,
    peak,
    rms,
    inference_ms: inferenceMs
  }, [audio.buffer]);
}

self.onmessage = (event) => {
  const msg = event.data || {};
  if (msg.type === 'init') {
    loadEngine()
      .then(() => self.postMessage({ type: 'ready', voice: 'Russian male · speaker 1', sampling_rate: SAMPLE_RATE }))
      .catch(error => self.postMessage({ type: 'error', stage: 'load', error: String(error?.message || error) }));
    return;
  }
  if (msg.type === 'speak') {
    synthesisQueue = synthesisQueue.then(() => synthesize(String(msg.text || '').slice(0, 320), msg.id)).catch(error => {
      self.postMessage({ type: 'error', id: msg.id, stage: 'speak', error: String(error?.message || error) });
    });
  }
};
