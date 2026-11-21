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

let session = null;
let vocab = null;
let loadingPromise = null;
let synthesisQueue = Promise.resolve();

function progress(stage, percent, message) {
  self.postMessage({ type: 'progress', stage, progress: percent, message });
}

async function fetchJson(url) {
  const response = await fetch(url, { mode: 'cors' });
  if (!response.ok) throw new Error(`Не удалось загрузить словарь голоса (${response.status}).`);
  return response.json();
}

async function getModelBytes() {
  let cache = null;
  try {
    if ('caches' in self) {
      cache = await caches.open(CACHE_NAME);
      const cached = await cache.match(MODEL_URL);
      if (cached) {
        progress('cache', 100, 'Голос найден в кэше браузера');
        return await cached.arrayBuffer();
      }
    }
  } catch (error) {
    console.warn('[TTS] Кэш недоступен:', error);
  }

  progress('download', 0, 'Скачиваю мужской русский голос · около 51 МБ');
  const response = await fetch(MODEL_URL, { mode: 'cors' });
  if (!response.ok) throw new Error(`Модель голоса недоступна (${response.status}). Проверь подключение к интернету.`);

  const total = Number(response.headers.get('content-length')) || 0;
  if (!response.body?.getReader) {
    const buffer = await response.arrayBuffer();
    try { if (cache) await cache.put(MODEL_URL, new Response(buffer.slice(0))); } catch (_) {}
    progress('download', 100, 'Модель скачана');
    return buffer;
  }

  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
    progress('download', total ? Math.round(received / total * 100) : null,
      `Скачано ${(received / 1048576).toFixed(1)} МБ${total ? ` из ${(total / 1048576).toFixed(1)} МБ` : ''}`);
  }

  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
  try {
    if (cache) await cache.put(MODEL_URL, new Response(bytes.slice(0), {
      headers: { 'Content-Type': 'application/octet-stream' }
    }));
  } catch (error) {
    console.warn('[TTS] Не удалось сохранить модель в кэш:', error);
  }
  progress('download', 100, 'Модель скачана; подготавливаю голосовой движок');
  return bytes.buffer;
}

async function loadEngine() {
  if (session && vocab) return;
  if (loadingPromise) return loadingPromise;

  loadingPromise = (async () => {
    progress('prepare', 0, 'Загружаю словарь и конфигурацию голоса');
    const [loadedVocab, modelBytes] = await Promise.all([fetchJson(VOCAB_URL), getModelBytes()]);
    if (!loadedVocab || typeof loadedVocab !== 'object' ||
        typeof loadedVocab['а'] !== 'number' ||
        typeof loadedVocab[BLANK_TOKEN] !== 'number') {
      throw new Error('Словарь голоса повреждён или имеет неизвестный формат.');
    }
    vocab = loadedVocab;

    progress('engine', 100, 'Подготавливаю ONNX Runtime Web');
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;

    session = await ort.InferenceSession.create(new Uint8Array(modelBytes), {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      executionMode: 'sequential',
      enableCpuMemArena: true,
      intraOpNumThreads: 1
    });

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

  // VitsTokenizer has add_blank=true: insert the "_" token (ID 1)
  // before, between and after characters. Space is a different token (ID 0).
  const blankId = BigInt(vocab[BLANK_TOKEN]);
  const result = new BigInt64Array(ids.length * 2 + 1);
  result.fill(blankId);
  for (let i = 0; i < ids.length; i++) result[i * 2 + 1] = BigInt(ids[i]);
  return result;
}

async function synthesize(text, id) {
  if (!session || !vocab) await loadEngine();

  const input = tokenize(text);
  const attention = new BigInt64Array(input.length);
  attention.fill(1n);
  const sid = new BigInt64Array([BigInt(MALE_SPEAKER_ID)]);

  const outputs = await session.run({
    input_ids: new ort.Tensor('int64', input, [1, input.length]),
    attention_mask: new ort.Tensor('int64', attention, [1, attention.length]),
    sid: new ort.Tensor('int64', sid, [1])
  });

  const outputName = session.outputNames[0];
  const waveform = outputs[outputName]?.data;
  if (!waveform?.length) throw new Error('Голосовая модель вернула пустой звук.');

  const audio = new Float32Array(waveform.length);
  for (let i = 0; i < waveform.length; i++) {
    audio[i] = Math.max(-1, Math.min(1, Number(waveform[i]) || 0));
  }
  self.postMessage({ type: 'audio', id, audio, sampling_rate: SAMPLE_RATE }, [audio.buffer]);
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
