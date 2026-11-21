/* Local LLM worker: WebGPU when available, standard CPU/WASM otherwise. */
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

env.allowRemoteModels = true;
env.useBrowserCache = true;

const MODEL_ID = 'onnx-community/Qwen2.5-0.5B-Instruct';
let generator = null;
let loading = null;
let activeDevice = 'wasm';
let activeDtype = 'q8';

function progress(message, percent = null, file = '') {
  self.postMessage({ type: 'progress', message, progress: percent, file, device: activeDevice, dtype: activeDtype });
}

async function createPipeline(device, dtype) {
  activeDevice = device;
  activeDtype = dtype;
  progress(device === 'webgpu'
    ? 'Проверяю аппаратное ускорение WebGPU…'
    : 'Использую совместимый режим CPU/WASM.');
  const options = {
    dtype,
    progress_callback: (item) => {
      const detail = item || {};
      const pct = typeof detail.progress === 'number' ? detail.progress : null;
      progress(detail.file ? `Загрузка модели: ${detail.file}` : 'Загружаю локальную языковую модель…', pct, detail.file || '');
    }
  };
  // CPU/WASM is Transformers.js default; do not pass a made-up "wasm" device.
  if (device === 'webgpu') options.device = 'webgpu';
  return pipeline('text-generation', MODEL_ID, options);
}

async function loadModel() {
  if (generator) return;
  if (loading) return loading;
  loading = (async () => {
    let adapter = null;
    if (self.navigator?.gpu?.requestAdapter) {
      try { adapter = await self.navigator.gpu.requestAdapter({ powerPreference: 'low-power' }); }
      catch (_) { adapter = null; }
    }
    if (adapter) {
      try {
        generator = await createPipeline('webgpu', 'q4f16');
        return;
      } catch (error) {
        generator = null;
        progress('WebGPU не поддерживается этим устройством. Переключаюсь на CPU/WASM…');
        console.warn('[LLM] WebGPU fallback:', error);
      }
    }
    activeDevice = 'wasm';
    activeDtype = 'q8';
    generator = await createPipeline('wasm', 'q8');
  })().catch(error => {
    generator = null;
    throw error;
  }).finally(() => { loading = null; });
  return loading;
}

self.onmessage = async (event) => {
  const msg = event.data || {};
  if (msg.type === 'init') {
    try {
      await loadModel();
      self.postMessage({ type: 'ready', device: activeDevice, dtype: activeDtype });
    } catch (error) {
      self.postMessage({ type: 'error', stage: 'load', error: String(error?.message || error) });
    }
    return;
  }
  if (msg.type === 'generate') {
    try {
      if (!generator) await loadModel();
      const prompt = [
        '<|im_start|>system\nТы — внимательный русскоязычный преподаватель истории. Отвечай по-русски, понятно и живо. Главный источник — переданный текст учебника. Не выдумывай даты, имена, причины и цитаты. Если сведений нет в источнике, прямо скажи об этом. Отделяй факты из учебника от дополнительного контекста.\n<|im_end|>',
        `<|im_start|>user\n${msg.prompt}\n<|im_end|>`,
        '<|im_start|>assistant\n'
      ].join('\n');
      const result = await generator(prompt, {
        max_new_tokens: Math.min(240, Math.max(80, msg.max_new_tokens || 190)),
        do_sample: true,
        temperature: 0.62,
        repetition_penalty: 1.08,
        return_full_text: false
      });
      let answer = result?.[0]?.generated_text ?? result?.generated_text ?? '';
      if (Array.isArray(answer)) answer = answer.at(-1)?.content || '';
      answer = String(answer)
        .replace(/<\|im_end\|>[\s\S]*$/, '')
        .replace(/^\s*<\|im_start\|>assistant\s*/, '')
        .trim();
      self.postMessage({ type: 'result', id: msg.id, answer, device: activeDevice, dtype: activeDtype });
    } catch (error) {
      self.postMessage({ type: 'error', id: msg.id, stage: 'generate', error: String(error?.message || error) });
    }
  }
};
