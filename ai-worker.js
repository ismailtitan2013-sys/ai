/* Local LLM worker. Model inference runs in the browser; no inference API is used. */
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

env.allowRemoteModels = true;
env.useBrowserCache = true;
let generator = null;
let loading = null;

self.onmessage = async (event) => {
  const msg = event.data || {};
  if (msg.type === 'init') {
    try {
      if (!generator) {
        if (!loading) {
          loading = pipeline('text-generation', 'onnx-community/Qwen2.5-0.5B-Instruct', {
            dtype: 'q4f16',
            progress_callback: (p) => self.postMessage({ type: 'progress', progress: p })
          });
        }
        generator = await loading;
      }
      self.postMessage({ type: 'ready' });
    } catch (error) {
      loading = null;
      self.postMessage({ type: 'error', error: String(error?.message || error), stage: 'load' });
    }
    return;
  }
  if (msg.type === 'generate') {
    try {
      if (!generator) throw new Error('Модель ещё не загружена. Нажми «Загрузить LLM».');
      const prompt = [
        '<|im_start|>system\nТы — внимательный русскоязычный преподаватель истории. Отвечай по-русски, понятно и живо. Главный источник — переданный текст учебника. Не выдумывай даты, имена, причины и цитаты. Если сведений нет в источнике, прямо скажи об этом. Отделяй факты из учебника от дополнительного контекста.\n<|im_end|>',
        `<|im_start|>user\n${msg.prompt}\n<|im_end|>`,
        '<|im_start|>assistant\n'
      ].join('\n');
      const result = await generator(prompt, {
        max_new_tokens: Math.min(260, Math.max(80, msg.max_new_tokens || 210)),
        do_sample: true,
        temperature: 0.65,
        repetition_penalty: 1.08,
        return_full_text: false
      });
      let answer = result?.[0]?.generated_text ?? result?.generated_text ?? '';
      if (Array.isArray(answer)) answer = answer.at(-1)?.content || '';
      answer = String(answer).replace(/<\\|im_end\\|>[\\s\\S]*$/,'').replace(/^\\s*<\\|im_start\\|>assistant\\s*/,'').trim();
      self.postMessage({ type: 'result', id: msg.id, answer });
    } catch (error) {
      self.postMessage({ type: 'error', id: msg.id, error: String(error?.message || error), stage: 'generate' });
    }
  }
};
