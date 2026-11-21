/* Local Russian neural TTS. Audio is generated on-device with Transformers.js. */
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

env.allowRemoteModels = true;
env.useBrowserCache = true;
let synthesizer = null;
let loading = null;

self.onmessage = async (event) => {
  const msg = event.data || {};
  if (msg.type === 'init') {
    try {
      if (!synthesizer) {
        if (!loading) {
          loading = pipeline('text-to-speech', 'Xenova/mms-tts-rus', {
            dtype: 'q8',
            progress_callback: (p) => self.postMessage({ type: 'progress', progress: p })
          });
        }
        synthesizer = await loading;
      }
      self.postMessage({ type: 'ready' });
    } catch (error) {
      loading = null;
      self.postMessage({ type: 'error', error: String(error?.message || error), stage: 'load' });
    }
    return;
  }
  if (msg.type === 'speak') {
    try {
      if (!synthesizer) throw new Error('Голосовая модель ещё не загружена.');
      const output = await synthesizer(String(msg.text || '').slice(0, 900));
      const audio = output.audio;
      self.postMessage({ type: 'audio', id: msg.id, audio, sampling_rate: output.sampling_rate }, [audio.buffer]);
    } catch (error) {
      self.postMessage({ type: 'error', id: msg.id, stage: 'speak', error: String(error?.message || error) });
    }
  }
};
