const CACHE_NAME = 'history-reader-studio-v4';
const SHELL = [
  './',
  './index.html',
  './manifest.webmanifest',
  './icon-192.png',
  './icon-512.png',
  './ai-worker.js'
];

self.addEventListener('install', event => {
  event.waitUntil((async () => {
    const cache = await caches.open(CACHE_NAME);
    await cache.addAll(SHELL);
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', event => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.filter(key =>
      key.startsWith('history-reader-') && key !== CACHE_NAME
    ).map(key => caches.delete(key)));
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', event => {
  const request = event.request;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  event.respondWith((async () => {
    const cache = await caches.open(CACHE_NAME);
    if (request.mode === 'navigate') {
      try {
        const fresh = await fetch(request);
        if (fresh.ok) cache.put('./index.html', fresh.clone()).catch(() => {});
        return fresh;
      } catch (_) {
        return (await cache.match('./index.html')) || Response.error();
      }
    }

    const isWorker = /\/ai-worker\.js$/.test(url.pathname);
    const cached = await cache.match(request);
    if (cached && !isWorker) return cached;
    try {
      const fresh = await fetch(isWorker ? new Request(request, { cache: 'reload' }) : request);
      if (fresh.ok) cache.put(request, fresh.clone()).catch(() => {});
      return fresh;
    } catch (_) {
      return cached || (isWorker ? (await cache.match(url.pathname)) : null) || Response.error();
    }
  })());
});
