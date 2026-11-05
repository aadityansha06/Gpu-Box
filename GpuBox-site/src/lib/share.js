// Share links: the whole session (code, GPU, cluster size) lives in the URL hash.
// Nothing is uploaded; the recipient's browser decodes it locally.

const b64url = (bytes) => {
  let s = '';
  bytes.forEach((b) => { s += String.fromCharCode(b); });
  return btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
};
const unb64url = (str) => {
  const s = atob(str.replace(/-/g, '+').replace(/_/g, '/'));
  return Uint8Array.from(s, (c) => c.charCodeAt(0));
};

async function pipe(bytes, stream) {
  const out = new Blob([bytes]).stream().pipeThrough(stream);
  return new Uint8Array(await new Response(out).arrayBuffer());
}

export async function encodeState(state) {
  const raw = new TextEncoder().encode(JSON.stringify(state));
  if (typeof CompressionStream !== 'undefined') {
    return 'z.' + b64url(await pipe(raw, new CompressionStream('deflate-raw')));
  }
  return 'b.' + b64url(raw);
}

export async function decodeState(token) {
  try {
    const [kind, data] = [token.slice(0, 2), token.slice(2)];
    let bytes = unb64url(data);
    if (kind === 'z.') bytes = await pipe(bytes, new DecompressionStream('deflate-raw'));
    else if (kind !== 'b.') return null;
    const s = JSON.parse(new TextDecoder().decode(bytes));
    return s && typeof s.code === 'string' ? s : null;
  } catch (e) {
    return null;
  }
}

/** Build a full share URL for the given page path. */
export async function makeShareUrl(path, state) {
  const token = await encodeState(state);
  return `${location.origin}${path}#s=${token}`;
}

/** Read a shared session from the current URL hash, if any. */
export async function readSharedState() {
  const m = /[#&]s=([^&]+)/.exec(location.hash || '');
  return m ? decodeState(m[1]) : null;
}
