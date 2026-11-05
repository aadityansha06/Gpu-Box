// Vercel serverless function: receives a bug report from the playground and emails it to the founder.
// Required env var (Vercel > Settings > Environment Variables):  RESEND_API_KEY
// Optional:  BUG_REPORT_TO (default below), BUG_REPORT_FROM (must be a verified Resend sender)

const TO = process.env.BUG_REPORT_TO || 'pkumariverma42@gmail.com';
const FROM = process.env.BUG_REPORT_FROM || 'Gpubox Bugs <onboarding@resend.dev>';
const MAX_IMAGES = 3;
const MAX_IMAGE_B64 = 1_800_000; // ~1.3 MB each after client-side resize

const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ ok: false, error: 'POST only' });
  const key = process.env.RESEND_API_KEY;
  if (!key) return res.status(503).json({ ok: false, error: 'not-configured' });

  let b = req.body;
  if (typeof b === 'string') { try { b = JSON.parse(b); } catch (e) { b = {}; } }
  b = b || {};
  if (b.website) return res.status(200).json({ ok: true }); // honeypot
  const message = String(b.message || '').trim().slice(0, 5000);
  if (message.length < 5) return res.status(400).json({ ok: false, error: 'Please describe the problem.' });
  const reporter = String(b.email || '').trim().slice(0, 200);

  const attachments = [];
  (Array.isArray(b.images) ? b.images : []).slice(0, MAX_IMAGES).forEach((im, i) => {
    const m = /^data:(image\/(?:png|jpeg|webp|gif));base64,([A-Za-z0-9+/=]+)$/.exec(String(im || ''));
    if (m && m[2].length <= MAX_IMAGE_B64) attachments.push({ filename: `screenshot-${i + 1}.${m[1].split('/')[1].replace('jpeg', 'jpg')}`, content: m[2] });
  });
  if (b.code) attachments.push({ filename: 'kernel.cu', content: Buffer.from(String(b.code).slice(0, 200000)).toString('base64') });

  const ctx = b.context || {};
  const text = [
    message, '',
    '---',
    `Reporter: ${reporter || '(not given)'}`,
    `GPU profile: ${ctx.profile || '-'}   Cluster: ${ctx.devices || 1}`,
    `Page: ${String(ctx.page || '').slice(0, 300)}`,
    `Browser: ${String(ctx.ua || '').slice(0, 200)}`,
    ctx.errors && ctx.errors.length ? `\nErrors shown to user:\n${ctx.errors.slice(0, 10).map(String).join('\n').slice(0, 3000)}` : '',
    ctx.output ? `\nConsole output:\n${String(ctx.output).slice(0, 2000)}` : ''
  ].join('\n');

  const payload = {
    from: FROM,
    to: [TO],
    subject: `[Gpubox bug] ${message.split('\n')[0].slice(0, 80)}`,
    text,
    html: `<pre style="font:13px/1.5 ui-monospace,monospace;white-space:pre-wrap">${esc(text)}</pre>`,
    attachments
  };
  if (/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(reporter)) payload.reply_to = reporter;

  try {
    const r = await fetch('https://api.resend.com/emails', {
      method: 'POST',
      headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!r.ok) return res.status(502).json({ ok: false, error: 'mail-provider' });
    return res.status(200).json({ ok: true });
  } catch (e) {
    return res.status(502).json({ ok: false, error: 'mail-provider' });
  }
}
