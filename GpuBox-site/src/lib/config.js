// Central site configuration.
// TODO(owner): replace CONTACT_EMAIL and the checkout URLs before going live.
export const CONTACT_EMAIL = 'hello@gpubox.dev';

const mail = (subject, body = '') =>
  `mailto:${CONTACT_EMAIL}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;

// Point these at a payment link (Stripe / Lemon Squeezy / Paddle) when ready.
// Until then the buttons open a pre-filled email so no lead is lost.
export const LINKS = {
  buyCommercial: mail('Gpubox Commercial License - $150/year', 'Company name:\nTeam size:\nBilling country (for taxes):\n'),
  buyEducation: mail('Gpubox Education License - $100/year', 'Institution name:\nOfficial institution email domain:\nCourse / department:\nBilling country (for taxes):\n'),
  openSource: mail('Gpubox for an Open Source project', 'Project name:\nRepository URL:\nOSI license:\nWhere Gpubox will be used (docs, CI, playground):\n'),
  contact: mail('Gpubox enquiry')
};

export const PRICING = {
  commercial: { price: 150, period: 'year' },
  education: { price: 100, period: 'year' }
};
