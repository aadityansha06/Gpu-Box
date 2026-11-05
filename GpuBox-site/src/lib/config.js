// Central site configuration. Licence buttons open a pre-filled email to CONTACT_EMAIL.
export const CONTACT_EMAIL = 'vermaadityansh@gmail.com';

const mail = (subject, body = '') =>
  `mailto:${CONTACT_EMAIL}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;

export const PRICING = {
  commercial: { price: 150, period: 'year' },
  education: { price: 100, period: 'year' }
};

export const LINKS = {
  buyCommercial: mail(
    `Gpubox Commercial License ($${PRICING.commercial.price}/year)`,
    'Hello,\n\nI would like to purchase a Gpubox Commercial license.\n\nCompany name:\nWebsite:\nTeam size:\nIntended use (CI, internal tooling, product integration):\nBilling country (for taxes):\nYour name and role:\n\nThank you.'
  ),
  buyEducation: mail(
    `Gpubox Education License ($${PRICING.education.price}/year)`,
    'Hello,\n\nI would like to purchase a Gpubox Education license.\n\nInstitution name:\nOfficial institution email domain:\nCourse or department:\nApproximate number of learners:\nBilling country (for taxes):\nYour name and role:\n\nThank you.'
  ),
  contact: mail(
    'Gpubox Enterprise Inquiry',
    'Hello,\n\nI would like to discuss a Gpubox Enterprise deployment.\n\nCompany name:\nWebsite:\nTeam size:\nDeployment needs (on-premises / air-gapped / private registry):\nCustom GPU or interconnect profiles required:\nTimeline:\nYour name and role:\n\nThank you.'
  ),
  openSource: mail(
    'Gpubox for an Open Source project',
    'Project name:\nRepository URL:\nOSI license:\nWhere Gpubox will be used (docs, CI, playground):\n'
  )
};
