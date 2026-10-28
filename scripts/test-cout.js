function splitStreamShift(str) {
  const parts = [];
  let depth = 0;
  let inString = false;
  let strChar = '';
  let current = '';

  for (let i = 0; i < str.length; i++) {
    const ch = str[i];
    if (inString) {
      if (ch === '\\') { current += ch + (str[++i] || ''); continue; }
      if (ch === strChar) inString = false;
      current += ch;
      continue;
    }
    if (ch === '"' || ch === "'") {
      inString = true;
      strChar = ch;
      current += ch;
      continue;
    }
    if (ch === '(' || ch === '[' || ch === '{') depth++;
    else if (ch === ')' || ch === ']' || ch === '}') depth--;

    if (depth === 0 && ch === '<' && str[i + 1] === '<') {
      parts.push(current.trim());
      current = '';
      i++;
      continue;
    }
    current += ch;
  }
  if (current.trim().length > 0) parts.push(current.trim());
  return parts;
}

function transformCout(code) {
  const pattern = /\b(?:std\s*::\s*)?(?:cout|cerr)\s*<<\s*/g;
  let m;
  while ((m = pattern.exec(code)) !== null) {
    let semiIdx = -1;
    let inString = false;
    let strChar = '';
    for (let j = m.index; j < code.length; j++) {
      const ch = code[j];
      if (inString) {
        if (ch === '\\') { j++; continue; }
        if (ch === strChar) inString = false;
        continue;
      }
      if (ch === '"' || ch === "'") {
        inString = true;
        strChar = ch;
        continue;
      }
      if (ch === ';') {
        semiIdx = j;
        break;
      }
    }
    if (semiIdx === -1) continue;

    const chainStr = code.substring(m.index + m[0].length, semiIdx);
    const parts = splitStreamShift(chainStr);

    const jsParts = parts.map(p => {
      p = p.trim();
      if (p === 'std::endl' || p === 'endl') return '"\\n"';
      if (p === "'\\n'") return '"\\n"';
      return p;
    });

    const replacement = '__rt.printf(' + jsParts.join(' + ') + ')';
    code = code.substring(0, m.index) + replacement + code.substring(semiIdx);
    pattern.lastIndex = m.index + replacement.length;
  }
  return code;
}

const sample = `
for (int i = 0; i < 10; i++)
{
    std::cout << A[i] << " + " << B[i]
              << " = " << C[i] << '\\n';
}
`;

console.log(transformCout(sample));
