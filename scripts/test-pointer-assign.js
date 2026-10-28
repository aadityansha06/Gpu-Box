function findMatchingDelimiter(str, start, openCh, closeCh) {
  let depth = 0;
  for (let i = start; i < str.length; i++) {
    if (str[i] === openCh) depth++;
    else if (str[i] === closeCh) {
      depth--;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function findStatementEnd(str) {
  let depth = 0;
  for (let i = 0; i < str.length; i++) {
    const ch = str[i];
    if (ch === '(' || ch === '[' || ch === '{') depth++;
    else if (ch === ')' || ch === ']' || ch === '}') depth--;
    else if (ch === ';' && depth === 0) return i;
  }
  return str.length;
}

function transformPointerAssignments(code, ptrNames) {
  for (let p = 0; p < ptrNames.length; p++) {
    const name = ptrNames[p];
    const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const assignPattern = new RegExp('\\b' + escaped + '\\s*\\[', 'g');
    let m;
    while ((m = assignPattern.exec(code)) !== null) {
      const bracketStart = m.index + m[0].length - 1;
      const bracketEnd = findMatchingDelimiter(code, bracketStart, '[', ']');
      if (bracketEnd === -1) continue;
      
      const indexExpr = code.substring(bracketStart + 1, bracketEnd).trim();
      const afterBracket = code.substring(bracketEnd + 1);
      
      const leadingWsMatch = afterBracket.match(/^\s*/);
      const wsLen = leadingWsMatch ? leadingWsMatch[0].length : 0;
      const trimmedAfter = afterBracket.substring(wsLen);
      
      // Compound
      const compoundMatch = trimmedAfter.match(/^(<<=|>>=|[+\-*/%&|^])=\s*/);
      if (compoundMatch) {
        const op = compoundMatch[1];
        const afterOp = trimmedAfter.substring(compoundMatch[0].length);
        const semiIdx = findStatementEnd(afterOp);
        const valExpr = afterOp.substring(0, semiIdx).trim();
        
        const totalLen = (bracketEnd + 1 + wsLen + compoundMatch[0].length + semiIdx + 1) - m.index;
        const replacement = name + '.set(' + indexExpr + ', ' + name + '.get(' + indexExpr + ') ' + op + ' (' + valExpr + '));';
        
        code = code.substring(0, m.index) + replacement + code.substring(m.index + totalLen);
        assignPattern.lastIndex = m.index + replacement.length;
        continue;
      }
      
      // Simple =
      const simpleMatch = trimmedAfter.match(/^=\s*/);
      if (simpleMatch) {
        const afterEq = trimmedAfter.substring(simpleMatch[0].length);
        const semiIdx = findStatementEnd(afterEq);
        const valExpr = afterEq.substring(0, semiIdx).trim();
        
        const totalLen = (bracketEnd + 1 + wsLen + simpleMatch[0].length + semiIdx + 1) - m.index;
        const replacement = name + '.set(' + indexExpr + ', ' + valExpr + ');';
        
        code = code.substring(0, m.index) + replacement + code.substring(m.index + totalLen);
        assignPattern.lastIndex = m.index + replacement.length;
        continue;
      }
    }
  }
  return code;
}

const inputCode = 'if (i < n) {\n    output[i] =\n        expf(input[i] - globalMax) / globalSum;\n}';
console.log('Transformed:');
console.log(transformPointerAssignments(inputCode, ['output', 'input']));
