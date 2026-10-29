/**
 * Deterministic WGSL AST and syntax normalizer for WebGPU compute shaders.
 * Enforces W3C WGSL address space rules:
 * 1. Hoists function-scope `var<workgroup>` declarations to module scope (outside fn main, before @compute).
 * 2. Rewrites runtime `const` declarations inside function bodies to `let`.
 * 3. Rewrites C++-style `auto` inside function bodies to `let`.
 * 4. Ensures module-scope variables use `const` (not `let`).
 */

/**
 * Sanitizes WGSL body scopes by hoisting function-scope var<workgroup> to module scope
 * and converting runtime const/auto inside function bodies to let.
 */
export function sanitizeWgslBodyScopes(wgsl: string): string {
  if (!wgsl || typeof wgsl !== "string") {
    return wgsl;
  }

  const lines = wgsl.split(/\r?\n/);
  let depth = 0;
  const hoistedVars: string[] = [];
  const processedLines: string[] = [];
  let accumulatingVarWorkgroup = false;
  let varWorkgroupBuffer = "";

  for (let i = 0; i < lines.length; i++) {
    let line = lines[i];

    // If currently accumulating a multi-line var<workgroup> declaration
    if (accumulatingVarWorkgroup) {
      varWorkgroupBuffer += " " + line.trim();
      if (line.includes(";")) {
        accumulatingVarWorkgroup = false;
        const semiIdx = varWorkgroupBuffer.indexOf(";");
        const fullDecl = varWorkgroupBuffer.slice(0, semiIdx + 1).trim();
        const remainder = varWorkgroupBuffer.slice(semiIdx + 1);
        if (!hoistedVars.includes(fullDecl)) {
          hoistedVars.push(fullDecl);
        }
        varWorkgroupBuffer = "";
        line = remainder;
      } else {
        continue;
      }
    }

    // Measure depth check for opening braces on the current line
    let lineInitialDepth = depth;
    let lineOpensBrace = line.includes("{");

    // Check if line contains a var<workgroup> declaration inside a function body
    if ((lineInitialDepth > 0 || (lineOpensBrace && depth > 0)) && /var\s*<\s*workgroup\s*>/.test(line)) {
      if (line.includes(";")) {
        line = line.replace(/var\s*<\s*workgroup\s*>\s+([^;]+);/g, (match) => {
          const decl = match.trim();
          if (!hoistedVars.includes(decl)) {
            hoistedVars.push(decl);
          }
          return "";
        });
      } else {
        // Multi-line var<workgroup> start detected
        const match = line.match(/(.*?)var\s*<\s*workgroup\s*>(.*)/);
        if (match) {
          processedLines.push(match[1]);
          varWorkgroupBuffer = "var<workgroup>" + match[2];
          accumulatingVarWorkgroup = true;
          // Update depth before continuing
          for (let c = 0; c < match[1].length; c++) {
            if (match[1][c] === "{") depth++;
            else if (match[1][c] === "}") depth = Math.max(0, depth - 1);
          }
          continue;
        }
      }
    }

    // Update brace depth for current line
    for (let c = 0; c < line.length; c++) {
      if (line[c] === "{") depth++;
      else if (line[c] === "}") depth = Math.max(0, depth - 1);
    }

    // Inside function body (depth > 0), convert const / auto to let
    if (depth > 0) {
      line = line.replace(/(\b)const(\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s*:[^=]+)?\s*=)/g, "$1let$2");
      line = line.replace(/(\b)auto(\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s*:[^=]+)?\s*=)/g, "$1let$2");
    }

    // At module scope (depth === 0), convert let to const
    if (depth === 0) {
      line = line.replace(/^(\s*)let(\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s*:[^=]+)?\s*=)/, "$1const$2");
    }

    processedLines.push(line);
  }

  let result = processedLines.join("\n");

  // Prepend hoisted var<workgroup> declarations at module scope directly above @compute
  if (hoistedVars.length > 0) {
    const hoistedBlock = hoistedVars.join("\n") + "\n\n";
    if (result.includes("@compute")) {
      result = result.replace("@compute", hoistedBlock + "@compute");
    } else {
      result = hoistedBlock + result;
    }
  }

  return result;
}

/**
 * Backward-compatible alias for sanitizeWgslBodyScopes.
 */
export const hoistWorkgroupVarsAndFixConst = sanitizeWgslBodyScopes;
