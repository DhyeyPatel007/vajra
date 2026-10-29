/**
 * Deterministic WGSL AST and syntax normalizer for WebGPU compute shaders.
 * Enforces W3C WGSL address space rules:
 * 1. Hoists function-scope `var<workgroup>` declarations to module scope (outside fn main).
 * 2. Rewrites runtime `const` declarations inside function bodies to `let`.
 * 3. Rewrites C++-style `auto` inside function bodies to `let`.
 * 4. Ensures module-scope variables use `const` (not `let`).
 */

/**
 * Hoists any `var<workgroup>` declarations declared inside function bodies out to module scope,
 * and fixes any runtime `const` declarations inside functions to `let`.
 */
export function hoistWorkgroupVarsAndFixConst(wgsl: string): string {
  if (!wgsl || typeof wgsl !== "string") {
    return wgsl;
  }

  const lines = wgsl.split("\n");
  let depth = 0;
  const hoistedVars: string[] = [];
  const processedLines: string[] = [];

  for (let i = 0; i < lines.length; i++) {
    const rawLine = lines[i];
    let line = rawLine;

    // Check if line contains var<workgroup> inside a function body (depth > 0)
    if (depth > 0 && /var\s*<\s*workgroup\s*>\s+[a-zA-Z0-9_]+\s*:\s*[^;]+;/.test(line)) {
      line = line.replace(/var\s*<\s*workgroup\s*>\s+([a-zA-Z0-9_]+)\s*:\s*([^;]+);/g, (_match, name, type) => {
        const decl = `var<workgroup> ${name}: ${type.trim()};`;
        if (!hoistedVars.includes(decl)) {
          hoistedVars.push(decl);
        }
        return "";
      });
    }

    // Inside function body (depth > 0), convert const/auto to let
    if (depth > 0) {
      // Convert const with initializer to let
      line = line.replace(/^(\s*)const(\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s*:[^=]+)?\s*=)/, "$1let$2");
      // Convert C++ auto with initializer to let
      line = line.replace(/^(\s*)auto(\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s*:[^=]+)?\s*=)/, "$1let$2");
    }

    // At module scope (depth === 0), convert let to const
    if (depth === 0) {
      line = line.replace(/^(\s*)let(\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s*:[^=]+)?\s*=)/, "$1const$2");
    }

    // Track brace depth
    for (let c = 0; c < rawLine.length; c++) {
      const ch = rawLine[c];
      if (ch === "{") depth++;
      else if (ch === "}") depth = Math.max(0, depth - 1);
    }

    processedLines.push(line);
  }

  let result = processedLines.join("\n");

  // Prepend hoisted var<workgroup> at module scope
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

