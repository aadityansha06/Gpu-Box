/**
 * rewriter.cpp
 * ────────────
 * Source-level CUDA → Simulator rewriter.
 *
 * Transforms:
 *   kernel<<<grid, block, shmem, stream>>>(args...)
 *   →
 *   [&]() {
 *     auto __grid = (grid); auto __block = (block);
 *     size_t __shmem = (shmem);
 *     __sim_launch_impl("kernel", __grid, __block, __shmem,
 *       [](void* __ctx) {
 *         auto& __c = *reinterpret_cast<decltype(__ctx_struct)*>(__ctx);
 *         kernel(__c.arg0, __c.arg1, ...);
 *       }, &__ctx_struct);
 *   }()
 *
 * MVP implementation: regex + paren-matching rewriter.
 * Handles: simple calls, template kernels, multi-line args, default shmem/stream.
 * Rejects: <<<>>> inside strings/comments (detected and skipped), ASM.
 *
 * Phase 4 upgrade: replace with LibTooling CUDAKernelCallExpr visitor.
 *
 * Also prepends:
 *   #include "cuda_sim.h"
 *   #line 1 "student.cu"
 * to map error messages back to student's file.
 */

#include "rewriter.h"
#include <string>
#include <sstream>
#include <regex>
#include <stdexcept>
#include <cstdio>

// ─── Utilities ────────────────────────────────────────────────────────────────

// Skip a balanced paren/angle region starting at pos.
// Returns the index past the closing delimiter.
static size_t skip_balanced(const std::string& s, size_t pos,
                             char open, char close) {
  int depth = 0;
  for (size_t i = pos; i < s.size(); i++) {
    if (s[i] == open)  depth++;
    if (s[i] == close) {
      depth--;
      if (depth == 0) return i + 1;
    }
  }
  throw std::runtime_error("Unbalanced delimiters in CUDA launch syntax");
}

// Check if position is inside a string or line comment
static bool in_string_or_comment(const std::string& s, size_t pos) {
  bool in_str   = false;
  bool in_cmnt  = false;
  bool in_blk   = false;
  for (size_t i = 0; i < pos; i++) {
    char c = s[i];
    if (in_cmnt) { if (c == '\n') in_cmnt = false; continue; }
    if (in_blk)  {
      if (c == '*' && i + 1 < s.size() && s[i+1] == '/') { in_blk = false; i++; }
      continue;
    }
    if (in_str)  { if (c == '"' && (i == 0 || s[i-1] != '\\')) in_str = false; continue; }
    if (c == '"') { in_str = true; continue; }
    if (c == '/' && i + 1 < s.size()) {
      if (s[i+1] == '/') { in_cmnt = true; i++; continue; }
      if (s[i+1] == '*') { in_blk  = true; i++; continue; }
    }
  }
  return in_str || in_cmnt || in_blk;
}

// ─── Main rewriter ────────────────────────────────────────────────────────────
std::string rewrite_cuda_source(const std::string& src) {
  std::string out;
  out.reserve(src.size() + 1024);

  // Prepend shim include and line directive
  out += "#include \"cuda_sim.h\"\n";
  out += "#line 1 \"student.cu\"\n";

  size_t i = 0;
  while (i < src.size()) {
    // Look for <<<
    size_t tri = src.find("<<<", i);
    if (tri == std::string::npos) {
      out.append(src, i, src.size() - i);
      break;
    }

    // Skip if in string or comment
    if (in_string_or_comment(src, tri)) {
      out.append(src, i, tri - i + 3);
      i = tri + 3;
      continue;
    }

    // Everything before the kernel call: find the kernel name by going backwards
    // from tri, skipping whitespace, then finding the identifier.
    size_t name_end = tri;
    while (name_end > 0 && src[name_end-1] == ' ') name_end--;
    size_t name_start = name_end;
    while (name_start > 0 &&
           (isalnum((unsigned char)src[name_start-1]) || src[name_start-1] == '_' ||
            src[name_start-1] == ':' || src[name_start-1] == '>')) {
      name_start--;
    }
    std::string kernel_name = src.substr(name_start, name_end - name_start);

    // Emit source up to start of kernel_name
    out.append(src, i, name_start - i);

    // Parse <<< launch config >>>
    size_t cfg_start = tri + 3;
    // Find matching >>>
    int depth = 1;
    size_t cfg_end = cfg_start;
    while (cfg_end < src.size() && depth > 0) {
      if (src[cfg_end] == '<') depth++;
      else if (src[cfg_end] == '>') {
        depth--;
        if (depth == 0) break;
      }
      // Also handle nested <<< >>>
      if (cfg_end + 2 < src.size() && src[cfg_end]=='<' && src[cfg_end+1]=='<' && src[cfg_end+2]=='<') {
        depth += 2; cfg_end += 2;
      }
      if (cfg_end + 2 < src.size() && src[cfg_end]=='>' && src[cfg_end+1]=='>' && src[cfg_end+2]=='>') {
        depth -= 2; cfg_end += 2;
        if (depth <= 0) break;
      }
      cfg_end++;
    }

    // Simple approach: find >>> directly
    size_t close3 = src.find(">>>", cfg_start);
    if (close3 == std::string::npos) {
      // Malformed; emit as-is
      out.append(src, name_start, tri + 3 - name_start);
      i = tri + 3;
      continue;
    }
    std::string cfg = src.substr(cfg_start, close3 - cfg_start);
    size_t after_cfg = close3 + 3;

    // Parse cfg: grid, block[, shmem[, stream]]
    // Split by top-level commas
    std::vector<std::string> cfg_parts;
    {
      std::string part;
      int pd = 0;
      for (char c : cfg) {
        if (c == '(' || c == '[' || c == '<') pd++;
        else if (c == ')' || c == ']' || c == '>') pd--;
        if (c == ',' && pd == 0) {
          cfg_parts.push_back(part);
          part.clear();
        } else {
          part += c;
        }
      }
      cfg_parts.push_back(part);
    }

    std::string grid_expr  = cfg_parts.size() > 0 ? cfg_parts[0] : "dim3(1)";
    std::string block_expr = cfg_parts.size() > 1 ? cfg_parts[1] : "dim3(1)";
    std::string shmem_expr = cfg_parts.size() > 2 ? cfg_parts[2] : "0";
    // stream (cfg_parts[3]) is ignored in sim

    // Parse kernel argument list
    // Find '(' after >>>
    size_t arg_open = after_cfg;
    while (arg_open < src.size() && src[arg_open] == ' ') arg_open++;
    if (arg_open >= src.size() || src[arg_open] != '(') {
      // Malformed — emit raw
      out += kernel_name;
      out.append(src, tri, close3 + 3 - tri);
      i = after_cfg;
      continue;
    }
    size_t arg_close = skip_balanced(src, arg_open, '(', ')');
    std::string args = src.substr(arg_open + 1, arg_close - arg_open - 2);

    // Emit the rewritten call:
    //   __sim_launch_kernel("name", (dim3)(grid), (dim3)(block), (size_t)(shmem),
    //     [=]() { kernel(args); })
    // We use a lambda to capture args by copy.
    out += "__sim_launch_kernel(\"";
    out += kernel_name;
    out += "\", (dim3)(";
    out += grid_expr;
    out += "), (dim3)(";
    out += block_expr;
    out += "), (size_t)(";
    out += shmem_expr;
    out += "), [=]() { ";
    out += kernel_name;
    out += "(";
    out += args;
    out += "); })";

    i = arg_close;
  }

  return out;
}

// ─── __sim_launch_kernel helper (declared in the shim, defined in api.cpp) ────
// This wraps the lambda into the void(*)(void*) + ctx form expected by
// __sim_launch_impl.  Defined here for linkage.

// (api.cpp provides the actual implementation using the template trick.)
