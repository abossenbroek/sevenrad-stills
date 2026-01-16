# td-linter LSP Integration Guide

td-linter includes a Language Server Protocol (LSP) server for real-time editor integration.

## Prerequisites

Install the LSP dependency:

```bash
pip install sevenrad-stills[td-linter-lsp]
```

Verify installation:
```bash
td-linter lsp --help
```

## What is LSP?

The Language Server Protocol provides:
- Real-time diagnostics as you edit
- Error squiggles in your editor
- Violation messages on hover
- Integration with any LSP-compatible editor

## Quick Test

Before configuring your editor, verify the LSP server works:

```bash
# Start in TCP mode for testing
td-linter lsp --transport tcp --port 2087
```

In another terminal:
```bash
# Test connection
nc localhost 2087
# Should connect without error
```

Or test the full protocol:
```bash
# Start LSP and send a basic request
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{}}}' | td-linter lsp
```

You should see a JSON response with `"result":{"capabilities":...}`.

## Starting the LSP Server

### stdio Mode (Default)

For editor integration:

```bash
td-linter lsp
```

The server communicates via stdin/stdout.

### TCP Mode

For remote or debugging:

```bash
td-linter lsp --transport tcp --port 2087
```

### WebSocket Mode

For browser-based editors:

```bash
td-linter lsp --transport ws --port 2087
```

### Server Options

| Option | Default | Description |
|--------|---------|-------------|
| `--transport` | `stdio` | Transport method: `stdio`, `tcp`, `ws` |
| `--host` | `127.0.0.1` | Host address for TCP/WebSocket |
| `--port` | `2087` | Port number for TCP/WebSocket |

## Editor Setup

### VS Code

For VS Code, you can use the [vscode-languageclient](https://marketplace.visualstudio.com/items?itemName=UnifiedJS.vscode-mdx) extension or create a custom extension.

**Option 1: Using a generic LSP extension**

Install an LSP client extension (like "LSP Sample" or similar), then configure `.vscode/settings.json`:

```json
{
  "languageServerExample.serverPath": "td-linter",
  "languageServerExample.serverArgs": ["lsp"],
  "files.associations": {
    "*.toe.dir/**/*.n": "tdn",
    "*.toe.dir/**/*.parm": "tdparm",
    "*.toe.dir/**/*.text": "tdtext"
  }
}
```

**Option 2: Using tasks.json for debugging**

Create `.vscode/tasks.json`:
```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "td-linter LSP (TCP)",
      "type": "shell",
      "command": "td-linter",
      "args": ["lsp", "--transport", "tcp", "--port", "2087"],
      "isBackground": true,
      "problemMatcher": []
    }
  ]
}
```

**Recommended Extensions:**
- **GLSL Syntax**: For syntax highlighting in `.text` files containing shaders
- **Python**: For Python code in `.text` files

### Neovim (nvim-lspconfig)

Add to your Neovim configuration:

```lua
local lspconfig = require('lspconfig')
local configs = require('lspconfig.configs')

-- Define td-linter server
if not configs.td_linter then
  configs.td_linter = {
    default_config = {
      cmd = { 'td-linter', 'lsp' },
      filetypes = { 'tdn', 'tdparm', 'tdtext' },
      root_dir = function(fname)
        return lspconfig.util.find_git_ancestor(fname)
          or lspconfig.util.path.dirname(fname)
      end,
    },
  }
end

lspconfig.td_linter.setup{}
```

Associate file types in `init.lua`:

```lua
vim.filetype.add({
  pattern = {
    ['.*%.toe%.dir/.*%.n'] = 'tdn',
    ['.*%.toe%.dir/.*%.parm'] = 'tdparm',
    ['.*%.toe%.dir/.*%.text'] = 'tdtext',
  },
})
```

### Sublime Text

Install the LSP package, then add to settings:

```json
{
  "clients": {
    "td-linter": {
      "command": ["td-linter", "lsp"],
      "selector": "source.tdn | source.tdparm",
      "schemes": ["file"]
    }
  }
}
```

### Emacs (lsp-mode)

Add to your Emacs configuration:

```elisp
(require 'lsp-mode)

(add-to-list 'lsp-language-id-configuration
  '("\\.toe\\.dir/.*\\.n\\'" . "tdn"))

(lsp-register-client
  (make-lsp-client
    :new-connection (lsp-stdio-connection '("td-linter" "lsp"))
    :major-modes '(tdn-mode)
    :server-id 'td-linter))
```

### Helix

Add to `~/.config/helix/languages.toml`:

```toml
[[language]]
name = "tdn"
scope = "source.tdn"
file-types = ["n"]
roots = [".toe.dir"]
language-servers = ["td-linter"]

[[language]]
name = "tdparm"
scope = "source.tdparm"
file-types = ["parm"]
roots = [".toe.dir"]
language-servers = ["td-linter"]

[language-server.td-linter]
command = "td-linter"
args = ["lsp"]
```

### Zed

Add to Zed settings (`~/.config/zed/settings.json`):

```json
{
  "lsp": {
    "td-linter": {
      "binary": {
        "path": "td-linter",
        "arguments": ["lsp"]
      }
    }
  },
  "languages": {
    "tdn": {
      "language_servers": ["td-linter"]
    }
  }
}
```

### Kate / KTextEditor

Create `~/.local/share/kate/lsp/td-linter.json`:

```json
{
  "servers": {
    "tdn": {
      "command": ["td-linter", "lsp"],
      "highlightingModeRegex": "^(tdn|TouchDesigner)$"
    }
  }
}
```

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                         Editor                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  project.toe.dir/ops/myop.n                         │   │
│  │  ┌──────────────────────────────────────────────┐   │   │
│  │  │ TOP:null                                     │   │   │
│  │  │ ~~~~~~ Error: Missing operator reference    │   │   │
│  │  │ inputs                                       │   │   │
│  │  │ {                                            │   │   │
│  │  │ 0   missing_op                               │   │   │
│  │  │     ~~~~~~~~~ Warning: Dangling input       │   │   │
│  │  │ }                                            │   │   │
│  │  └──────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
        │                                    │
        │ textDocument/didOpen               │
        │ textDocument/didSave               │
        ▼                                    │
┌─────────────────────────────────────────────────────────────┐
│                    td-linter LSP Server                     │
│  1. Finds .toe.dir from file path                           │
│  2. Runs linting on project                                 │
│  3. Converts violations to diagnostics                      │
│  4. Publishes diagnostics to editor                         │
└─────────────────────────────────────────────────────────────┘
```

## LSP Events

The server responds to:

| Event | Action |
|-------|--------|
| `textDocument/didOpen` | Lint and publish diagnostics |
| `textDocument/didSave` | Re-lint and update diagnostics |
| `textDocument/didClose` | Clear diagnostics for file |

## Diagnostic Severity Mapping

| td-linter Severity | LSP Severity |
|-------------------|--------------|
| error | Error (red squiggle) |
| warning | Warning (yellow squiggle) |
| info | Information (blue squiggle) |

## Troubleshooting

### "pygls not installed"

Install the LSP dependency:

```bash
pip install sevenrad-stills[td-linter-lsp]
```

### No Diagnostics Appearing

1. Check the server is running:
   ```bash
   td-linter lsp --transport tcp --port 2087
   ```
   Then connect with `telnet localhost 2087`

2. Check file is in a `.toe.dir` directory

3. Check your editor's LSP logs

### Server Crashes

Check stderr output for Python errors. Common issues:
- Missing dependencies
- Permission errors
- Invalid project structure

### Wrong Diagnostics Location

LSP uses 0-based line/column numbers. If diagnostics appear on wrong lines:
1. Check the file wasn't modified after linting
2. Report a bug if consistent

## Performance Tuning

### Caching

The server caches:
- `.toe.dir` path lookups from file URIs
- Parsed file contents (short TTL)

### Large Projects

For large projects with many operators:

1. **Limit rules** - Skip expensive rules:
   ```yaml
   # td-linter.yaml
   ignore: [F, G, P]  # Skip performance, GLSL, Python rules
   ```

2. **Use minimal preset**:
   ```yaml
   extends: minimal
   ```

3. **Disable embedded validation**:
   Configure to skip GLSL/Python validation which involves external tools.

### Memory Usage

The LSP server loads the full project graph for each lint. For very large projects (1000+ operators):
- Expect ~50-100MB RAM usage
- Consider splitting into smaller `.toe.dir` projects

## Debugging

### Enable Verbose Logging

Redirect stderr to a log file:

```bash
td-linter lsp 2>lsp.log
```

View logs in real-time:
```bash
tail -f lsp.log
```

### TCP Mode Debugging

Run in TCP mode to inspect traffic:

```bash
# Terminal 1: Start server
td-linter lsp --transport tcp --port 2087

# Terminal 2: Connect and watch
nc localhost 2087

# Terminal 3: Send test requests
echo '{"jsonrpc":"2.0","id":1,"method":"shutdown"}' | nc localhost 2087
```

### Common Debug Steps

1. **Verify server starts:**
   ```bash
   td-linter lsp --transport tcp --port 2087
   # Should show no errors
   ```

2. **Check file detection:**
   The server must find a `.toe.dir` parent directory. Check your file path.

3. **Inspect editor logs:**
   - VS Code: Output panel > "Language Server"
   - Neovim: `:LspLog`
   - Sublime: View > Show Console

4. **Test manually:**
   ```bash
   # Direct lint to verify rules work
   td-linter lint path/to/project.toe.dir --verbose
   ```

### Debug Checklist

| Issue | Check |
|-------|-------|
| No diagnostics | Is file in a `.toe.dir`? Is LSP connected? |
| Wrong file path | Check URI encoding (spaces, special chars) |
| Slow response | Try `--select S,C` to limit rules |
| Server crash | Check stderr log for Python traceback |
| Missing squiggles | Check severity mapping, editor config |
