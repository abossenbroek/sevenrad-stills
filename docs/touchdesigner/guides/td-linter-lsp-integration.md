# td-linter LSP Integration Guide

td-linter includes a Language Server Protocol (LSP) server for real-time editor integration.

## Prerequisites

Install the LSP dependency:

```bash
pip install sevenrad-stills[td-linter-lsp]
```

## What is LSP?

The Language Server Protocol provides:
- Real-time diagnostics as you edit
- Error squiggles in your editor
- Violation messages on hover
- Integration with any LSP-compatible editor

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

## Editor Setup

### VS Code

Create `.vscode/settings.json` in your workspace:

```json
{
  "lsp.serverPath": "td-linter",
  "lsp.serverArgs": ["lsp"],
  "lsp.documentSelector": [
    { "pattern": "**/*.toe.dir/**/*.n" },
    { "pattern": "**/*.toe.dir/**/*.parm" },
    { "pattern": "**/*.toe.dir/**/*.text" }
  ]
}
```

Or use a generic LSP extension and configure:

```json
{
  "languageServerExample.serverPath": "td-linter",
  "languageServerExample.serverArgs": ["lsp"]
}
```

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

## Performance Considerations

- The server caches `.toe.dir` lookups
- Full project is linted on each save
- For large projects, consider using `--select` to limit rules

## Debugging

Enable verbose logging:

```bash
td-linter lsp 2>lsp.log
```

Or for TCP mode, inspect traffic:

```bash
td-linter lsp --transport tcp --port 2087
# In another terminal:
nc localhost 2087
```
