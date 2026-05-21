# PyRIT jailbreak templates

Vendored from Microsoft's PyRIT repo, MIT-licensed.
Source: https://github.com/microsoft/PyRIT/tree/main/pyrit/datasets/jailbreak/templates

Each YAML has:
- `name`, `description`, `authors`, `source` — metadata
- `parameters: [prompt]` (most templates) — single substitution point
- `value: |-` or `value: >` — the template body with `{{ prompt }}` placeholder

Subdirectories:
- `Arth_Singh/`, `multi_parameter/`, `pliny/` — variant template collections (some take multiple parameters)

Intended use: a new attack `template_jailbreak` that, given a harmful behavior,
swaps it into the `{{ prompt }}` slot and sends the rendered text to the model.
