---
name: commit
description: Stage and commit changes following Conventional Commits specification
disable-model-invocation: true
allowed-tools: Bash(git *)
argument-hint: "[--push]"
---

# Conventional Commit

Stage and commit pending changes following the [Conventional Commits](https://www.conventionalcommits.org/) specification.

## Format

```
<type>(<scope>): <description>

<body>

<footer>
```

- **type**: obrigatório — categoria da mudança (ver abaixo)
- **scope**: opcional — contexto afetado ex: `(auth)`, `(api)`, `(ui)`
- **description**: obrigatório — frase curta no imperativo, sem ponto final
- **body**: opcional — explica o **porquê**, não o o quê
- **footer**: opcional — `BREAKING CHANGE: ...` ou referência a issue `Closes #123`

## Types

| Type | Quando usar |
|---|---|
| `feat` | Nova funcionalidade |
| `fix` | Correção de bug |
| `docs` | Apenas documentação |
| `refactor` | Reestruturação sem novo comportamento |
| `perf` | Melhoria de performance |
| `test` | Adição ou correção de testes |
| `chore` | Build, dependências, CI, configuração |
| `style` | Formatação sem impacto funcional |
| `ci` | Mudanças em pipelines CI/CD |

## Instructions

1. Check what is staged — **only staged changes matter**:

!`git diff --cached`

!`git diff --cached --stat`

2. If nothing is staged, stop and tell the user to run `git add <files>` first.
3. Analyze **only the staged diff** to understand what changed and why.
4. Generate the commit message automatically — do not ask the user for it:
   - Choose the most accurate `type` and optional `scope` based solely on the staged diff.
   - Write the description in the imperative mood, lowercase, no period — e.g. `add station distance script` not `Added station distance script.`
   - Add a short body only if the reason behind the change is non-obvious.
5. Commit using a HEREDOC to preserve formatting:

```bash
git commit -m "$(cat <<'EOF'
<type>(<scope>): <description>

<body if needed>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

6. Verify with `git log --oneline -3`.
7. If the user passed `--push`, run `git push origin`.

## Examples

```
/commit
/commit --push
```

## Rules

- Never run `git add` — only commit what the user already staged.
- Never commit `.env`, credentials, or secrets — warn the user if found staged.
- Never use `--no-verify` unless explicitly requested.
- Never amend a previous commit — always create a new one.
- If nothing is staged, report it and stop.
