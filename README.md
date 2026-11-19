# CORU-10M-ACT — CPU / TPU Automation Chat & Action Test Runtime

**RRR PRO MEX**  
**Model family:** CORU Automation  
**Model type:** Structured Automation Action-Routing Model  
**Hugging Face repository:** https://huggingface.co/rrrpromex/CORU-10M-ACT

---

## Runtime capabilities

- Resolves the current Hugging Face `main` revision and downloads that immutable snapshot. It does not silently fall back to an older model revision.
- Uses the repository's own `coru_inference.py`, strict integrity verifier and native `load()` API.
- Supports PyTorch CPU and TPU/XLA. `auto` prefers an active TPU/XLA backend, then CUDA, Apple MPS, and CPU.
- Treats Markdown documentation as optional in the private verification workspace while preserving checksum protection for every non-Markdown package file.
- Validates model architecture, parameter count, finite weights, tokenizer availability and the model's action schema.
- Extracts declared actions from explicit schema action/tool containers only. Parameter properties, required fields, examples and other nested JSON-schema data are not presented as separate action routes.
- Opens an interactive automation chat and provides individual action tests for routes in the action schema.
- Supports expected-route scoring inside chat, so a well-formed but semantically incorrect proposal is reported as a route mismatch rather than a successful test.
- Writes `coru_run.log` and `coru_test_report.json`.

## Colab / Jupyter usage

Upload `coru_loader.py` to the notebook session, then run:

```python
from coru_loader import main
main()
```

The loader ignores only Jupyter's injected `-f ...kernel-*.json` argument. Other unknown command-line arguments remain errors.

### Force CPU

```python
from coru_loader import main
main(["--device", "cpu"])
```

### Require TPU/XLA

```python
from coru_loader import main
main(["--device", "tpu"])
```

An explicitly requested TPU run stops if a working TPU/XLA device is not available. It does not silently fall back to CPU.

## Automation chat and route validation

After model integrity checks, native model loading, weight checks, and inference-contract smoke tests pass, the interactive console opens:

| Option | Function |
|---|---|
| `1` | Natural-language Automation Chat |
| `2` | Select and test one action from the schema |
| `3` | Walk through all parsed action declarations one by one |
| `4` | Display available action routes |
| `5` | Display recent scored route tests |
| `Q` | Exit |

### Score a chat request against an expected route

Inside option `1`, set the expected route before entering the request:

```text
/expect email.send
send the email
```

Or select a route by its displayed catalog number:

```text
/expect #12
Please send this email to Alex
```

The console compares both `tool` and `action` (when the tool is declared in the schema). It records `PASS` only when the prediction matches. A valid structured response with the wrong route, such as `clarify.ask` when `email.send` was expected, is recorded as `MISMATCH`.

Clear the expectation with:

```text
/expect off
```

Other chat commands: `/routes`, `/back`, and `/quit`.

Without an expected route, the console validates the proposal structure only and explicitly reports that semantic correctness was not scored. A valid JSON proposal is not proof that the selected action is correct.

## Individual action testing

Choose option `2`, select the required action from the catalog, enter a natural-language prompt, and accept or override the expected action. The model output is compared against the selected schema route. Option `3` repeats this process for every parsed action declaration.

For a command with missing information, evaluate both the route and the clarification. For example, a request such as `send the email` may lack recipient or message content, but its proposal should still be consistent with the email-send intent. A clarification about an unrelated reminder is a semantic mismatch, not a passing result.

## Structured proposal contract

A CORU inference response must contain:

```json
{
  "tool": "...",
  "action": "...",
  "parameters": {},
  "status": "...",
  "needs_confirmation": false,
  "clarification": null
}
```

CORU proposes structured actions. It does not itself click, open applications, create notes, send emails, or execute external tools. Actual execution requires a host application with the relevant connectors, permission checks, confirmation rules and adapters.

## Offline self-tests

Run the loader's offline suite without downloading the model:

```python
from coru_loader import main
main(["--self-test"])
```

The 20-test suite covers Jupyter compatibility, device selection, checksum/documentation policy, CPU mock integration, action-schema extraction, refusal to count parameter fields as action routes, and semantic route-mismatch reporting. It does not claim a fresh test of the remote checkpoint or physical TPU hardware.

---

**© All Rights Reserved By RRR PRO MEX**
