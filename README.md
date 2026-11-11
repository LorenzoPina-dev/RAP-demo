# RAP Demo

Implementazione di **Reasoning via Planning (RAP)** con **MCTS** su Blocksworld, supporto **Paper / Hybrid mode**, modelli locali (`llama.cpp`), API esterne (`litellm`) con vincoli TPM/RPM per modello, salvataggio/replay cache delle run e dashboard Web in tempo reale.

La modalità **paper** è la ricostruzione fedele di *Reasoning with Language Model is Planning with World Model* (Hao et al., EMNLP 2023): prompt di Appendix C.1, adaptive prompting di Appendix E, reward `r1 + r2`, MCTS di Algorithm 1 e aggiornamento dell'albero tramite il world model LLM. La modalità **hybrid** resta disponibile come interfaccia JSON aggiuntiva, non presente nel paper.

---

## Setup

```bash
uv sync
source .venv/bin/activate
```

*(Su Python 3.15, se necessario per compilare `fastuuid`: `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 uv sync`)*

---

## Configurazione: `.env` (API Key) + `models.json` (Modelli & Limiti)

- **`.env`** contiene solo i token API:
  ```bash
  cp .env.example .env
  ```
  ```env
  GEMINI_API_KEY=AIza...
  # OPENAI_API_KEY=sk-...
  # ANTHROPIC_API_KEY=sk-ant-...
  ```

- **`models.json`** contiene i profili dei modelli, backend, `api_base`, timeout e limiti **`tpm` / `rpm`** per singolo modello (`gemma4-26b`, `gemma4-31b`, `gpt-4o-mini`, `local-llama`, ecc.).
  ```bash
  # Elenca i modelli configurati in models.json
  rap models
  ```

---

## Attivare `llama.cpp` (Modello Locale)

1. **Installa `llama.cpp`** (macOS con Homebrew):
   ```bash
   brew install llama.cpp
   ```
2. **Avvia il server locale** scaricando un GGUF da Hugging Face (`-hf`) con GPU Metal (`-ngl 99`):
   ```bash
   llama-server -hf saidutta69/LFM3-VL-2B-heretic --port 8080 -ngl 99
   ```
3. In un altro terminale usa `--model local-llama` (configurato in `models.json`) oppure `--backend llama-server`.

---

## Utilizzo CLI

### 1. Esecuzione da terminale (`rap run`)

```bash
# Mock offline (nessun modello richiesto)
rap run --iterations 10 -v

# Usando un alias definito in models.json (risolve automaticamente backend, model, tpm, rpm)
rap run --model gemma4-26b -v
rap run --model gemma4-31b -v

# Modello ad-hoc non presente in models.json
rap run --backend litellm --model openai/gpt-4o-mini --tpm 30000 --rpm 60 -v

# Modello locale llama.cpp
rap run --model local-llama -v
```

### 2. Dashboard Web (`rap ui`)

Il modello si sceglie **solo dal comando nel terminale** (nessun selettore nella pagina web). Avvia e apri `http://127.0.0.1:8000`:

```bash
rap ui --backend mock
rap ui --model gemma4-26b --port 8000
rap ui --model local-llama --port 8000
```

### 3. Confronto Hybrid vs Paper (`rap compare`)

```bash
rap compare --backend mock --iterations 10
rap compare --model gemma4-26b --iterations 10
```

### 4. Cache & Replay delle Run (`rap history` / `rap replay`)

Ogni esecuzione (`rap run` e `rap ui` al completamento) salva automaticamente un JSON in `runs/` con data, modello, iterazioni e albero MCTS completo (`--no-save` per disabilitare):

```bash
# Elenca le run salvate
rap history

# Rivedi una run salvata da terminale (zero chiamate API)
rap replay latest -v
rap replay 1 -v

# Riaprila nella Web UI (zero chiamate API)
rap replay latest --ui --port 8000
```

---

## Stato iniziale e obiettivo

Il problema di default è definito in `src/rap_demo/domains/blocksworld/problem.py` (`DEFAULT_INITIAL_STATE`, `DEFAULT_GOAL`) e viene usato da `run`, `ui`, `compare` e dalla Web UI. Si può sovrascrivere da riga di comando:

```bash
# Obiettivo: più condizioni separate da "and" (oppure con --goal ripetuto)
rap run --goal "the red block is on top of the blue block and the blue block is on the table"

# Stato iniziale in JSON
rap run --initial-state '{"clear": ["red"], "on_table": ["blue"], "on": {"red": "blue"}, "holding": null}'

# Stato iniziale in inglese canonico (lo stesso formato mostrato dalla Web UI)
rap run --initial-state "the red block is clear, the hand is empty, the red block is on top of the blue block and the blue block is on the table."

# Anche con la Web UI e con i modelli locali
rap ui --model gpt-oss-20b --iterations 10 --goal "the orange block is on top of the red block"

# Limite di profondità L dell'albero (Algorithm 1). Deve essere almeno pari al
# numero minimo di azioni richieste dall'obiettivo: un obiettivo a 8 passi con
# il default L=6 non può essere risolto, per quanti nodi si esplorino.
rap run --model gpt-oss-20b --iterations 20 --max-depth 8 --goal "the red block is on top of the blue block" --goal "the yellow block is on top of the red block" --goal "the orange block is on top of the yellow block"
```

Nella Web UI puoi modificare stato iniziale e obiettivo senza riavviare: apri **Domain State → Edit problem**, incolla le condizioni (una per riga) e lo stato (JSON o inglese canonico), poi **Apply & reset**. La modifica azzera l'albero e aggiorna anche l'obiettivo usato dal modello e dalla funzione di reward.

Profili locali pronti in `models.json` (Ollama su `http://localhost:11434`):

| Profilo | Modello | Note |
| :--- | :--- | :--- |
| `gpt-oss-20b` | `ollama/gpt-oss:20b` | Consigliato su GPU con ≥16 GB VRAM (circa 2–5 s per chiamata a caldo). `reasoning_effort: "low"` per evitare che i reasoning token esauriscano il budget |
| `qwen2.5-coder-3b` | `ollama/qwen2.5-coder:3b` | Molto leggero, meno affidabile nel formato JSON |

---

## Ricostruzione del paper (modalità `paper`)

La modalità predefinita riproduce RAP come descritto in Hao et al. (2023):

| Componente | Riferimento | Implementazione |
| :--- | :--- | :--- |
| Prompt azione | Appendix C.1 | Regole di Blocksworld + 4 dimostrazioni `[STATEMENT] … [PLAN] … [PLAN END]`, prompt chiuso da `[PLAN]` |
| Prompt world model | Appendix C.1 | 2 scenari `[SCENARIO] … [STATE 1]`, cue `[CHANGE]`, stop a `[STATE 1]` |
| Adaptive prompting | Appendix E | Le dimostrazioni vengono accorciate dall'inizio man mano che l'albero scende (`remaining_steps`); gli stati intermedi sono ricalcolati in modo deterministico |
| Spazio azioni | §4.1 / Appendix A | Tutte le azioni legali del dominio, non un campione generato dal modello |
| Reward `r1` | §4.1 | Log-probabilità dell'azione sotto il prompt C.1, normalizzata con softmax sui fratelli |
| Reward `r2` | §4.1 | Frazione di condizioni dell'obiettivo soddisfatte dallo stato predetto |
| Reward combinato | §4.1 | `alpha * r1 + (1 - alpha) * r2` (`alpha = 0.5`); `100` quando tutte le condizioni sono vere |
| UCT | Eq. 1 / Appendix A | Figli non visitati (`N = 0`) hanno priorità massima e sono ordinati tra loro dal prior `r1`; `Q` medio per i figli visitati |
| Back-propagation | §3.3 / Eq. 2 / §4.3 | `Q(s, a)` è la media dei return delle simulazioni (media incrementale), non una somma; `M` conserva il massimo |
| Rollout | §3.3 | Simulazione leggera: fisica deterministica e nessuna chiamata di log-prob, perché il paper scarta i reward costosi durante la simulazione |
| Terminazione | Algorithm 1 | Nodo terminale quando l'obiettivo è raggiunto o si tocca `max_depth` (default 6) |

Note pratiche:

- Con backend che non espongono log-probabilità (per esempio Ollama) `r1` degenera in un prior uniforme: la ricerca funziona comunque grazie a `r2` e alla fisica del world model.
- Se il world model LLM produce una transizione non interpretabile, l'azione viene scartata; nei rollout si usa direttamente la fisica esatta come fallback, senza interrompere la ricerca.
- Il world model usa `max_tokens=2048`, `temperature=0.8` e stop su `[STATE 1]` (Appendix B.1).
- Il world model può omettere una clausola `clear` (per esempio dopo uno stack). Lo stato predetto viene normalizzato con le regole del prompt: `clear` e `on_table` sono ricalcolati da `on`/`holding`, così un blocco non resta "non clear" per una clausola mancante e lo spazio delle azioni non collassa.
- I prompt sono verbatim da Appendix C.1, con una sola aggiunta pratica: al world model si chiede esplicitamente di rispondere con una singola descrizione `[CHANGE]` senza spiegazioni, perché i modelli "reasoning" (es. `gpt-oss:20b`) altrimenti consumano il budget in ragionamento e non emettono mai il cambio di stato.

### Ispezione dei nodi: prompt e output del modello

Selezionando un nodo nell'albero, il pannello **Node Inspector** mostra:

- **Reasoning & Action** — il `thought` e l'azione associati al nodo;
- **Model Input / Output** — in modalità `paper` mostra l'azione valutata con `log P(action | state, goal)` e il valore `r1` normalizzato; in modalità `hybrid` mostra il prompt esatto e la risposta grezza, con `source` (`model` / `fallback`) e, in caso di fallback, il motivo (JSON non valido, azione illegale, output vuoto);
- **LLM Calls** — il log completo delle chiamate fatte per quel nodo: una voce per ogni azione valutata (`action scoring`, con `logprob` e `r1`) e una per ogni transizione del world model (`world model`, con prompt e risposta `[CHANGE]`). Ogni voce è espandibile e mostra prompt e risposta integrali. La sezione **Whole path** aggrega le stesse chiamate lungo tutto il percorso radice → nodo selezionato, raggruppate per profondità, così si vedono tutte le chiamate dei vari step in un unico posto. Le transizioni rifiutate restano visibili con l'errore e il fallback fisico.

I trace e i log delle chiamate vengono salvati anche nei file delle run, quindi restano consultabili con `rap replay latest --ui`. Le run create prima dell'introduzione dei trace non li contengono e il pannello lo segnala.

Schema JSON dello stato:

| Campo | Tipo | Significato |
| :--- | :--- | :--- |
| `clear` | lista di nomi | Blocchi liberi (nessun blocco sopra) |
| `on_table` | lista di nomi | Blocchi appoggiati sul tavolo |
| `on` | oggetto `blocco → blocco` | Relazioni "X è sopra Y" |
| `holding` | nome o `null` | Blocco nella mano (`null` = mano vuota) |

Le specifiche incoerenti vengono rifiutate con un errore: blocco in due posti, blocco in mano anche appoggiato, blocco `clear` che sostiene un altro blocco, ciclo nella relazione `on`.

---

## Flag Principali

| Flag | Valori | Default | Descrizione |
| :--- | :--- | :--- | :--- |
| `--model` | alias in `models.json` o ID modello | `None` | Seleziona modello (risolve anche backend/limiti da `models.json`) |
| `--config` | percorso `.json` | `models.json` | File JSON custom dei modelli e limiti |
| `--backend` | `mock`, `litellm`, `llama-server` | `mock` | Override esplicito del backend |
| `--api-base` | URL | `None` | Endpoint API custom |
| `--tpm` / `--rpm` | intero | da `models.json` | Override limiti **Token/Richieste al Minuto** |
| `--mode` | `paper`, `hybrid` | `paper` | Modalità di ragionamento |
| `--initial-state` | JSON o testo inglese canonico | stato demo | Stato iniziale del problema |
| `--goal` | condizione (flag ripetibile) | obiettivo demo | Obiettivo; più condizioni separate da `and` o con flag ripetuto |
| `--iterations` | intero | `10` (`50` su `ui`) | Iterazioni MCTS |
| `-v`, `--verbose` | flag | `False` | Stampa l'albero MCTS completo |

---

## Test

```bash
.venv/bin/pytest tests/
```
