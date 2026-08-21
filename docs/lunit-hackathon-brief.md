# Lunit Hackathon — Platform Brief

Consolidated English reference for the Lunit hackathon: MCP tooling, the Lunit FM L2 model,
the Patient Simulator, harness design, and submission requirements.

---

## 1. Endpoints & Credentials

One team API key (prefix `lunit_`) authenticates **all three** services.

| Service | URL | Auth |
| --- | --- | --- |
| Model API (Lunit FM L2) | `https://model.hackathon.lunit.io` | `Authorization: Bearer <API_KEY>` |
| Patient Simulator | `https://patient.hackathon.lunit.io` | same key |
| MCP server (Streamable HTTP) | `https://mcp.hackathon.lunit.io/mcp` | same key |

```bash
export LUNIT_FM_API_URL="https://model.hackathon.lunit.io"
export LUNIT_FM_API_KEY="lunit_..."
export LUNIT_FM_MODEL="Lunit/L2-preview"
```

> All Lunit assets (endpoints, dashboard) are reachable **only from inside the Lunit network**.
> Remote participation is allowed, but evaluation runs in a fully isolated environment with no
> external network access.

---

## 2. MCP Server Connection

### Codex configuration (`~/.codex/config.toml`)

```toml
[mcp_servers.lunit_mcp]
url = "https://mcp.hackathon.lunit.io/mcp"
bearer_token_env_var = "LUNIT_FM_API_KEY"
required = true
tool_timeout_sec = 60
```

Steps: export the key → add the server block → restart Codex → run `/mcp` to verify the connection.

- `lunit_mcp` is an arbitrary local server name; rename it freely.
- Any other Streamable HTTP MCP client works with the same endpoint and the same
  `Authorization: Bearer <API_KEY>` header.

---

## 3. Available MCP Tools

Tools are exposed with the prefix `mcp__lunit_mcp__`.

### 3.1 Drug labels & regulatory

| Tool | Source | Description |
| --- | --- | --- |
| `adr_retrieve_drug_info` | `dailymed_26_08` (DailyMed) | Look up key sections of an official DailyMed drug label by English brand name or INN. Includes warnings, adverse reactions, interactions, and a source link. |
| `openapi_mfds_check_drug_permission` | MFDS Drug Approval OpenAPI | Check a drug's current MFDS approval status via partial product-name search; distinguishes valid from withdrawn approvals. |
| `openapi_mfds_find_drugs_by_ingredient` | MFDS Drug Approval OpenAPI | Find MFDS-approved products sharing the same active ingredient; returns approval status of alternative candidates. |
| `openapi_mfds_get_drug_indication` | MFDS Product Approval Detail OpenAPI | Retrieve MFDS-approved indications; optionally dosage, administration, warnings, ATC data, and contraindications. |
| `openapi_hira_get_drug_price` | HIRA Drug Price OpenAPI | Reimbursement listing status, drug price code, and ceiling price from HIRA drug-price data, including deletion and effective-date info. |

### 3.2 HIRA reimbursement & Korean law

| Tool | Source | Description |
| --- | --- | --- |
| `hira_updates_search` | `hira_biz_infobank`, `hira_cancer_drug_notice`, `hira_cancer_drug_regimen` | Search HIRA reimbursement-criteria notices and published deliberation cases across current/revised guidance, oncology notices, and recognized off-label oncology regimens. |
| `openapi_hira_disease_check_code` | HIRA Disease Master OpenAPI | Verify whether a diagnosis code is valid for HIRA claims; returns code completeness plus principal-diagnosis, sex, age, and infectious-disease restrictions. |
| `openapi_law_search` | law.go.kr OpenAPI | Search Korean statutes; returns the `MST` identifier needed for follow-up lookups of acts, administrative rules, and local ordinances. |
| `openapi_law_list_articles` | law.go.kr OpenAPI | List the articles of a given Korean statute, with title filtering, returning stable article keys for full-text retrieval. |
| `openapi_law_get_article` | law.go.kr OpenAPI | Retrieve the citable full text of a selected article, including its enforcement date and a law.go.kr link. |

### 3.3 Document index (HIRA: 249 docs · guideline: 120 docs)

| Tool | Description |
| --- | --- |
| `index_list_documents` | List corpus documents from the HIRA and clinical-guideline collections, or rank them by query relevance. |
| `index_get_relevant_nodes` | Find document sections semantically related to a query; returns matching documents, ancestor nodes, and page ranges. |
| `index_get_document_structure` | Walk a document as a section tree from a starting node; returns up to 50 nodes with page ranges. |
| `index_get_page_content` | Return raw text for a chosen page range. Pages are 1-indexed; max 20 pages per call; also returns extracted flowchart paths. |
| `index_keyword_search` | Case-insensitive exact-keyword page search, ranked by matched terms and occurrence counts, with pagination. |

### 3.4 KCD codes

| Tool | Description |
| --- | --- |
| `kcd_search_codes` | Fuzzy-search candidate KCD codes by Korean or English disease name; KCD version selectable. |
| `kcd_get_name` | Return the official Korean/English disease name for an exact KCD code. Supports KCD-8 and KCD-9 (default KCD-9). |

### 3.5 Generic RAG layer

Data sources: `pubmed_abstracts`, `hira_faq`, `faers_12q4_25q4`, `dailymed_26_08`, `kcd`.

| Tool | Description |
| --- | --- |
| `rag_get_all_data_sources` | List identifiers and purposes of all available SQL, vector, and hybrid data sources. |
| `rag_get_data_source_detail` | Show schema, tables, columns, and metadata for a single data source. |
| `rag_sql_query` | Run SQL against structured PostgreSQL data (FAERS, DailyMed, KCD). |
| `rag_vector_query` | Semantic search over supported Qdrant collections (`pubmed_abstracts`, `hira_faq`) via vector or dense+sparse hybrid retrieval. |

---

## 4. Model API

### Chat Completions

```bash
curl "$LUNIT_FM_API_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LUNIT_FM_API_KEY" \
  -d '{
    "model": "Lunit/L2-preview",
    "messages": [
      {"role": "system", "content": "You are a careful medical assistant."},
      {"role": "user", "content": "Summarize the key findings."}
    ]
  }'
```

Tool-calling chat completions are also supported (see the official "supported parameters" reference).

---

## 5. Lunit FM L2 — How It Works

L2 is Lunit's medical-domain LLM. **It is not a general-purpose chat model.** Two properties make it
behave differently:

1. It operates in **two distinct stages** — retrieval and generation — with a **separate model call
   for each stage**.
2. It was trained with a **specific tool set** for gathering evidence during retrieval — namely the
   MCP tools provided in this hackathon.

Treating it like a general LLM will not work as expected. **Building the harness that controls this
model is the core challenge.**

### 5.1 Retrieval stage

Given a question, the model decides what evidence it needs and repeatedly calls MCP tools to search,
read, and collect relevant information. When it has enough, it outputs the items it judged relevant.

**It does not write the final answer in this stage.**

Some MCP tool results carry a `cite_uid` field. That field marks an item as citable and is how the
model refers to the item afterwards. At the end of retrieval the model reports each relevant item's
`cite_uid` — not its content.

The stage ends when the model calls `finalize_retrieval`. **This is not an MCP tool** — you define it
yourself, expose it alongside the MCP tools, and instruct the model in the system prompt to call it.

```python
from typing import Literal
from pydantic import BaseModel, Field

class CitableItem(BaseModel):
    cite_uid: str
    relevance_score: float

class CitationSelection(BaseModel):
    status: Literal["sufficient", "partial", "no_evidence"]
    items: list[CitableItem] = Field(default_factory=list)
    note: str = ""

def finalize_retrieval(
    status: Literal["sufficient", "partial", "no_evidence"],
    items: list[CitableItem],
    note: str = "",
) -> CitationSelection:
    """Submit your final citation selection and end the retrieval phase.

    Call this only:
    - once you have gathered enough evidence to answer the query
    - the query does not need any retrieval
    - you exhausted the tool call budget and must end the retrieval
    """
    return CitationSelection(status=status, items=items, note=note)
```

#### Example retrieval trajectory

> **User:** 가이드라인에 따르면 만성 신장질환 환자에게 어떤 혈압 목표를 권고하나요?
> *(What blood-pressure target do the guidelines recommend for chronic kidney disease patients?)*

```text
index_list_documents(corpus_tag="guideline", query="hypertension chronic kidney disease")
  -> 12 documents, each with node_id, title, summary

index_get_relevant_nodes(corpus_tag="guideline", query="blood pressure target CKD", node_id="0823b")
  -> 4 leaf nodes with page ranges and ancestor chains

index_get_page_content(corpus_tag="guideline", doc_id="0823b", start_page=48, end_page=52)
  -> page text for pages 48-52, carrying cite_uid "cite-3f9a1c7d2e5b8046"

finalize_retrieval(
    status="sufficient",
    items=[{"cite_uid": "cite-3f9a1c7d2e5b8046", "relevance_score": 0.95}],
    note=""
)
--- retrieval stage ends ---
```

### 5.2 Generation stage

Given the question, L2 first decides whether it can answer from memory or needs more information.
General medical questions it can answer directly; questions about specific guidelines or laws need
grounding.

For that it uses retrieval. In the generation stage you should expose **exactly one tool**:

```python
def retrieve_relevant_content(query: str):
    """Retrieve relevant content to ground your answer. Pass a single, self-contained query."""
    # Run the retrieval stage here and return the relevant information.
```

This tool runs the retrieval stage, collects the relevant information, and hands it to the model to
produce the final answer. How you wire retrieval to generation is your design decision.

#### Example generation trajectory

```text
User: 가이드라인에 따르면 만성 신장질환 환자에게 어떤 혈압 목표를 권고하나요?

L2 -> retrieve_relevant_content(
        query="recommended blood pressure target for adults with chronic kidney disease")

Tool result:
  status: sufficient

  [1]
  source_type: guideline
  url: https://example.org/guideline/0823b
  title: 2024 Clinical Practice Guideline for the Management of Hypertension
  content: In adults with chronic kidney disease, treat to a systolic blood pressure target of
           less than 120 mmHg when tolerated, ...

L2 -> "가이드라인에 따르면 만성 신장질환 성인에게 내약 가능한 경우 수축기 혈압 120 mmHg 미만을
       목표로 치료할 것을 권고합니다 [1]."
--- generation stage ends ---
```

### 5.3 Tips

- Write a **separate system prompt for each stage**; do not merge them.
- Retrieval gets the MCP tools + `finalize_retrieval`. Generation gets **only**
  `retrieve_relevant_content`.
- Retrieval queries must be **self-contained** — resolve references ("what's the dose of that drug?")
  before passing them to retrieval.
- **Cap the number of tool calls** during retrieval.
- Use `status` and `note` to pass information from retrieval to generation.

### 5.4 Constraints

- L2 is optimized for **single-turn** conversation, but the hackathon also evaluates **multi-turn**
  scenarios. Mitigating this — query rewriting, context summarization, etc. — is part of the
  challenge.
- This guide is the *recommended* usage, not a requirement. Any other system design is fine as long
  as the rules are followed.
- **The final output must be generated by Lunit's L2.** Intermediate steps may use other models or
  systems; only the final answer must come from L2.
- Beyond the provided MCP tools, you may use external data sources for which you hold an appropriate
  license.

---

## 6. Patient Simulator

An OpenAI-compatible **question generator** that plays the *user* side (patient or clinician) of a
Korean-language medical conversation.

- Your harness is the **assistant**. Simulator messages are `user` turns; your responses are
  `assistant` turns.
- Keep the full conversation history **client-side** and POST it every turn. **No session ID needed.**
- Model name: `patient-simulator-ko`.

### First question — send an empty `messages` array

```bash
curl -s "https://patient.hackathon.lunit.io/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LUNIT_FM_API_KEY" \
  -d '{
    "model": "patient-simulator-ko",
    "messages": []
  }' \
  | jq -r '.choices[0].message.content'
```

Repeat this request for another fresh question. Concurrent requests are supported;
**~14 s per call.**

### Follow-up questions — append the received question and your answer verbatim, resend all history

```bash
curl -s "https://patient.hackathon.lunit.io/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LUNIT_FM_API_KEY" \
  -d '{
    "model": "patient-simulator-ko",
    "messages": [
      {"role": "user", "content": "<received question>"},
      {"role": "assistant", "content": "<your system answer>"}
    ]
  }' \
  | jq -r '.choices[0].message.content'
```

**~8 s per follow-up.**

**Rules of thumb**

- Preserve the first question **exactly**; modifying it can break conversation continuation.
- Stop after roughly **3 turns** — longer conversations tend to repeat questions.
- `404` → start a new conversation with an empty `messages` array. `502` → retry the request.

---

## 7. Submission

### What you submit

A **containerized multi-turn conversation driver**. The evaluator sends each conversation turn to your
service; the driver must use the conversation context to orchestrate whatever models/tools your
approach needs and return the next assistant response.

### Hard constraints

- A `Dockerfile` must sit at the **repository root**, and the image must build in **under 5 minutes**
  on the evaluation VM.
- The container must start with **no manual steps** and serve on **`0.0.0.0:8000`**. Only container
  port 8000 is evaluated.
- An **OpenAI-compatible API** is required, with at minimum:
  - `GET /v1/models`
  - `POST /v1/chat/completions`

### Pre-submission checklist

| Item | Value |
| --- | --- |
| Branch | `lunit/hackathon-submission` |
| Server bind | `0.0.0.0:8000` |
| Dockerfile | `EXPOSE 8000` |

### Local build & run

```bash
docker build -t my-team-submission:local .
docker run --rm -p 8000:8000 my-team-submission:local
```

### Example Dockerfile

```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Submitting for evaluation

- Register an **immutable repository and nickname** in team settings before submitting.
- Submit the **full 40-character SHA** of `HEAD` on branch `lunit/hackathon-submission`,
  e.g. `0123456789abcdef0123456789abcdef01234567`.
- Enter the **model name** your driver uses; it is stored with the trial and reused for the official
  evaluation.
- Submissions are **closed by default** — an admin must enable submissions for your team before you
  can evaluate or submit.
- Submissions can be searched by SHA and filtered by status; the trial table shows
  *Trial · Status · SHA · Submitted · Last updated · Score · Errors · Retries*.

---

## 8. Evaluation

- During the hackathon, use the **dashboard** to test your solution against the organizers'
  **validation set** and measure benchmark performance. Use it to debug and to confirm your
  submission runs correctly on the evaluation server.
- **The last submission sent through the dashboard is treated as your final submission.**
- Final submissions are scored on a separate **HealthBench holdout test set** defined by the
  organizers.
- The same submission is also used for **expert evaluation of chat quality**.
- A **single submission** counts for both the **Benchmark** track and the **Frontier** track awards.
- Evaluation runs in a **fully isolated environment with no external access**.

---

## 9. Design Checklist for the Harness

1. **Two prompts, two stages.** Never merge the retrieval and generation system prompts.
2. **Tool exposure per stage.** Retrieval: MCP tools + `finalize_retrieval`. Generation:
   `retrieve_relevant_content` only.
3. **Self-contained retrieval queries.** Resolve pronouns/references before dispatch.
4. **Budget tool calls** in retrieval; call `finalize_retrieval` on budget exhaustion with
   `status="partial"`.
5. **Cite by `cite_uid`**, then resolve those UIDs back to content when building the generation
   context.
6. **Multi-turn handling.** Rewrite queries and/or summarize context, since L2 is single-turn tuned.
7. **Final answer must come from L2**, even if intermediate orchestration uses other models.
8. **Container contract.** Root `Dockerfile`, <5 min build, auto-start, `0.0.0.0:8000`,
   `GET /v1/models` + `POST /v1/chat/completions`.
