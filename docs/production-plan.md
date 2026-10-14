# LIBRA AI — Simple Production Plan

Plan for moving LIBRA AI from a laptop to the company's OpenShift platform on AWS. (October 2026)

## 1. The system in one picture

```
 Department users        Docs chatbot (MCP)        Docs site push (2/week)
        |                        |                          |
        v                        |                          |
 +-------------- Our OpenShift project ---------------------------+
 |  +-----------+       +--------------------------------------+ |        +----------------+
 |  | Frontend  | ----> | Backend (FastAPI)                    | | -----> | Amazon Bedrock |
 |  | Streamlit |       | chat agent, ingestion, MCP endpoint, | |        | Claude, Titan  |
 |  +-----------+       | docgen in background threads         | |        | private link,  |
 |                      +--------------------------------------+ |        | no internet    |
 |                         |              |              |       |        +----------------+
 |                         v              v              v       |
 |                 Elasticsearch     PostgreSQL     Langfuse block |
 |                 (official image) (official image) (6 containers)|
 |                                                                 |
 |  OpenShift built-in tools: restart on crash, logs, monitoring, alerts
 +-----------------------------------------------------------------+
```

LIBRA AI moves from my laptop to the company's OpenShift platform on AWS. I follow the platform team's pattern: every piece runs as a container, and every database runs in our own container, not a cloud service. There is **one frontend**, **one backend** that does all the work, **one Elasticsearch**, **one Postgres**, and **Langfuse** for watching the LLM. The only AI models are on Amazon Bedrock, reached through a private connection, so company data never goes on the internet.

## 2. Five words you need

| Word | Plain meaning |
| --- | --- |
| **Container** | One program packed with everything it needs to run, built from an *image* (the recipe). Our backend is one container, Postgres is another. |
| **Pod** | OpenShift's name for a running container. "Restart the pod" means "restart the program". |
| **Persistent volume** | A disk that OpenShift attaches to a container. If the container restarts, the disk and its data are still there. Every database gets one. |
| **Health check** | A small web address (`/health`) that OpenShift calls every few seconds. If it stops answering, OpenShift restarts the container on its own. |
| **Route** | The web address OpenShift gives to the frontend and the backend so people (and the docs chatbot) can reach them from the company network. |

## 3. The numbers

The load is small. That is why one copy of each container is enough.

| Usage | Volume |
| --- | --- |
| Chat search | 5 users a day × 4 questions = **20 questions a day** |
| Document uploads | **3 documents a day**, about 10 pages each |
| Docs-site chatbot (through MCP) | 10 people × 10 questions = **100 questions a day** |
| Docs-site updates | **2 pushes a week**, only the changed pages are re-ingested |
| Docgen | **2 sessions a week**, each with 5 full runs = 10 runs a week |
| Most requests at the same moment | **5** (requirement) |

**Resources we ask the platform team for:** about **5 CPUs, 16 GB of memory, and 120 GB of disk** (including a backup disk).

**Bedrock cost:** roughly **$150–200 a month** (rough estimate from list prices). The docs chatbot is the biggest part because it has the most questions.

## 4. The 10 containers

| # | Container | Image | What it does | Disk |
| --- | --- | --- | --- | --- |
| 1 | **Frontend** | our image | Streamlit: the chat page and the docgen page | none |
| 2 | **Backend** | our image | FastAPI: chat agent, document ingestion, docgen in the background, and the MCP endpoint for the docs chatbot | small disk for generated docgen files |
| 3 | **Elasticsearch** | official Elasticsearch image | Stores the document chunks; does both vector search and keyword search | 30 GB |
| 4 | **PostgreSQL** | official Postgres image | Conversations, document list, feedback, docgen runs and the LangGraph checkpoints (saved steps of a docgen run) | 10 GB |
| 5–10 | **Langfuse block**: web, worker, Postgres, ClickHouse, Redis, MinIO | official Langfuse images, from Langfuse's own docker-compose file | Records every LLM call, tool call and search so we can see what the agent did | 30 GB |

Plus one small **nightly backup job**: it copies the Postgres data and an Elasticsearch snapshot onto a separate backup disk.

**Why Langfuse is 6 containers and not 1.** The current Langfuse version is shipped as 6 containers: the web page, a worker that processes traces in the background, Postgres for users and settings, **ClickHouse** (a database built to store millions of trace lines and count them fast), Redis (a small memory cache) and MinIO (a file store). We do not build or tune them. We copy the official compose file as it is and treat the 6 as one block.

## 5. The main decisions, in plain English

**1. One copy of each container.**
Why: at most 5 requests at once, about 120 a day. One copy handles that easily. If it crashes, OpenShift restarts it in about a minute, and the requirement accepts a few minutes of downtime.
What we gave up: zero-downtime on a crash. If usage grows, we change "1 copy" to "2 copies" in the configuration.

**2. One backend does everything, docgen runs in the background.**
Why: a docgen run can take minutes and waits for the user's answers, so it cannot run inside one web request. The backend starts it in a background thread and returns at once; the page checks progress every few seconds. This is how the code already works, so no new system (no Kafka, no job queue).
What we gave up: if the backend restarts during a run, the run stops. Because LangGraph saved every step in Postgres, the user clicks *resume* and it continues from the last saved step.

**3. Databases are plain containers from official images.**
Why: the platform team's pattern is "your own containers, your own databases". The official Elasticsearch and Postgres images are the simplest way to follow it. Each one gets a persistent volume, so data survives restarts.
What we gave up: no automatic second copy of the database. We rely on the nightly backup instead.

**4. One Postgres for conversations and checkpoints, instead of SQLite files.**
Why: SQLite is a single file meant for one program on one laptop. Postgres is a real server that the backend, the backup job and later a second backend can all use safely. LangGraph already has a ready-made Postgres checkpointer, so the change is small.
What we gave up: a small code change (SQLite → Postgres) before go-live.

**5. Bedrock is the only AI, reached privately.**
Why: company data must not leave AWS. The backend calls Bedrock (Claude for answers, Titan for embeddings) through a private AWS connection (a *VPC endpoint*: a private door from our network to Bedrock that never goes through the internet). The platform team gives the backend an AWS role, so there is no password or API key stored anywhere.
What we gave up: a different embedding model than on my laptop, so all documents are re-embedded once at go-live.

**6. The MCP server is part of the backend.**
Why: the docs chatbot needs to call our search. Adding an `/mcp` endpoint to the existing backend means no extra container, and the same access rules apply: the chatbot only sees public docs-site pages.

**7. Docs-site pages are re-ingested on each push.**
Why: the docs website's pipeline calls our backend on each push (2 a week): it sends changed pages to `/ingest` and asks to delete removed pages.

**8. Login through the company SSO.**
Why: the backend already supports company login tokens (OIDC). In production it is switched on, so every search only returns documents the person is allowed to see.

## 6. When something breaks

OpenShift restarts any crashed container on its own. Data lives on disks that survive restarts, and a nightly backup covers the rare case where a disk is lost.

| What breaks | What happens | Users notice | Back in |
| --- | --- | --- | --- |
| Backend or frontend crashes | OpenShift sees it and restarts it | Requests fail for about a minute; conversations are saved in Postgres, nothing lost | ~1 min |
| Backend freezes (still running, not answering) | Health check fails 3 times, OpenShift restarts it | Same as above | ~1 min |
| Docgen run interrupted by a restart | Run stops at its last saved step | User clicks *resume* | user's choice |
| Elasticsearch or Postgres crashes | Restarted on the same disk, data intact | Search or history unavailable meanwhile | ~2 min |
| The server under our containers dies | OpenShift moves our containers to another server and re-attaches the disks | Tool unavailable for a few minutes | ~10 min |
| A disk is lost or data corrupted | We restore last night's backup (a written procedure) | Changes since the last backup are lost | ~1 hour |
| Bedrock is slow or refuses requests | The backend waits and retries | Slower answers | depends on AWS |

This fits the requirement: minutes of downtime are fine, a whole day never.

## 7. Watching it run

Two tools, each with one job.

**OpenShift's built-in tools watch the containers.** In the OpenShift web console:

- **Logs**: every container prints its logs; OpenShift collects them so I can search them by container and time.
- **Monitoring**: CPU, memory and restart count of each container, shown as graphs.
- **Alerts**: an email or chat message if a container keeps crashing, a disk is almost full, or the backup did not run.

**Langfuse watches the AI.** For each question it shows what the agent did: which searches it ran, which chunks came back, the prompt sent to Bedrock, the answer, the time taken and the tokens used. It is also where test scores are stored (section 9).

The simple rule: *"Is the tool running?"* → OpenShift. *"Why was this answer wrong?"* → Langfuse.

## 8. Releasing a new version

I follow the platform team's deployment pattern. In short:

1. Push code to Git. The pipeline runs the tests (with a real Elasticsearch) and builds one image for frontend and backend.
2. The image goes to the company's image registry.
3. OpenShift starts the new container, waits until its health check says "ready", then stops the old one. Users see no interruption.
4. If something is wrong, I redeploy the previous image. That is the rollback.

Settings (model names, database addresses) come from OpenShift configuration, and passwords from OpenShift *secrets*. Nothing secret is inside the image or in Git.

## 9. Three production stories, and the tests they created

Three incident scenarios from the run phase, each with the same shape: what happened, how we saw it, why, the fix, and the new test.

### Story 1: docgen asked users for help it did not need

- **What happened.** During a docgen session, the run kept stopping with "I need your help" on questions the documents clearly answered. Chat answers were slow at the same time.
- **How we saw it.** OpenShift logs showed many Bedrock "too many requests" errors. In Langfuse, the answer-checking step had *crashed*, not *rejected* the answer.
- **Why.** A docgen run sends many large requests in a row and hit Bedrock's request limit. Our code counted each Bedrock error as "bad answer, try again", so after a few errors it asked the human.
- **Fix.** Bedrock errors now mean "wait a bit and retry the same call"; they no longer count as a bad answer. We also asked AWS to raise our Bedrock limit.
- **New test.** A fake LLM that fails 30% of the time on purpose. The docgen test must still finish without asking a human because of those failures.

### Story 2: the docs chatbot recommended a retired tool

- **What happened.** The chatbot told people to use a tool the department had stopped using.
- **How we saw it.** More thumbs-down on chatbot answers. In Langfuse, the bad answers quoted a docs page that had been deleted from the website.
- **Why.** Our docs update only added and changed pages. Deleted or moved pages stayed in Elasticsearch forever.
- **Fix.** On each push we also delete removed pages. Once a week, a check compares our indexed pages with the website's page list and removes leftovers.
- **New test (freshness test).** After each docs push: ask one question about each changed page. The answer must quote the new version, and no deleted page may remain in the index.

### Story 3: exact error codes were not found

- **What happened.** Searching an exact code like `ERR-4012` returned documents about other errors.
- **How we saw it.** A weekly look at thumbs-down questions: most contained a code. In Langfuse, the right document was ranked 10th to 20th, too low for the agent to use it.
- **Why.** Embeddings see `ERR-4012` and `ERR-4021` as nearly the same thing. Keyword search split the code at the hyphen into `ERR` and `4012`, which are too common to help.
- **Fix.** An extra "exact match" field in Elasticsearch that keeps codes whole. We built the new index next to the old one, tested it, then switched (the project already supports this).
- **New test.** Real failed user questions are added to the test set every week, grouped by type: codes, general questions, tables. We now check the score of each group, not only the overall average, so one weak group can't hide behind a good average.

## 10. Interview questions, short answers

**How did you deploy it?**
On the company's OpenShift on AWS, following the platform team's pattern: 10 containers. Frontend, backend, Elasticsearch, Postgres, and Langfuse's 6 official containers.

**Why no managed database?**
Company rule. We run the official Elasticsearch and Postgres images ourselves, each with a disk that survives restarts, plus a nightly backup.

**What happens if the backend crashes?**
OpenShift restarts it in about a minute. Conversations and docgen progress are saved in Postgres, so nothing is lost. A docgen run continues from its last saved step.

**Why only one copy of each container?**
At most 5 requests at once and about 120 a day. One copy is enough, and the requirement accepts minutes of downtime. Going to two copies is a configuration change.

**How does docgen run without blocking the backend?**
It runs in a background thread. The backend answers immediately, the page checks progress every few seconds, and LangGraph saves every step in Postgres.

**Why Postgres and not SQLite?**
SQLite is a single file for one program on one machine. Postgres is a real server that the backend, the backup job and a future second backend can all use safely.

**How do you keep company data inside AWS?**
The only AI is Bedrock, reached through a private AWS connection, never the internet. Our containers have no internet access, and no API key is stored: the platform gives the backend an AWS role.

**How is the docs chatbot connected?**
Through an MCP endpoint on our backend. It logs in as its own user that can only see public docs-site pages, so it can never reach confidential documents.

**How do you monitor it?**
OpenShift's built-in logs, monitoring and alerts for the containers. Langfuse for the AI: every search, prompt, answer, time and cost.

**What went wrong in production, and what did you learn?**
Bedrock limits during docgen, stale docs pages, and exact codes not found. Each one became a new test: a fake failing LLM, a freshness test after each docs push, and test scores split by question type.
