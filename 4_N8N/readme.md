# Zadání:
Navrhni a vytvoř agenta v n8n, který pracuje s databází, používá nástroje a odpovídá
na dotazy přes LLM.
## Forma odevzdání:
Vypracovaný úkol odevzdejte ve formě JSON souboru s definicí workflow. JSON
soubor odevzdejte v Google Classroom.


# 🧠 n8n homework (→ preperation for LangGraph / Langfuse News Summarizer)

A daily automation workflow that fetches crypto and tech news, summarizes it with an LLM, and delivers it by email.  
Originally built in **n8n**, this serves as a baseline for migration to **LangGraph** (workflow orchestration) and **Langfuse** (telemetry and tracing).

---

## ⚙️ What It Does

### 📰 Fetch RSS
- Reads **Bitcoin** and **Ethereum** RSS feeds from [Cointelegraph](https://cointelegraph.com/).

### 🧠 Summarize with LLM
- Uses the **n8n LangChain Agent** with **Google Gemini** chat model.  
- The prompt requests a **500-word brief** split into:
  - **“World news:”**
  - **“Tech news:”**
- Includes a concise **economic dynamics** comment for context.

### 🧩 Memory
- A **simple window buffer** maintains recent run context and history.

### ✉️ Deliver
- Sends the summarized output **via Gmail** as plain text.

### ⏰ Schedule
- Automatically runs **every day in the morning** (default: around **09:05–09:40**).  
- Can also be triggered manually.

---

> ✅ This workflow is ready for use in n8n as-is and provides a modular foundation for extension into LangGraph pipelines with Langfuse observability.

---

**Author:** Marek Ciklamini, Ph.D.  
**Version:** 0.1  
**License:** MIT / Apache-2.0



---

## 🚀 Overview

This workflow:
1. **Fetches RSS feeds** (Bitcoin, Ethereum, and world tech news)
2. **Summarizes content** using a **Google Gemini** LLM agent via LangChain
3. **Stores context** in a lightweight memory buffer
4. **Sends an email summary** every morning
5. **Runs automatically** on a daily schedule

The system provides a working baseline for transitioning n8n automations into modular LangGraph workflows with integrated Langfuse monitoring.

---

## 📁 Files

| File | Description |
|------|--------------|
| `n8nmemoryDBnewsRSS.json` | n8n workflow export |
| `README.md` | This documentation file |

---

## 🧩 Workflow Summary

### 🔗 Node Connections

```text
Manual Trigger  ─┐
Schedule Trigger ─┴─> AI Summary Agent ──> Output ──> Gmail (send)
       │                ▲       ▲           │
       │                │       │           └─ uses $json.output
       │        Language Model  │
       │        (Gemini Chat)   │
       └─ RSS: BTC ─────────────┤
           RSS: ETH ────────────┤
           RSS: “World News” ───┤
                         Memory Buffer Window ──┘
