# AI Agents Workbench

This repository is a workbench for experimenting and testing with various AI agents, both API-based and local. It is organized to facilitate rapid prototyping, integration, and evaluation of agent capabilities for general AI tasks.

## Repo Structure

Agent Workbench/                          (Main project root for AI agent experimentation)
├── index_docs.py                         (Main script for indexing documentation or data)
├── README.md
├── requirements.txt

├── agents/                               (All agent implementations, organized by type & source)

│   ├── API_based_agents/                 (Agents that interact with external APIs)

│   │   ├── code_agents_llama/            (Agents using Llama via API)
│   │   │   ├── core.ipynb                (Core logic & testing notebook)
│   │   │   ├── db_chroma/                (Chroma vector DB files for embeddings)
│   │   │   └── tools/                    (Utility scripts for tasks)
│   │   │       ├── gmail.py              (Gmail API integration)
│   │   │       └── weather.py            (Weather API integration)

│   │   ├── code_agents_smolagents/       (Agents using Smol AI models)
│   │   │   ├── gpt5.py                   (GPT-5 agent)
│   │   │   └── llama8b.py                (Llama 8B agent)

│   │   ├── lang_graph_agent/             (Workflow orchestration w/ LangGraph)
│   │   │   └── email_classifier_langgraph.ipynb   (Email classifier)

│   │   └── llama_rag/                    (Retrieval-Augmented Generation agents w/ Llama)
│   │       ├── custom_query_engine.py    (Custom query logic)
│   │       ├── default_query_engine.py   (Default query logic)
│   │       └── db_chroma/                (Chroma vector DB files)

│   └── Local_based_agents/               (Agents running locally, no API calls)

│       ├── code_agents/                  (Local code-based agents)
│       │   ├── mist.py                   (Mistral agent)
│       │   └── qwen.py                   (Qwen agent)

│       └── json_agents/                  (Agents using JSON-based logic)
│           ├── qwen-v1.py                (Qwen v1 agent)
│           └── qwen-v2.py                (Qwen v2 agent)

├── data/                                 (Data storage for tasks & indexing)
│   ├── cheat/                            (Cheat sheets & reference material)
│   └── indexed/                          (Indexed data for retrieval)

├── Qwen3/                                (Model files & configs for Qwen3, difinetely not pushed to github)

├── scripts/                              (Utility scripts for data/vector DB management)
│   ├── __init__.py
│   ├── embeddings.py                     (Embedding generation)
│   ├── read_data.py                      (Data reading/preprocessing)
│   └── vector_db.py                      (Vector DB management)

├── test/                                 (Testing scripts & modules)
│   ├── __init__.py
│   └── index_docs.py                     (Test for doc indexing)

└── tools/                                (General utilities)
    ├── search.py                         (Search utility)
    └── ...
