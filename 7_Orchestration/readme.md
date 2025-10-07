gantt
    title Agent Orchestration Pilot — 2h Timeline
    dateFormat  HH:mm
    axisFormat  %H:%M

    section Setup & Boot
    Clone repo + env keys         :milestone, m1, 10:00, 5min
    Docker compose up (services)  :active, s1, 10:05, 10min
    Health-check services         :s2, after s1, 5min

    section Channels & Graph
    Define channels (registry.yaml) :c1, 10:20, 10min
    Planner node ready               :p1, after c1, 5min
    Executor node (Tavily, MCP)      :e1, after p1, 15min
    Observer node (signals)          :o1, after e1, 5min

    section Persistence & Memory
    Mongo collections (runs,tasks) :db1, 10:55, 5min
    Qdrant collection (ensure)     :db2, after db1, 5min

    section First Processing Loop
    Fetch → Plan → Execute         :loop1, 11:05, 10min
    Upsert memory (Qdrant)         :mem1, after loop1, 5min
    Save run (Mongo)               :log1, after mem1, 2min
    Send email via MCP             :mail1, after log1, 3min
    Verify inbox & logs            :ver1, after mail1, 5min

    section Observability (Optional in Pilot)
    Wire basic Langfuse spans      :obs1, 11:35, 10min

    section Hardening (Stretch)
    Dedup/rate-limit + retries     :hard1, 11:45, 10min
    Replace fake embeddings        :hard2, after hard1, 10min
    ✅ Pilot complete (E2E)        :milestone, done1, 12:00, 0min
