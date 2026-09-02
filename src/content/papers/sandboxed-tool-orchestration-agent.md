---
title: "Sandboxed Tool Orchestration: Auditing 200,000 Multi-Turn Agent Trajectories with Docker"
authors:
  - "Solstice-AI Research Team"
pubDate: 2026-09-01
tldr: "High-throughput containerized sandboxing framework evaluating real bash commands, file edits, and git operations in Manus/OpenHands traces."
abstract: "Multi-turn agent trajectories involve tool calls (bash commands, file edits, git operations) that must be executed to verify their correctness. We present AgentSandbox, a Docker-based framework that executes and audits 200,000 multi-turn agent trajectories in isolated containers, verifying that each tool call produces the expected side effects. AgentSandbox detects 15.8% of hallucinated tool calls (commands that would fail if executed), 8.3% of incorrect file edits (modifications that break existing functionality), and 6.2% of invalid git operations (commits that would fail), improving downstream agent task completion by 11.4%."
venue: "Research Technical Report"
highlightMetrics:
  - label: "Trajectories Audited"
    value: "200k"
  - label: "Tool Hallucinations"
    value: "15.8%"
  - label: "Task Completion Gain"
    value: "+11.4%"
bibtex: |
  @article{solstice2026agentSandbox,
    title={Sandboxed Tool Orchestration: Auditing 200,000 Multi-Turn Agent Trajectories with Docker},
    author={Solstice-AI Research Team},
    journal={Solstice-AI Technical Report},
    year={2026},
    url={https://solstice-ai.co/papers/sandboxed-tool-orchestration}
  }
tags:
  - "Agent Sandbox"
  - "Tool Validation"
  - "Docker"
  - "Multi-Turn"
featured: false
---

## 1. Introduction

Agent trajectories involve sequential tool calls that modify the environment (file system, git repository, running processes). These tool calls must be valid—bash commands must be syntactically correct, file edits must produce valid files, and git operations must succeed on valid repositories.

AgentSandbox executes each tool call in an isolated Docker container, capturing stdout/stderr and file system state to verify correctness.

## 2. Sandboxed Execution Framework

### 2.1 Container Architecture

Each trajectory is executed in a dedicated Docker container with:
- Ubuntu 22.04 base image
- Pre-installed tools (git, python, node, rust, go)
- Pre-populated repository (for git-based trajectories)
- Resource limits (2 CPU, 4GB RAM, 60s timeout per step)
- Network isolation (no internet access)

### 2.2 State Tracking

AgentSandbox tracks file system state at each step using:
- **Diff snapshots:** Compute file diffs before and after each tool call.
- **Git snapshots:** Commit file system state after each tool call (for git-based trajectories).
- **Process monitoring:** Track running processes and their resource usage.

### 2.3 Hallucination Detection

AgentSandbox detects hallucinated tool calls by:
1. **Command validation:** Check that bash commands are syntactically valid.
2. **Execution verification:** Run the command and check the exit code.
3. **Output comparison:** Compare actual output with the agent's claimed output.
4. **Side effect verification:** Verify that file system changes match the agent's description.

## 3. Tool Call Categories

### 3.1 Bash Commands

AgentSandbox validates bash commands by executing them in the container:
- **Syntax validation:** `bash -n` checks command syntax without execution.
- **Execution:** Run the command and capture exit code, stdout, stderr.
- **Safety checks:** Block dangerous commands (rm -rf /, fork bombs).

### 3.2 File Edits

AgentSandbox validates file edits by:
1. **Parse the edit:** Extract the file path and modification.
2. **Apply the edit:** Apply the modification to the file.
3. **Syntax check:** Run language-specific linters on the modified file.
4. **Test execution:** Run existing tests to verify the edit doesn't break functionality.

### 3.3 Git Operations

AgentSandbox validates git operations by:
1. **Clone/initialize:** Set up the git repository.
2. **Execute the operation:** Run git add, commit, push, etc.
3. **Verify the result:** Check git log, diff, and status.
4. **Conflict detection:** Detect merge conflicts and resolution failures.

## 4. Experiments

### 4.1 Setup

We audit 200,000 multi-turn agent trajectories (80k OpenHands, 60k Manus, 60k ScienceWorld) using a 64-node Docker cluster.

### 4.2 Results

| Metric | Value |
|--------|-------|
| Trajectories Audited | 200,000 |
| Total Tool Calls | 1.4M |
| Hallucinated Commands | 15.8% |
| Incorrect File Edits | 8.3% |
| Invalid Git Operations | 6.2% |
| Clusters Required | 64 nodes |
| Audit Time | 12.4 hours |

### 4.3 Downstream Impact

| Metric | Pre-Audit | Post-Audit | Gain |
|--------|-----------|------------|------|
| Task Completion | 62.3% | 73.7% | +11.4% |
| Tool Call Accuracy | 78.4% | 89.1% | +10.7% |
| Average Steps | 23.1 | 19.8 | -14.3% |

## 5. Analysis

### 5.1 Hallucination Type Distribution

| Hallucination Type | Prevalence | Detection Method |
|-------------------|-----------|-----------------|
| Non-existent commands | 6.2% | Command validation |
| Wrong arguments | 4.8% | Execution verification |
| Incorrect file paths | 3.1% | Side effect verification |
| Invalid git refs | 1.7% | Git operation verification |

### 5.2 Teacher-Specific Rates

| Teacher | Tool Hallucination Rate |
|---------|------------------------|
| GPT-5.6 Sol | 13.2% |
| Claude Fable 5 | 14.7% |
| DeepSeek V4 Pro | 11.8% |
| Manus Agents | 18.3% |

Manus agents have the highest hallucination rate, likely because they operate in more diverse environments.

## 6. Limitations

AgentSandbox requires pre-configured Docker environments for each trajectory type, which adds setup overhead. For trajectories involving external services (APIs, databases), AgentSandbox cannot fully validate the tool calls.

## 7. Conclusion

Multi-turn agent trajectories contain hallucinated tool calls that degrade student model performance. AgentSandbox detects 15.8% of hallucinated commands through containerized execution and state tracking, improving downstream task completion by 11.4%.

The key insight is that **tool call validity is verifiable through execution**, and sandboxed execution provides a deterministic, scalable mechanism for validating agent trajectories.

## References

1. OpenHands: An Open Platform for AI Software Developers. 2025.
2. Manus: A Versatile AI Agent. 2025.
3. ScienceWorld: Benchmarking Agent Reasoning. 2025.
4. SWE-Bench: Can Language Models Resolve Real-World GitHub Issues? 2024.
5. Curriculum Distillation for Long-Horizon Agent Traces. Solstice-AI, 2026.
6. Backtracking and Recovery Traces. Solstice-AI, 2026.
7. Unit-Test Driven Synthesis. Solstice-AI, 2026.
8. Docker for Reproducible Research. 2025.
9. Agent Trajectory Validation. 2025.
10. Tool Call Hallucination Detection. 2025.
