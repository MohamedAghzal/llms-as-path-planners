# **Path Planning from Natural Language (PPNL) Benchmark**

PPNL is a benchmark designed to assess the spatial-temporal reasoning abilities of **Large Language Models (LLMs)** through **path planning tasks**. It evaluates an LLM’s capability to navigate grid-based environments while avoiding obstacles and adhering to constraints.

## **Benchmark Overview**

### **Task Formulation**
- Given an **N × N grid** with **obstacles (O)** and a constraint **(C)**, the LLM agent must navigate from an initial location **P₀** to a set of **goal locations (P)**.
- The agent performs a sequence of actions **A = (A₁, …, Aₜ)** to complete the task successfully.

### **Task Settings**
#### **1. Single-Goal Path Planning**
- The agent must reach a **single** goal location (**l = 1**).
- **Action space**: **Up, Down, Left, Right**.

#### **2. Multi-Goal Path Planning**
- The agent must visit **multiple** goal locations (**l > 1**).
- **Two variants**:
  - **No Constraints**: Visit all goals in any order.
  - **Constrained Ordering**: Visit specific goals before others.
- **Additional action**: **Inspect** (marks a location as visited).

![PPNL Benchmark Diagram](PPNL.pdf)

In order to generate all single goal data (using the same values as the paper), you can run the script 

``./generate_all_sg_data.sh``

In order to generate all single goal data (using the same values as the paper), first run the script for the single-goal data then run the following script 

``./generate_all_mg_data.sh``

