---
title: AI Glossary
description: Comprehensive glossary of Artificial Intelligence, Machine Learning, Generative AI, and Agentic AI terminology.
author: Amol C
version: 1.0
last_updated: 2026-07
---

# AI Glossary

> **The Complete Agentic AI Handbook (2026 Edition)**

This glossary provides definitions for commonly used Artificial Intelligence, Machine Learning, Generative AI, and Agentic AI terminology used throughout this handbook.

---

# A

## Accuracy

A performance metric used to evaluate machine learning classification models.

**Formula**

```
Accuracy = Correct Predictions / Total Predictions
```

**Related Chapters**

- Chapter 2 – Machine Learning

---

## Activation Function

A mathematical function that determines whether a neuron should activate.

Examples

- ReLU
- Sigmoid
- Tanh
- GELU

**Related Chapters**

- Chapter 3 – Deep Learning

---

## Agent

An autonomous software entity capable of:

- Reasoning
- Planning
- Memory
- Tool usage
- Decision making
- Executing actions

Unlike traditional chatbots, agents actively perform tasks.

**Related Chapters**

- Chapter 7 – Agentic AI

---

## Agent Loop

The continuous cycle followed by an AI agent:

```
Observe

↓

Think

↓

Plan

↓

Act

↓

Observe
```

---

## Agent Memory

Information retained by an agent across one or more interactions.

Types

- Short-term memory
- Long-term memory
- Episodic memory
- Semantic memory

---

## Agent Orchestration

The coordination of multiple AI agents working together toward a common objective.

---

## Agentic AI

AI systems capable of autonomous reasoning, planning, memory management, and tool execution to accomplish goals.

---

## AI Gateway

A centralized platform that manages AI requests, authentication, routing, quotas, logging, and governance.

---

## Artificial General Intelligence (AGI)

A theoretical AI capable of performing any intellectual task that a human can perform.

Currently does not exist.

---

## Artificial Intelligence (AI)

The field of computer science focused on creating systems capable of performing tasks requiring human intelligence.

---

## Artificial Neural Network (ANN)

A network of interconnected neurons inspired by biological brains.

Foundation of Deep Learning.

---

## Attention

A mechanism allowing transformer models to determine which tokens are most important when processing input.

---

## AutoGen

An open-source framework for building multi-agent AI systems.

---

# B

## Backpropagation

Algorithm used to train neural networks by propagating errors backward.

---

## Batch Inference

Predicting results for many records simultaneously.

Example

Daily fraud scoring.

---

## Batch Training

Training models using an entire dataset or large batches.

---

## Bias (Machine Learning)

Systematic error caused by overly simplistic assumptions.

High bias usually leads to underfitting.

---

## Business Rule Engine

Software component executing deterministic business rules independently of AI models.

---

# C

## Chain of Thought (CoT)

Prompting technique encouraging an LLM to reason through intermediate steps before generating a final answer.

---

## Chunking

Splitting large documents into smaller pieces before generating embeddings for Retrieval-Augmented Generation (RAG).

---

## Classification

A supervised learning task predicting categories.

Examples

- Spam / Not Spam
- Fraud / Genuine

---

## Context Engineering

The practice of designing and managing all information provided to an AI model—including prompts, retrieved knowledge, memory, and tool results—to improve response quality.

---

## Context Window

Maximum amount of text a language model can process in a single request.

Usually measured in tokens.

---

## Continuous Learning

Updating models with newly available data over time.

---

## CrewAI

An open-source framework for orchestrating multiple collaborating AI agents.

---

# D

## Data Drift

Changes in production data distribution over time that reduce model performance.

---

## Dataset

Collection of data used to train, validate, and test machine learning models.

---

## Decision Tree

Tree-based machine learning algorithm used for classification and regression.

---

## Deep Learning

Subset of Machine Learning using neural networks with multiple hidden layers.

---

## Dimensionality Reduction

Reducing the number of features while preserving useful information.

Examples

- PCA
- t-SNE
- UMAP

---

## Drift

Any change causing model performance degradation.

Types

- Data Drift
- Concept Drift
- Model Drift

---

# E

## Embedding

A numerical vector representing semantic meaning.

Embeddings enable semantic search and Retrieval-Augmented Generation.

---

## Embedding Model

Model that converts text, images, or other data into vector representations.

---

## Encoder

Transformer component responsible for understanding input sequences.

---

## Epoch

One complete pass through the training dataset.

---

## Evaluation Metrics

Measurements used to assess model performance.

Examples

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

## Explainability

Ability to understand why an AI model made a particular prediction.

Essential in regulated industries.

---

# F

## Feature

Input variable used by a machine learning model.

---

## Feature Engineering

Creating, transforming, and selecting features to improve model performance.

---

## Feature Store

Centralized repository for reusable machine learning features.

---

## Fine-Tuning

Further training a pre-trained foundation model using domain-specific data.

---

## Foundation Model

Large pre-trained model that can be adapted to many downstream tasks.

Examples

- GPT
- Llama
- Gemini

---

# G

## Generative AI

AI capable of generating new content such as text, images, code, audio, or video.

---

## Gradient Descent

Optimization algorithm minimizing model loss.

---

## GraphRAG

Retrieval-Augmented Generation enhanced with Knowledge Graphs.

---

## Guardrails

Policies and controls restricting undesirable AI behavior.

Examples

- Toxicity filtering
- PII masking
- Prompt injection protection

---

# H

## Hallucination

Confident but incorrect output generated by an AI model.

---

## Human-in-the-Loop (HITL)

Workflow requiring human approval before executing high-risk AI decisions.

---

## Hyperparameter

Configuration value controlling model training.

Examples

- Learning rate
- Batch size
- Epochs

---

# I

## Inference

Using a trained model to generate predictions.

---

## Instruction Tuning

Training LLMs to better follow natural language instructions.

---

# J

## JSON Mode

Model capability to generate structured JSON responses.

Frequently used in enterprise APIs.

---

# K

## Knowledge Graph

Graph-based representation of entities and relationships.

---

## Knowledge Retrieval

Fetching external information before generating responses.

---

# L

## Label

Expected output for supervised learning.

---

## LangGraph

Framework for building stateful AI agents using graph-based workflows.

---

## Large Language Model (LLM)

Large neural network trained on vast text corpora capable of understanding and generating human language.

Examples

- GPT
- Claude
- Gemini
- Llama

---

## Latency

Time taken to produce an AI response.

---

## Learning Rate

Controls the magnitude of parameter updates during training.

---

## LLMOps

Operational practices for deploying and managing Large Language Models.

---

## Loss Function

Measures prediction error during training.

---

# M

## Memory

Mechanism enabling AI agents to retain information across interactions.

---

## Model Context Protocol (MCP)

Open protocol enabling AI models to securely interact with external tools and data sources.

---

## Model Drift

Decline in prediction quality over time.

---

## Model Registry

Repository storing versioned machine learning models.

---

## MLOps

Practices for deploying, monitoring, governing, and maintaining machine learning systems.

---

## Multi-Agent System

Architecture where multiple autonomous AI agents collaborate to solve problems.

---

# N

## Neural Network

Computational model composed of interconnected neurons.

---

## Normalization

Scaling numeric values into a standard range.

---

# O

## One-Hot Encoding

Technique converting categorical values into binary vectors.

---

## Overfitting

Model memorizes training data instead of learning general patterns.

---

# P

## Parameter

Learned value inside a neural network.

---

## Planner

Agent component responsible for breaking complex goals into executable tasks.

---

## Policy

Decision-making strategy used in reinforcement learning.

---

## Precision

Percentage of positive predictions that are actually correct.

---

## Prompt

Instructions provided to an LLM.

---

## Prompt Engineering

Designing effective prompts to improve model responses.

---

## Prompt Injection

Attack attempting to manipulate AI instructions.

---

# Q

## Quantization

Reducing model precision to decrease memory and inference costs.

---

# R

## RAG (Retrieval-Augmented Generation)

Technique combining language models with external knowledge retrieval.

---

## Recall

Percentage of actual positive cases correctly identified.

---

## Reasoning

Process by which an AI model evaluates information before generating conclusions.

---

## Reflection

Agent capability to evaluate previous outputs and improve future decisions.

---

## Regression

Predicting continuous numeric values.

---

## Reinforcement Learning

Learning through rewards and penalties.

---

# S

## Semantic Search

Searching based on meaning rather than exact keywords.

---

## Semantic Kernel

Microsoft framework for enterprise AI orchestration.

---

## Similarity Search

Finding vectors closest to a query embedding.

---

## Supervised Learning

Learning from labeled datasets.

---

# T

## Temperature

Controls randomness of LLM responses.

Lower values

- More deterministic

Higher values

- More creative

---

## Token

Smallest text unit processed by an LLM.

---

## Tokenization

Converting text into tokens.

---

## Tool Calling

Ability of AI models to invoke external APIs or functions.

---

## Transformer

Deep learning architecture based on self-attention.

Foundation of modern LLMs.

---

# U

## Underfitting

Model too simple to capture data patterns.

---

## Unsupervised Learning

Learning without labeled data.

---

# V

## Validation Dataset

Dataset used to tune models during training.

---

## Vector Database

Database optimized for storing and searching embeddings.

Examples

- Pinecone
- pgvector
- Milvus
- Weaviate
- Qdrant

---

## Vector Search

Searching based on embedding similarity.

---

# W

## Workflow Agent

AI agent responsible for orchestrating business processes.

---

# X

## XGBoost

Popular gradient boosting algorithm used in structured data problems.

---

# Z

## Zero-Shot Prompting

Prompting technique where the model performs tasks without examples.

---

# Acronyms

| Acronym | Meaning                                   |
| ------- | ----------------------------------------- |
| AI      | Artificial Intelligence                   |
| AGI     | Artificial General Intelligence           |
| ANN     | Artificial Neural Network                 |
| API     | Application Programming Interface         |
| A2A     | Agent-to-Agent Protocol                   |
| CoT     | Chain of Thought                          |
| CNN     | Convolutional Neural Network              |
| DL      | Deep Learning                             |
| ETL     | Extract Transform Load                    |
| GenAI   | Generative AI                             |
| HITL    | Human in the Loop                         |
| LLM     | Large Language Model                      |
| LLMOps  | Large Language Model Operations           |
| MCP     | Model Context Protocol                    |
| ML      | Machine Learning                          |
| MLOps   | Machine Learning Operations               |
| NLP     | Natural Language Processing               |
| PCA     | Principal Component Analysis              |
| PII     | Personally Identifiable Information       |
| RAG     | Retrieval-Augmented Generation            |
| REST    | Representational State Transfer           |
| RL      | Reinforcement Learning                    |
| RNN     | Recurrent Neural Network                  |
| SDK     | Software Development Kit                  |
| SLM     | Small Language Model                      |
| SQL     | Structured Query Language                 |
| TF-IDF  | Term Frequency–Inverse Document Frequency |
| UI      | User Interface                            |
| UX      | User Experience                           |

---

# Cross Reference

This glossary is updated throughout the handbook.

As new concepts are introduced in later chapters (Spring AI, LangGraph, MCP, A2A, OpenAI Agents SDK, Azure AI Foundry, GraphRAG, etc.), additional terms will be added here.

---

**End of Glossary**
