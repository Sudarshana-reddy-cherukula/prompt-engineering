# Week 02 — Baseline vs. Enhanced Prompts

## 1. Aim
To understand the difference between baseline and enhanced prompts and evaluate how prompt enhancement techniques improve the relevance, accuracy, conciseness, and formatting of LLM responses.

## 2. Objectives
- Understand baseline and enhanced prompts.
- Compare responses generated using different prompt styles.
- Explore role prompting and few-shot prompting.
- Learn output formatting, constraints, and context injection.
- Evaluate the quality of LLM responses using defined criteria.

## 3. Introduction
Prompt engineering involves designing instructions to guide a Large Language Model (LLM) toward producing useful and relevant responses.

A baseline prompt provides minimal instructions, whereas an enhanced prompt includes additional guidance, structure, constraints, examples, or contextual information.

Enhanced prompts generally produce more predictable and useful outputs.

## 4. Types of Prompts

### 4.1 Baseline Prompt
A baseline prompt contains a minimal instruction, question, or command.

**Example:**
"Tell me about climate change."

**Characteristics:**
- Provides minimal guidance.
- May produce vague or lengthy answers.
- May generate responses that do not fully match the user's requirements.

### 4.2 Enhanced Prompt
An enhanced prompt includes additional instructions, constraints, or context.

**Example:**
"You are a climate scientist. Explain the main cause of climate change in three bullet points, each less than 20 words."

**Characteristics:**
- Provides clear instructions.
- Defines the expected output structure.
- Establishes length and content constraints.
- Produces more predictable responses.

## 5. Prompt Enhancement Techniques

### 5.1 Role Prompting
Assigns a specific role or persona to the LLM.

**Example:**
"Act as a teacher and explain machine learning to a beginner."

### 5.2 Few-Shot Prompting
Provides examples of desired input-output pairs to guide the model.

**Example:**
Input: Happy
Output: Positive

Input: Sad
Output: Negative

Input: Excellent
Output: ?

Expected output: Positive

### 5.3 Output Formatting
Specifies the required format of the response.

Common formats include:
- JSON
- Bullet points
- Tables

**Example:**
"Explain artificial intelligence in a table with two columns: Concept and Description."

### 5.4 Constraints
Defines limits on length, style, or content.

**Example:**
"Explain cloud computing in exactly three bullet points."

### 5.5 Context Injection
Provides relevant background information to help the model generate an appropriate response.

**Example:**
"Assume the reader is a beginner with no programming experience. Explain Python variables using simple examples."

## 6. Experimental Procedure

1. Select an LLM for the experiment.
2. Enter a baseline prompt.
3. Record the generated response.
4. Create an enhanced version of the same prompt.
5. Include suitable roles, constraints, examples, formatting, or context.
6. Submit the enhanced prompt to the same LLM.
7. Record the enhanced response.
8. Compare both responses using the evaluation criteria.
9. Document the observations and result.

## 7. Experimental Example

### Baseline Prompt
"Tell me about climate change."

### Enhanced Prompt
"You are a climate scientist. Explain the main cause of climate change in three bullet points, each less than 20 words."

### Comparison

| Evaluation Criterion | Baseline Prompt | Enhanced Prompt |
|---|---|---|
| Relevance | May include broad or unrelated details | More focused on the requested topic |
| Conciseness | Length is not specified | Length is explicitly constrained |
| Accuracy | Requires factual verification | Requires factual verification |
| Format Adherence | No specific format required | Requires three bullet points |

Note: Actual results depend on the model and its generated responses. The comparison describes the expected differences between the prompt designs.

## 8. Evaluation Criteria

### 8.1 Relevance
Checks whether the response directly answers the question.

### 8.2 Conciseness
Checks whether the response avoids unnecessary information.

### 8.3 Accuracy
Checks whether the response contains factually correct information.

### 8.4 Format Adherence
Checks whether the response follows the specified structure and formatting instructions.

## 9. Observations
- Baseline prompts provide minimal guidance.
- Enhanced prompts specify the expected response more clearly.
- Role prompting establishes a desired perspective.
- Few-shot prompting uses examples to guide output.
- Output formatting and constraints improve consistency.
- Context injection supplies relevant background information.

## 10. Result
Baseline and enhanced prompts were studied and compared. Different prompt enhancement techniques were examined, and the responses were evaluated for relevance, conciseness, accuracy, and format adherence.

## 11. Conclusion
Enhanced prompts provide clearer instructions than baseline prompts by incorporating roles, examples, formatting requirements, constraints, and relevant context. These techniques can improve the predictability and usefulness of LLM responses.