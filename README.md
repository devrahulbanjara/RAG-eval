# RAG Eval Project

A simple RAG application built on insurance policy documents, so that it can be
evaluated. The goal is not to improve the RAG's performance. The goal is to catch
it confidently giving a wrong answer — hallucinating, or leaving something out —
in a way that still looks correct to a human reading the demo.

Right now most RAG projects are judged by how they answer a few questions during a
demo, with no metrics behind it. This project replaces that with measurement: how
the system actually performs, and whether a small change to the RAG logic causes
any metric to drift — a regression.

## Metrics

Each part of the system is measured on its own, so that when an answer comes out
wrong we can figure out which part caused it.

| Level | Metric | What it evaluates |
|---|---|---|
| **Retriever** | Contextual Precision | If the retriever pulls many irrelevant chunks, then the generator can be faithful by generating the answer from those irrelevant chunks only, which makes it faithful but wrong, so we need to figure out if the retriever is actually pulling the most relevant chunks first. |
|  | Contextual Recall | To evaluate if all the chunks that were needed for the answer have been retrieved at all. A coverage answer usually needs facts from more than one section of the policy, so if even one of them is not pulled the answer cannot be complete. Contextual Precision is about the order of the chunks, and Contextual Recall is about whether something is missing. |
| **Generator** | Faithfulness | To evaluate if the generator is generating the answer based on only the context provided to it. For example, a generator including a standard 24 month waiting period in the response because most policies have one, even when this policy does not state it anywhere. |
|  | Answer Relevance | To evaluate if the generator is actually answering the question that was asked. The generator can write a paragraph that is faithful to the context and still not answer the question, like answering a question about knee replacement with a general paragraph about hospitalisation. |
| **Pipeline** | Contextual Relevancy | To evaluate how much of the retrieved context was actually useful for the question. A policy clause bundles a definition, a condition and an exception into one block, so even when the correct chunk is retrieved it still carries a lot of text that has nothing to do with the question. |
| **Application** | Correctness | To evaluate if the answer is true according to the document. This is checked against the policy wording itself, and not against the context that was given to the generator. |
|  | Completeness | To evaluate if the generator says everything in the answer and does not miss any important information that was needed for the answer to be complete. In insurance an answer that leaves out the waiting period is not a partial answer, it is a wrong answer. |

## Why insurance domain ?

Insurance policy wordings were chosen because
they produce the exact failure this project is hunting, by construction.

A policy says what it covers in a **grant of cover** clause, and then takes some of
it back in an **exclusion** clause. The two are written in almost the same words:

> **Grant of cover** — "We cover surgical treatment of the knee joint, including
> joint replacement, where the treatment requires in-patient hospitalisation."
>
> **Exclusion** — "We do not cover surgical treatment of the knee joint, including
> joint replacement, where it arises from a degenerative condition."

To a human these are opposites. To an embedding model they are nearly the same
vector — same treatment, same body part, same terms. So a retriever searching by
similarity alone can rank the exclusion above the grant, or the reverse.

Now ask *"Is knee replacement covered?"*. If the exclusion comes back first, the
generator writes an answer that is fully supported by the context it was handed
and says the opposite of what the policy means. Faithfulness scores it as fine.
Only Contextual Precision catches it.

The same document also forces the Completeness case. A correct answer to that
question needs three facts from three different sections — the grant of cover, the
24-month waiting period, and the exclusion. Contextual Recall is what checks all
three were retrieved. "Yes, knee replacement is covered" is faithful, relevant, and
still gets a customer's claim rejected.
