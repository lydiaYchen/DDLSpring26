- [1. Important links](#1-important-links)
- [2. Course description](#2-course-description)
- [3. Learning objectives](#3-learning-objectives)
- [4. Course team](#4-course-team)
- [5. :dart: Grading policy](#5-dart-grading-policy)
  - [5.1. Homework](#51-homework)
  - [5.2. Group projects](#52-group-projects)
- [6. Detailed schedule](#6-detailed-schedule)
  - [All assignments are due before the lecture/lab day of the noted week.](#all-assignments-are-due-before-the-lecturelab-day-of-the-noted-week)

This repository contains the materials of the MSc course on **Distributed Deep Learning Systems** course running in Spring 2026 at UniNE.

## 1. Important links

- [Project description](project.md)
- [Lab and assigments](lab/README.md)

## 2. Course description

Machine learning systems are often conventionally designed for centralized processing in that they first collect data from distributed sources and then execute algorithms on a single server. Due to the limited scalability of processing large amounts of data and the long latency delay, there is a strong demand for a paradigm shift to distributed or decentralized ML systems that execute ML algorithms on multiple and, in some cases, even geographically dispersed nodes.

This course aims to teach students how to design and build distributed ML systems via paper reading, presentation, and discussion. We provide a broad overview of the design of state-of-the-art distributed ML systems, with a strong focus on the solutions' scalability, resource efficiency, data requirements, and robustness. We cover an array of methodologies and techniques that can efficiently scale ML analysis to a large number of distributed nodes against all operation conditions, e.g., system failures and malicious attacks. The specific course topics are listed below.

The course materials are based on classic and recently published papers.

## 3. Learning objectives

- To understand the design principles of distributed and federated learning systems
- To analyze distributed and federated ML in terms of the scalability and accuracy-performance tradeoff
- To understand and implement horizontal and vertical federated learning systems
- To understand and implement federated learning systems on different  models, e.g., classification and generative models
- To understand and analyze vulnerabilities and threats to federated learning systems, e.g., data poison attacks and free-rider attacks
- To design and implement defense strategies against adversarial clients in federated systems

## 4. Course team

This course is mainly taught by [Prof. Lydia Y Chen](https://lydiaychen.github.io/).
The TAs are [Abel Malan](mailto:abele.malan@unine.ch), [Giulio Segalini](mailto:giulio.segalini@unine.ch), who run the lab and grade homework.

Lydia is the responsible instructor for this course and can be reached at **lydiaychen@ieee.org**.

## 5. :dart: Grading policy

The grade of this course is determined through three components:

1. Lab assignment (30%): 3 individual lab assignments, due at the beginning of weeks 5, 10, 15.
2. Group project (70%): group project report (60%) and presentation (10%). The goal is to reproduce a paper and propose an algorithm to extend the paper.
   * There is an initial proposal in week 6
   * The interim discussion with each team is held in week 10
   * The final report and a 20-minute presentation is due in week 15 (the last one)

**All assessment items (homework assignments; project report) must be submitted via ILIAS.**

### 5.1. Homework

- Homework 1: due in week 5
- Homework 2: due in week 10
- Homework 3: due in week 15

Submissions after the grace period are not considered, and students will have the corresponding 10% of their final grade set as 0.

### 5.2. Group projects

The objective is to reproduce and improve the performance of a paper from the course list. The students must hand in a final project report in the style of a short scientific paper, stating each team member's contribution to the overall system performance. There are four milestones associated with this project. See the [project description](project.md) for more information.

- Group size: 1-2 students
- Schedule: initial proposal (week 6), interim meeting (week 10), and report + presentation/interview (week 15).

For the initial proposal, teams must submit a 1-page plan, and they will receive written feedback.
For the interim meeting, teams must submit and give a presentation on their progress, for which they will receive oral feedback.
For the final step, teams must submit a written report and give a 20-minute presentation + Q&A about their final output.

## 6. Detailed schedule

### All deadlines are due **the day before the lecture/lab of the noted week**.

| Week             | Lecture Topic                      | Lab Topic        | Deadline               |
|:-----------------|:-----------------------------------|:-----------------|:-----------------------|
| Week 1 (Feb 17)  | Introduction + Deep Learning       | SGD              |                        |
| Week 2 (Feb 24)  | Distributed Learning               | Parallelism      |                        |
| Week 3 (Mar 3)   | Horizontal Federated Learning      | HFL              |                        |
| Week 4 (Mar 10)  | Memory and Acceleration Technology | HW 1 Q&A         |                        |
| Week 5 (Mar 17)  | Vertical Federated Learning        | VFL              | HW 1                   |
| Week 6 (Mar 24)  | Federated Generative AI            | VGI              | Project Proposal       |
| Week 7 (Mar 31)  | Decentralized RL                   | Decentralized RL |                        |
| Week 8: *Easter* | *No Lecture*                       | *No Lab*         |                        |
| Week 9 (Apr 14)  | Robust Distributed Learning        | HW 2 Q&A         |                        |
| Week 10 (Apr 21) | Project Midterm                    | Project Midterm  | HW 2 + Project Midterm |
| Week 11 (Apr 28) | Attacks and Defenses I             | Inference Attack |                        |
| Week 12 (May 5)  | Attacks and Defenses II            | Training Attack  |                        |
| Week 13 (May 12) | Privacy-Enhancing Technologies     | DP               |                        |
| Week 14 (May 19) | Distributed Inference              | HW 3 Q&A         |                        |
| Week 15 (May 26) | Final Presentation                 |                  | HW 3 + Project Endterm |
