# Co<sup>2</sup>L: Contrastive Continual Learning

<span id="page-0-1"></span>Hyuntak Cha Jaeho Lee Jinwoo Shin Korea Advanced Institute of Science and Technology (KAIST) Daejeon, South Korea

{hyuntak.cha, jaeho-lee, jinwoos}@kaist.ac.kr

## Abstract

*Recent breakthroughs in self-supervised learning show that such algorithms learn visual representations that can be transferred better to unseen tasks than joint-training methods relying on task-specific supervision. In this paper, we found that the similar holds in the continual learning context: contrastively learned representations are more robust against the catastrophic forgetting than jointly trained representations. Based on this novel observation, we propose a rehearsal-based continual learning algorithm that focuses on continually learning and maintaining transferable representations. More specifically, the proposed scheme (1) learns representations using the contrastive learning objective, and (2) preserves learned representations using a self-supervised distillation step. We conduct extensive experimental validations under popular benchmark image classification datasets, where our method sets the new state-of-the-art performance.*

# 1. Introduction

Modern deep learning algorithms show impressive performances on the task at hand, but it is well known that they often struggle to retain their knowledge on previously learned tasks after being trained on a new one [\[30\]](#page-9-0). To mitigate such "catastrophic forgetting," prior works in the continual learning literature focus on *preserving* the previously learned knowledge using various types of information about the past task. Replay-based approaches store a small portion of past samples and rehearse the samples along with present task samples [\[33,](#page-9-1) [28,](#page-8-0) [32,](#page-9-2) [5\]](#page-8-1). Regularization-based approaches force the current model to be sufficiently close to the past model—which may be informative about the past task—in the parameter/functional space distance [\[24,](#page-8-2) [6,](#page-8-3) [37\]](#page-9-3). Expansion-based approaches allocate a unit (*e.g*., network node, sub-network) for each task and keep the unit untouched during the training for other tasks [\[36,](#page-9-4) [29\]](#page-9-5).

In this paper, instead of asking how to isolate previous knowledge from new knowledge, we draw attention to the following fundamental question:

*What type of knowledge is likely to be useful for future tasks (and thus not get forgotten), and how can we learn and preserve such knowledge?*

To demonstrate its significance, consider the simple scenario that the task at hand is to classify the given image as an apple or a banana. An easy way to solve this problem is to extract and use the color feature of the image; red means apple, and yellow means banana. The color, however, will no longer be useful if our future task is to classify another set of images as apples or strawberries; color may not be used anymore and eventually get forgotten. On the other hand, if the model had learned more complicated features, *e.g*., shape/polish/texture, the features may be re-used for future tasks and remain unforgotten. This line of thoughts suggests that forgetting does not only come from the limited access to the past experience, but also from the innately restricted access to future events; to suffer less from forgetting, learning more *transferable representations* in the first hand may be as important as carefully preserving the knowledge gained in the past.

To learn more transferable representations for continual learning, we draw inspirations from a recent advance in self-supervised learning, in particular, *contrastive learning* [\[18,](#page-8-4) [10\]](#page-8-5). Contrastive methods learn representations using the inductive bias that the prediction should be invariant to certain input transformations instead of relying on taskspecific supervisions. Despite their simplicity, such methods are known to be surprisingly effective; for ImageNet classification [\[35\]](#page-9-6), contrastively trained representations closely achieve the fully-supervised performance even without labels [\[10\]](#page-8-5) and outperform joint-trained counterparts in the supervised case [\[23\]](#page-8-6). More importantly, while the methods are originally proposed for better in-domain[1](#page-0-0) performance, recent works also show that such methods provide significant performance gains on unseen domains [\[10,](#page-8-5) [20\]](#page-8-7). Under a continual scenario, we make a similar observation: *contrastively learned representations suffer less from forgetting than the jointly trained ones* (see Section [5.2](#page-5-0) for details).

<span id="page-0-0"></span><sup>1</sup>The term 'in-domain' is used here for the setup where data distributions for representation learning and linear classifier training are the same.

<span id="page-1-1"></span>![](_page_1_Figure_0.jpeg)

<span id="page-1-0"></span>Figure 1. An overview of the Co<sup>2</sup>L framework. Mini-batch samples from the current task and the memory buffer are augmented and passed through current and past (stored at the end of the previous task) representations. Co<sup>2</sup>L minimizes the weighted sum of two losses: (1) Asymmetric SupCon loss contrasts anchor samples from the current task against the samples from other classes (Section [4.1\)](#page-3-0). (2) IRD loss measures the drift of the instance-wise similarities given by the current model from the one given by the previous model (Section [4.2\)](#page-4-0).

Unfortunately, applying this idea to continual settings is not straightforward due to at least two reasons: First, having access to informative negative samples is known to be crucial for the success of contrastive learning [\[34\]](#page-9-7), while the instantaneous demographics of negatives samples are severely restricted under standard continual setups; in classincremental learning, for instance, it is common to assume that the learner can access samples from only a small number of classes at each time step. Second, the question of how to preserve the learned *representations* not as a part of a jointly trained representation-classifier pair has not been fully answered. Indeed, recent works on representation learning for continual setups aim to learn representations accelerating future learning under a similar decoupled learning setup but lack an explicit design to preserve representations.

Contribution. To address these challenges, we propose a new rehearsal-based continual learning algorithm, coined Co2L (Contrastive Continual Learning). Unlike previous continual (representation) learning methods, we aim to *learn* and *preserve* representations continually in a decoupled representation-classifier scheme. The overview of Co<sup>2</sup>L is illustrated in Figure [1.](#page-1-0)

Our contribution under this setup is twofold:

- 1. *Contrastive learning:* We design an asymmetric version of supervised contrastive loss for learning representations under continual learning setup (Section [4.1\)](#page-3-0) and empirically show its benefits on improving the representation quality.
- 2. *Preserving representations:* We propose a novel preservation mechanism for contrastively learned representations, which works by self-distillation of instance-wise relations (Section [4.2\)](#page-4-0); to the best of our knowledge, this is a first method explicitly designed to preserve representations

without a jointly trained classifier.

We validate Co<sup>2</sup>L under various experimental scenarios encompassing task-incremental learning, domain-incremental learning, and class-incremental learning. Co<sup>2</sup>L consistently outperforms all baselines on various datasets, scenarios, and memory setups. With careful ablation studies, we also show that both components we propose (asymmetric supervised contrastive loss, instance-wise relation distillation) are essential for performance. In the ablation of distillation, we empirically show that distillation preserves learned representations and efficiently uses buffered samples, which might be the main source of consistent gains over all comparisons: distillation provides 22.40% and 10.59% relative improvements with/without buffered samples respectively on the Seq-CIFAR-10 dataset. In the ablation of asymmetric supervised contrastive loss, we quantitatively verify that the asymmetric version consistently provides performance gains over the original one on all setups, *e.g*., 8.15% relative improvement on the Seq-CIFAR-10 with buffer size 500. We also provide qualitative implication on this performance gain by visualizing learned representations, which shows our asymmetric version prevents severe drifts of learned features.

## 2. Related Work

Rehearsal-based continual learning. Continual learning methods have been developed in three major streams: using a fixed-sized buffer to replay past samples (rehearsal-based approach), regulating model parameter changes through learning (regularization-based approach), or dynamically expanding model architecture on demand (expansion-based approach). Among them, the rehearsal-based approach has shown great performance in continual learning settings, albeit with its simplicity. The idea of Experience Replay (ER

<span id="page-2-0"></span>[32]) is simply managing a fixed-sized buffer to retain a small number of samples and replaying those samples to prevent forgetting past knowledge. Following this simple setup, several methods have been proposed by expanding this framework in two aspects: which samples should be stored and how to utilize stored samples. Those works mainly focus on either regulating model updates not to contradict the learning objectives on past samples [28, 5] or selecting the most representative/forgetting-prone samples to prevent changes in past predictions [2, 7, 31]. In a purely decoupled representation learning setup, however, there are few studies on those two aspects since learning objectives of representation learning may not be directly aligned to the task-specific objectives in typical joint training schemes. In this work, we focus on utilizing buffered samples to learn representations continually on a decoupled representation-classifier learning scheme.

Representation learning in continual learning. Only a few recent studies on continual learning focus on representations models learned in two aspects: how to maintain learned representations [31] and how to learn representations accelerating future learning [22, 16]. iCaRL [31] prevents representations from being forgotten by leveraging distillation. Recent approaches [22, 16] leverage meta-learning [14] separate concerns of representation learning and classifier training and directly optimizes objectives that minimize forgetting by learning representations that accelerate future learning on meta-learning frameworks. In this work, we focus on constructing representations that suffer less from forgetting and also focus on preserving learned representations in continual learning context not as a part of joint training.

**Contrastive representation learning.** Recent progress in contrastive representation learning shows superior downstream task performance, even competitive to supervised training. Noise-contrastive estimation [17] is the seminal work that estimates the latent distribution by contrasting with artificial noises. Info-NCE [40] tries to learn representations from visual inputs by leveraging an auto-regressive model to predict the future in an unsupervised manner. Recent advances in this area stem from the use of multiple views as positive samples [38]. These core concepts have been followed by studies [10, 20, 15, 12] that have resolved practical limitations that have previously made learning difficult such as negative sample pairs, large batch size, and momentum encoders. Meanwhile, it has been shown that supervised learning can also enjoy the benefits of contrastive representation learning by simply using labels to extend the definition of positive samples [23]. In this work, we mainly leverage contrastive representation learning schemes on the continual learning setup based on our novel observation (Section 5.2).

**Knowledge distillation**. In continual learning, knowledge distillation is widely used to mitigate forgetting by distilling

past signatures to the current models [27, 31]. However, it has not been studied to design/utilize knowledge distillation for decoupled representation-classifier training in the continual learning setup. In this work, we develop novel self-distillation loss for contrastive continual learning, which is inspired by the recently proposed distillation loss [13] for contrastive learning framework.

## 3. Problem Setup and Preliminaries

In this section, we formalize the considered continual learning setup and briefly describe a recently proposed supervised contrastive learning scheme [23] that will be used as the main framework for designing Co<sup>2</sup>L (Section 4).

#### 3.1. Problem Setup: Continual Learning

We consider three popular scenarios of continual learning as categorized by [39]: task-incremental learning (Task-IL), domain-incremental learning (Domain-IL), and class-incremental learning (Class-IL).

Formally, the learner is trained on a sequence of tasks indexed by  $t \in \{1, 2, ..., T\}$ . For each task, we suppose that there is a task-specific class set  $C_t$ . For Task-IL and Class-IL,  $\{C_t\}_{t=1}^T$  are assumed to be disjoint, *i.e.*,

$$t \neq t' \Rightarrow C_t \cap C_{t'} = \emptyset,$$
 (Task/Class-IL). (1)

For Domain-IL,  $C_t$  remains the same throughout the tasks:

$$C_1 = C_2 = \dots = C_T$$
, (Domain-IL). (2)

During each task,  $n_t$  copies of training input-label pairs are independently drawn from some task-specific distribution, i.e.,  $\{(\mathbf{x}_i,y_i)\}_{i=1}^{n_t} \sim D_t$ . Here,  $\mathbf{x}$  denotes the input image, and  $y_i \in C_t$  denotes the class label belonging to the task-specific class set. For Task-IL, the learned models are assumed to have access to the task label t during the test phase; the goal is to find a predictor  $\varphi_{\theta}(\mathbf{x},t)$  parameterized by  $\theta$  such that

$$\mathcal{L}(\theta) := \sum_{t=1}^{T} \mathbb{E}_{D_t}[\ell(y, \varphi_{\theta}(x, t))], \qquad \text{(Task-IL)}$$
 (3)

is minimized for some loss function  $\ell(\cdot,\cdot)$ . For Domain-IL and Class-IL, the model cannot access the task label during the test phase; the goal is to find a predictor  $\varphi_{\theta}(\mathbf{x})$  minimizing

$$\mathcal{L}(\theta) = \sum_{t=1}^{T} \mathbb{E}_{D_t}[\ell(y, \varphi_{\theta}(x))],$$
 (Domain/Class-IL). (4)

<span id="page-3-7"></span><span id="page-3-2"></span>![](_page_3_Figure_0.jpeg)

(a) Asymmetric SupCon Loss

<span id="page-3-4"></span>(b) Instance-wise Relation Distillation Loss

Figure 2. Illustration of Asymmetric Supervised Constrastive Loss and Instance-wise Relation Distillation (IRD). (a) Given augmented mini-batch samples, asymmetric SupCon considers samples from the same class of the current task as positives. In other words, the pulling effects between anchors only exist between current task samples. (b) Given augmented mini-batch samples, the instance-wise relation is defined on the normalized projected feature vectors. The relation vectors, *i.e.*, dot products  $(\odot)$  of feature vectors, are computed from the learnable  $(\psi_{e-1}^t)$  and reference model  $(\psi_{e-1}^{t-1})$ , respectively. For E epoch training, such temperature scaled relation is distilled from the reference model to the learnable model. Note that the reference model is snapped at the end of (t-1)-th task training, and we only update the learnable model's weights using stop-gradient (denoted by sg).

#### 3.2. Preliminaries: Contrastive Learning

We now describe the SupCon (Supervised Contrastive learning) algorithm, proposed by [23]. Suppose that the classification model can be decomposed into two components

$$\varphi_{\theta} = \mathbf{w} \circ f_{\vartheta} \tag{5}$$

with parameter pairs  $\theta=(\vartheta,\mathbf{w})$ , where  $\mathbf{w}(\cdot)$  is the linear classifier and  $f_\vartheta(\cdot)$  is the representation. Without training  $\mathbf{w}$ , SupCon directly trains  $f_\vartheta$  as follows: Given a batch of N training samples  $\{(\mathbf{x}_i,y_i)\}_{i=1}^N$ , SupCon first generates an augmented batch  $\{(\tilde{\mathbf{x}}_i,\tilde{y}_i)\}_{i=1}^{2N}$  by making a two randomly augmented versions of  $\mathbf{x}_k$  as  $\tilde{\mathbf{x}}_{2k},\tilde{\mathbf{x}}_{2k-1}$ , with  $\tilde{y}_{2k}=\tilde{y}_{2k-1}=y_k$ . The samples in the augmented batch are mapped to a unit d-dimensional Euclidean sphere as

$$\mathbf{z}_i = (g \circ f)_{\psi}(\tilde{\mathbf{x}}_i),\tag{6}$$

where  $g=g_{\phi}$  denotes the projection map parametrized by  $\phi$ , and  $\psi$  denotes the concatenation of  $\vartheta$  and  $\phi$ . Now, the feature map  $(g \circ f)_{\psi}$  is trained to minimize the supervised contrastive loss

$$\mathcal{L}^{\text{sup}} = \sum_{i=1}^{2N} \frac{-1}{|\mathbf{p}_i|} \sum_{j \in \mathbf{p}_i} \log \left( \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_j / \tau)}{\sum_{k \neq i} \exp(\mathbf{z}_i \cdot \mathbf{z}_k / \tau)} \right), \quad (7)$$

where  $\tau > 0$  is some temperature hyperparameter and  $\mathfrak{p}_i$  is the index set of positive samples with respect to the anchor  $\tilde{\mathbf{x}}_i$ , defined as

$$\mathfrak{p}_i = \{ j \in \{1, \dots, 2N\} \mid j \neq i, y_j = y_i \}.$$
 (8)

In other words, the sample in  $\mathfrak{p}_i$  is either the other augmentation of the unaugmented version of  $\tilde{\mathbf{x}}_i$ , or one of the other augmented samples having the same label.

# <span id="page-3-1"></span>4. Co<sup>2</sup>L: Contrastive Continual Learning

Here, we propose a rehearsal-based contrastive continual learning scheme, coined  $Co^2L$  (Contrastive Continual Learning). At a high level,  $Co^2L$  (1) *learns* the representations with an asymmetric form of supervised contrastive loss (Section 4.1) and (2) *preserves* learned representations using self-supervised distillation (Section 4.2) in a decoupled representation-classifier training scheme. This is done by a mini-batch gradient descent based on the compound loss

<span id="page-3-5"></span>
$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{asym}}^{\text{sup}}}_{\text{(1) learning}} + \underbrace{\lambda \cdot \mathcal{L}^{\text{IRD}}}_{\text{(2) preserving}}.$$
 (9)

Here, each batch is composed of two independently augmented views of N samples (thus 2N in total), where each sample is drawn from the union of current task samples and buffered samples.

#### <span id="page-3-0"></span>4.1. Representation Learning with Asymmetric Supervised Contrastive Loss

<span id="page-3-6"></span>For learning representation continually, we use an asymmetrically modified version of the SupCon objective  $\mathcal{L}^{\text{sup}}$ . In the modified version, we only use current task samples as anchors; past task samples from the memory buffer will only be used as negative samples (see Figure 2(a)). Formally, if we let  $S \subset \{1,\dots,2N\}$  be the set of indices of current task samples in the batch, the modified supervised contrastive loss is defined as

<span id="page-3-3"></span>
$$\mathcal{L}_{\text{asym}}^{\text{sup}} = \sum_{i \in S} \frac{-1}{|\mathbf{p}_i|} \sum_{p \in \mathbf{p}_i} \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_p / \tau)}{\sum_{k \neq i} \exp(\mathbf{z}_i \cdot \mathbf{z}_k / \tau)}.$$
 (10)

The motivation behind this asymmetric design is to prevent a model from overfitting to a small number of past task <span id="page-4-3"></span>samples. It turns out that such a design indeed helps to boost the performance. In Section 5.3, we empirically observe that the asymmetric version  $\mathcal{L}^{\text{sup}}_{\text{asym}}$  outperforms the original  $\mathcal{L}^{\text{sup}}$  and generates better-spread features of buffered samples.

# <span id="page-4-0"></span>**4.2.** Instance-wise Relation Distillation (IRD) for Contrastive Continual Learning

While using the contrastive learning objective (eq. 10) readily provides a more transferable representation, one may still benefit from having an explicit mechanism to preserve the learned knowledge. Taking the inspiration from [13], we propose an instance-wise relation distillation (IRD); IRD regulates the changes in feature relation between batch samples via self-distillation (see Figure 2(b)). Formally, we define the IRD loss  $\mathcal{L}^{IRD}$  as follows: For each sample  $\tilde{\mathbf{x}}_i$  in a batch  $\mathcal{B}$ , we define the *instance-wise similarity vector* 

$$\mathbf{p}(\tilde{\mathbf{x}}_i; \psi, \kappa) = [p_{i,1}, \dots, p_{i,i-1}, p_{i,i+1}, \dots, p_{i,2N}],$$
 (11)

where  $p_{i,j}$  denotes the normalized instance-wise similarity

$$p_{i,j} = \frac{\exp\left(\mathbf{z}_i \cdot \mathbf{z}_j / \kappa\right)}{\sum_{k \neq i}^{2N} \exp\left(\mathbf{z}_i \cdot \mathbf{z}_k / \kappa\right)}$$
(12)

given the representation parameterized by  $\psi$  and the temperature hyperparameter  $\kappa$ . In other words, the instance-wise similarity vector  $\mathbf{p}(\cdot)$  is the normalized similarity of a sample to other samples in the batch.

Roughly, the IRD loss quantifies the discrepancy between the instance-wise similarities of the current representation and the past representation; the past representation is a snapshot of the model at the end of the previous task. Denoting the parameters of the past/current model as  $\psi^{\rm past}$  and  $\psi,$  the IRD loss is defined as

$$\mathcal{L}^{\text{IRD}} = \sum_{i=1}^{2N} -\mathbf{p}\left(\tilde{\mathbf{x}}_{i}; \psi^{\text{past}}, \kappa^{*}\right) \cdot \log \mathbf{p}\left(\tilde{\mathbf{x}}_{i}; \psi, \kappa\right), \quad (13)$$

where the logarithms and multiplications on the vectors denote the entrywise logarithms and multiplications. We note that we are using different temperature hyperparameters for the past and current similarity vectors; on the other hand, both  $\kappa, \kappa^*$  will remain fixed throughout the tasks.

By using fixed model weights snapped at the end of previous task training as the reference model  $\psi^{\rm past}$ , IRD distills learned representations to the current training model  $\psi$ , thereby leading to preserving learned representations. Since contrastive representation learning stems from deep metric learning, IRD achieves knowledge preservation by regulating overall *structure* changes of learned representations. Note that IRD does not regulate exact changes in feature space and does not define relation from encoder outputs like [13]. More detailed comparisons between [13] and ours are provided in the supplementary material.

#### <span id="page-4-2"></span>**Algorithm 1** Co<sup>2</sup>L: Contrastive Continual Learning

1: **Input**: Encoder parameters  $\theta$ , projector parameters  $\phi$ ,

number of tasks T, family of augmentations  $\mathcal{H}$ , a set of

```
training sets \{\{(x_i^t, y_i^t)\}\}_{t=1}^T, a set of disjoint class sets
       \{\mathcal{C}_t\}_{t=1}^T, learning rate \eta, number of epochs of t-th task
      E_t, distillation temperatures \kappa, \kappa^*, distillation power \lambda.
 2: Initialize network (g \circ f)_{\psi}(\cdot) where \psi = (\vartheta, \phi).
 3: for t = 1, \dots, T do
          Construct dataset \mathcal{D} by \mathcal{D} \leftarrow \{(x_i^t, y_i^t)\} \cup \mathcal{M}
 4:
          for e=1,\cdots,E_t do
 5:
               Draw a mini-batch \{(x_i,y_i)\}_{i=1}^N from \mathcal D
 6:
 7:
               for all k \in \{1, \cdots, N\} do
                   Draw two augmentations h \sim \mathcal{H}, h' \sim \mathcal{H}
 8:
                   Initialize anchor indices sets S \leftarrow \emptyset, I \leftarrow \emptyset
 9:
                   \tilde{x}_{2k-1} = h(x_k)
10:
11:
                   \tilde{x}_{2k} = h'(x_k)
12:
                   I \leftarrow I \cup \{2k-1, 2k\}
                   if y_k \in C_t then
13:
                       S \leftarrow S \cup \{2k-1, 2k\}
14.
15:
               end for
16:
              Compute \mathcal{L} by \mathcal{L} \leftarrow \mathcal{L}_{\text{asym}}^{\text{sup}}(I, S; \psi_{e-1}^t) (eq. 10)
17:
18:
               if t > 1 then
                   \begin{array}{l} \text{Update } \mathcal{L} \text{ by} \\ \mathcal{L} \leftarrow \mathcal{L} + \lambda \cdot \mathcal{L}^{\text{IRD}}(\psi_{E_{t-1}}^{t-1}, \psi_{e-1}^{t}, \kappa^*, \kappa) \text{ (eq. 13)} \end{array}
19:
```

Manage buffer  $\mathcal{M}$  for the number of each class samples to be same by uniform sampling.

Update  $\psi_{e-1}^t$  by  $\psi_e^t \leftarrow \psi_{e-1}^t - \eta \nabla_{\psi^t}$   $\mathcal{L}$ 

#### 4.3. Algorithm Details

20:

21:

22:

**24**: **end for** 

<span id="page-4-1"></span>Here, we give a complete picture of the overall training procedure and give additional details. The full algorithm is provided in Algorithm 1.

**Data preparation.** As the initial or new task arrives, the dataset is built as a union of current task samples and buffered samples, without any oversampling [9, 19]. The mini-batch is drawn from this dataset, where each sample is independently drawn with equal probability. To enjoy the benefits of contrastive representation learning, each sample is augmented into two views following [11]. The detailed augmentation scheme for contrastive learning is provided in the supplementary material.

**Learning new representation.** The augmented samples are forwarded to the encoder  $f_{\vartheta}$  and projection map  $g_{\phi}$  sequentially. The projection map outputs are used to compute asymmetric supervised contrastive loss (eq. 10).

**Preserving learned representation.** When a new task arrives (*i.e.*, t > 1), we compute instance-wise relation drifts

<span id="page-5-2"></span>![](_page_5_Figure_0.jpeg)

<span id="page-5-1"></span>Figure 3. Observation on two learning schemes, representation-classifier joint training and contrastive representation learning on Seq-CIFAR-10 without any design used for the continual learning settings. As new task arrives, each model is trained only with current task samples with model weights without re-initialization. After each task training ends, a new linear classifier is trained on the fixed current representation with samples observed so far (denoted by "seen objects") or all samples including ones from future tasks (denoted by "all objects"). The pair of left figures shows contrastively trained representations suffer less from forgetting than the joint trained ones. The right pair shows contrastively learned representation is much more useful to perform unseen objects classification tasks.

between reference model and the training model with IRD loss (eq. [13\)](#page-4-1). To this end, we settle the reference model as the trained model at the end of the training of (t − 1)-th task. Note that while optimizing total loss (eq. [9\)](#page-3-5), the reference model is not updated.

Buffer management. At the end of training each task, a small portion of training samples is pushed into a replay buffer. Due to its buffer size constraint, a small subset of samples from each class is pulled out of the replay buffer at the same ratio. The sample to be pushed or pulled is uniformly randomly selected for all procedures.

## <span id="page-5-3"></span>5. Experiment

#### 5.1. Experimental Setup

Learning scenarios and datasets. Following [\[39\]](#page-9-11), we conduct continual learning experiments on Task Incremental Learning (Task-IL), Class Incremental Learning (Class-IL) and Domain Incremental Learning (Domain-IL) scenarios. We conduct experiments on Seq-CIFAR-10 and Seq-Tiny-ImageNet for Task-IL and Class-IL scenarios and R-MNIST for Domain-IL scenario. Seq-CIFAR-10 is the set of splits (tasks) of the CIFAR-10 [\[25\]](#page-8-21) dataset. We split the CIFAR-10 dataset into five separate sample sets, and each sample set consists of two classes. Similarly, Seq-Tiny-ImageNet is built from Tiny-ImageNet [\[1\]](#page-8-22) by splitting 200 class samples into 10 disjoint sets of samples, each consisting of 20 classes. Seq-CIFAR-10 and Seq-Tiny-ImageNet split are given in the same order across different runs, as in [\[5\]](#page-8-1). We conduct experiments on R-MNIST [\[28\]](#page-8-0) for Domain-IL experiments. For Domain-IL scenario, R-MNIST is constructed by rotating the original MNIST [\[26\]](#page-8-23) images by a random degree in the range of [0, π). R-MNIST consists of 20 tasks, corresponding to 20 uniformly randomly chosen degrees. We note

that we treat samples from different domains with the same digit class as different classes while applying asymmetric supervised contrastive loss.

Training. We compare our contrastive continual learning algorithm with rehearsal-based continual learning baselines: ER [\[32\]](#page-9-2), iCaRL [\[31\]](#page-9-8), GEM [\[28\]](#page-8-0), A-GEM [\[8\]](#page-8-24), FDR [\[4\]](#page-8-25), GSS [\[2\]](#page-8-8), HAL [\[7\]](#page-8-9), DER [\[5\]](#page-8-1), and DER++ [\[5\]](#page-8-1). We train ResNet-18 [\[21\]](#page-8-26) on Seq-CIFAR-10 and Tiny-ImageNet, and a simple network with convolution layers on R-MNIST. For all baselines, we report performance given in [\[5\]](#page-8-1) of buffer size 200 and 500 except for R-MNIST since we choose a different architecture. More training details are provided in the supplementary material.

Evaluation protocol for Co2L. As Co<sup>2</sup>L is a representation learning scheme and not a joint representation-classifier training, we need to train a classifier additionally. For a fair comparison, we train a classifier using only the last task samples and buffered samples on top of the frozen representations learned by Co<sup>2</sup>L. To avoid the class-imbalance problems, we train a linear classifier with a class balanced sampling strategy, where first a class is selected uniformly from the set of classes, and then an instance from that class is subsequently uniformly sampled. We train a linear classifier for 100 epochs for all experiments, and we report classification test accuracy on this classifier.

## <span id="page-5-0"></span>5.2. Main Results

Validation on our key hypothesis. Before we provide results of Co<sup>2</sup>L in comparison with other methods, we first validate our running premise for method design: *Contrastive learning learns more useful representation for the future task than the joint classifier-representation supervised learning.* This premise, however, is not easy to verify under the standard continual learning setup. Indeed, the quality of

<span id="page-6-2"></span>

| Buffer | Dataset     | Seq-CIFAR-10 |            |            | Seq-Tiny-ImageNet |             |  |
|--------|-------------|--------------|------------|------------|-------------------|-------------|--|
|        | Scenario    | Class-IL     | Task-IL    | Class-IL   | Task-IL           | Domain-IL   |  |
|        | ER [32]     | 44.79±1.86   | 91.19±0.94 | 8.49±0.16  | 38.17±2.00        | 93.53±1.15  |  |
|        | GEM [28]    | 25.54±0.76   | 90.44±0.94 | -          | -                 | 89.86±1.23  |  |
|        | A-GEM [8]   | 20.04±0.34   | 83.88±1.49 | 8.07±0.08  | 22.77±0.03        | 89.03±2.76  |  |
|        | iCaRL [31]  | 49.02±3.20   | 88.99±2.13 | 7.53±0.79  | 28.19±1.47        | -           |  |
|        | FDR [4]     | 30.91±2.74   | 91.01±0.68 | 8.70±0.19  | 40.36±0.68        | 93.71±1.51  |  |
| 200    | GSS [2]     | 39.07±5.59   | 88.80±2.89 | -          | -                 | 87.10±7.23  |  |
|        | HAL [7]     | 32.36±2.70   | 82.51±3.20 | -          | -                 | 89.40±2.50  |  |
|        | DER [5]     | 61.93±1.79   | 91.40±0.92 | 11.87±0.78 | 40.22±0.67        | 96.43±0.59  |  |
|        | DER++ [5]   | 64.88±1.17   | 91.92±0.60 | 10.96±1.17 | 40.87±1.16        | 95.98±1.06  |  |
|        | Co2L (ours) | 65.57±1.37   | 93.43±0.78 | 13.88±0.40 | 42.37±0.74        | 97.90±1.92  |  |
|        | ER [32]     | 57.74±0.27   | 93.61±0.27 | 9.99±0.29  | 48.64±0.46        | 94.89±0.95  |  |
|        | GEM [28]    | 26.20±1.26   | 92.16±0.64 | -          | -                 | 92.55±0.85  |  |
|        | A-GEM [8]   | 22.67±0.57   | 89.48±1.45 | 8.06±0.04  | 25.33±0.49        | 89.04±7.01  |  |
|        | iCaRL [31]  | 47.55±3.95   | 88.22±2.62 | 9.38±1.53  | 31.55±3.27        | -           |  |
| 500    | FDR [4]     | 28.71±3.23   | 93.29±0.59 | 10.54±0.21 | 49.88±0.71        | 95.48±0.68  |  |
|        | GSS [2]     | 49.73±4.78   | 91.02±1.57 | -          | -                 | 89.38±3.12  |  |
|        | HAL [7]     | 41.79±4.46   | 84.54±2.36 | -          | -                 | 92.35±0.81  |  |
|        | DER [5]     | 70.51±1.67   | 93.40±0.39 | 17.75±1.14 | 51.78±0.88        | 97.57±1.47  |  |
|        | DER++ [5]   | 72.70±1.36   | 93.88±0.50 | 19.38±1.41 | 51.91±0.68        | 97.54±0.43  |  |
|        | Co2L (ours) | 74.26±0.77   | 95.90±0.26 | 20.12±0.42 | 53.04±0.69        | 98.65 ±0.31 |  |

<span id="page-6-1"></span>Table 1. Classification accuracies for Seq-CIFAR-10, Seq-Tiny-ImageNet and R-MNIST on rehearsal-based baselines and our algorithm. We report performance of baslines of Seq-CIFAR-10 and Seq-Tiny-ImageNet from [\[5\]](#page-8-1). '-' indicates experiments unable to run due to compatibility issues (*e.g*., iCaRL in Domain-IL) or intractable training time (*e.g*., GEM, HAL or GSS on Tiny ImageNet). All results are averaged over ten independent trials. The best performance marked as bold.

a representation is typically defined as the joint predictive performance with the best possible (linear) downstream classifier (see, *e.g*., [\[3\]](#page-8-27), and references therein), but optimal classifiers are only rarely learned under continual setups.

To circumvent this obstacle, we consider the following synthetic, yet insightful scenario: After training representations under the standard continual setup, we freeze the representations and freshly train the downstream classifier, using training data from *all tasks*. Here, the classifier trained on all observed samples so far will perform learned tasks well unless frozen representations suffer from forgetting.

As shown in the left pair of heatmaps in Figure [3,](#page-5-1) the average test accuracy on the previous tasks is surprisingly higher in *contrastive* than in *joint* (for off-diagonal parts, 21.79% vs. 66.46%). In other words, without any specific method to account for continual setup, contrastive method learns representations that suffer less from forgetting than jointly trained ones.

In the right pair of heatmaps in Figure [3,](#page-5-1) we report test accuracies of the classifiers that are trained with *all samples*, including the samples from unseen tasks. Interestingly, we observe that the average task accuracy on the unseen task is also notably higher in contrastively trained representations (rightmost heatmap) than jointly trained ones (second to right); for lower triangle parts, 32.77% vs. 62.76%. This implies that contrastive learning methods learn more highly transferable representations to future tasks, which might be the source of its robustness against forgetting.

Superiority of Co<sup>2</sup>L over baselines. As shown in Table [1,](#page-6-1)

our contrastive continual learning algorithm consistently outperforms all baselines in various scenarios, datasets, and memory sizes. Such results indicate that our algorithm successfully learns and preserves representations useful for future learning, and thus it significantly mitigates catastrophic forgetting. Moreover, such consistent gains over all comparisons show that our scheme is not limited to certain incremental learning scenarios. In what follows, we provide a more detailed analysis of our algorithm.

#### <span id="page-6-0"></span>5.3. Ablation Studies

Effectiveness of IRD. To verify the effectiveness of IRD, we perform an ablation experiment with the class-IL setup on the Seq-CIFAR-10 dataset (identical to the setup in Section [5.2\)](#page-5-0), with three additional variants of Co<sup>2</sup>L. *(a) without buffer and IRD:* We optimize using only the SupCon loss (eq. [7\)](#page-3-6); the symmetric version is identical to the asymmetric one since we do not use a replay buffer. *(b) with IRD only:* We use both (symmetric) SupCon loss and IRD loss. *(c) with replay buffer only:* We optimize the asymmetric SupCon loss (eq. [10\)](#page-3-3) without an IRD loss. Note that while we do not use buffered samples to learn representations for (a,b), we still need buffered samples to train the downstream linear classifier; for (a,b), we use 200 auxiliary buffered samples to train the classifier (as in (c) and Co<sup>2</sup>L).

As shown in Table [2,](#page-7-0) IRD brings a significant performance gain, with or without the replay buffer. With the replay buffer (rows (c,d)), we observe a 22.40% relative improvement; without the replay buffer (rows (a,b)), there is

|                             | Buffer Size | IRD | Accuracy(%)      |
|-----------------------------|-------------|-----|------------------|
| (a) w/o buffer and IRD      | 0           | X   | $53.25 \pm 1.70$ |
| (b) w/ IRD only             | 0           | ✓   | $58.89 \pm 2.61$ |
| (c) w/ buffer only          | 200         | X   | $53.57 \pm 1.03$ |
| (d) Co <sup>2</sup> L(ours) | 200         | ✓   | $65.57 \pm 1.37$ |

<span id="page-7-0"></span>Table 2. Ablation study of Instance-wise Relation Distillation (IRD). We train our model on Seq-CIFAR-10 dataset under class-IL scenario (identical to the setup in Section 5.2) with ablated Co<sup>2</sup>L. IRD brings significant gain with or without replay buffer. All results are averaged over ten independent trials.

![](_page_7_Figure_2.jpeg)

<span id="page-7-1"></span>Figure 4. Performance comparison of original and asymmetric SupCon losses on Seq-CIFAR-10 under the ideal class-IL scenario. Both settings use all past task samples. Instance-wise Relation Distillation (IRD) effectively closes the performance gap, which indicates IRD successfully retains learned representations without using past samples as positive pairs.

a 10.59% relative improvement. The former is noticeably larger than the latter; we suspect that maintaining the similarity structure of buffered samples (along with current task samples) is essential in preserving learned representations.

We also note that IRD seems to complement the asymmetric SupCon in terms of using buffered samples, leading to a performance boost. To verify this, we consider a synthetic *infinite-buffer* class-IL scenario: all past samples are available throught the training. Under this setup, we train a model with  $\mathcal{L}^{sup}$  and another with  $\mathcal{L}^{sup}_{asym}$  on Seq-CIFAR-10. As shown in Figure 4, asymmetric SupCon performs relatively poor without using IRD; under this class-balanced setup, not using past task samples as positive pairs only restricts learning. With increasing IRD power, however, the performance gap closes, indicating that IRD complements asymmetric SupCon by helping fully utilize the buffered samples. Such trend is also aligned with the results in Table 2; the performance boost from buffered samples—and thus asymmetric SupCon loss-is relatively small without using IRD. This, however, does not necessarily imply that asymmetricity does not bring any benefit, as we will observe in the following ablation study on asymmetric SupCon.

**Effectiveness of asymmetric supervised contrastive loss.** To verify the effectiveness of asymmetric supervised con-

|                                                                     | Seq-CIFAR-10             |                                          | Seq-Tiny-                | ImageNet                                 |
|---------------------------------------------------------------------|--------------------------|------------------------------------------|--------------------------|------------------------------------------|
| Buffer                                                              | 200                      | 500                                      | 200                      | 500                                      |
| $\mathcal{L}^{\text{sup}}_{\mathcal{L}^{\text{sup}}_{\text{asym}}}$ | 60.49±0.72<br>65.57±1.37 | 68.66±0.68<br><b>74.26</b> ± <b>0.77</b> | 13.51±0.48<br>13.88±0.40 | 19.68±0.62<br><b>20.12</b> ± <b>0.42</b> |

<span id="page-7-2"></span>Table 3. The effectiveness of asymmetric SupCon loss ( $\mathcal{L}_{\text{asym}}^{\text{sup}}$ ) versus the original SupCon loss ( $\mathcal{L}_{\text{sup}}^{\text{sup}}$ ), combining with the IRD loss. All results are averaged over ten independent trials.

![](_page_7_Figure_9.jpeg)

<span id="page-7-3"></span>Figure 5. Top: t-SNE visualization of features from buffered (colored) and entire (gray) training samples of Seq-CIFAR-10. Bottom: Same as Top, but non-buffered samples are in opaque color instead of gray for a clear illustration of clusters. Left: Buffered samples' features trained with original SupCon are close to the same class samples but distant from different classes. Right: Buffered samples' features trained on asymmetric SupCon are well-spread; buffered samples better represent the entire class sample population.

trastive loss, we compare two contrastive learning losses, the original SupCon and the asymmetric SupCon, as variants of Co<sup>2</sup>L with the identical settings of Section 5.2. As shown in Table 3, asymmetric SupCon consistently provides gains over all counterparts with the original SupCon.

We also compare the visualizations of encoders' outputs of buffered and entire training samples of the Seq-CIFAR-10 dataset where the encoders are trained in the ablation experiments of Table 3. As illustrated in Figure 5, buffered samples' features trained with original SupCon are close to the same class samples while ones with asymmetric SupCon are well-spread. Since the buffered samples with asymmetric SupCon better represents the entire class sample population, representations trained on asymmetric SupCon show better task performance with linear classifiers. Such qualitative results are also well aligned with the motivation of asymmetric SupCon mentioned in Section 4.1 and provide the benefits of asymmetricity.

#### 6. Conclusion

We propose a contrastive continual learning scheme for learning representations under continual learning scenarios. The proposed asymmetric form of contrastive learning loss and the instance-wise relation distillation help model learn and preserve new and past representations and show a better performance over jointly trained baselines on various learning setups. We hope that our work will serve as a good reference to how representation learning for continual learning should be designed.

## References

- <span id="page-8-22"></span>[1] Stanford 231n. Tiny ImageNet visual recognition challenge. <https://tiny-imagenet.herokuapp.com>, 2015. [6](#page-5-2)
- <span id="page-8-8"></span>[2] Rahaf Aljundi, Min Lin, Baptiste Goujaud, and Yoshua Bengio. Gradient based sample selection for online continual learning. In *Advances in Neural Information Processing Systems*, 2019. [3,](#page-2-0) [6,](#page-5-2) [7](#page-6-2)
- <span id="page-8-27"></span>[3] Sanjeev Arora, Hrishikesh Khandeparkar, Mikhail Khodak, Orestis Plevrakis, and Nikunj Saunshi. A theoretical analysis of contrastive unsupervised representation learning. In *Proceedings of the International Conference on Machine Learning*, 2019. [7](#page-6-2)
- <span id="page-8-25"></span>[4] Ari S. Benjamin, David Rolnick, and Konrad P. Kording. ¨ Measuring and regularizing networks in function space. In *International Conference on Learning Representations*, 2019. [6,](#page-5-2) [7](#page-6-2)
- <span id="page-8-1"></span>[5] Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, and Simone Calderara. Dark experience for general continual learning: a strong, simple baseline. In *Advances in Neural Information Processing Systems*, 2020. [1,](#page-0-1) [3,](#page-2-0) [6,](#page-5-2) [7](#page-6-2)
- <span id="page-8-3"></span>[6] Arslan Chaudhry, Puneet K. Dokania, Thalaiyasingam Ajanthan, and Phillip H. S. Torr. Riemannian walk for incremental learning: Understanding forgetting and intransigence. In *European Conference on Computer Vision*, 2018. [1](#page-0-1)
- <span id="page-8-9"></span>[7] Arslan Chaudhry, Albert Gordo, Puneet K. Dokania, Philip H. S. Torr, and David Lopez-Paz. Using hindsight to anchor past knowledge in continual learning. In *Association for the Advancement of Artificial Intelligence*, 2020. [3,](#page-2-0) [6,](#page-5-2) [7](#page-6-2)
- <span id="page-8-24"></span>[8] Arslan Chaudhry, Marc'Aurelio Ranzato, Marcus Rohrbach, and Mohamed Elhoseiny. Efficient lifelong learning with A-GEM. In *International Conference on Learning Representations*, 2019. [6,](#page-5-2) [7](#page-6-2)
- <span id="page-8-18"></span>[9] Nitesh V. Chawla, Kevin W. Bowyer, Lawrence O. Hall, and W. Philip Kegelmeyer. SMOTE: Synthetic minority oversampling technique. *Journal of artificial intelligence research*, 2002. [5](#page-4-3)
- <span id="page-8-5"></span>[10] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In *Proceedings of the International Conference on Machine Learning*, 2020. [1,](#page-0-1) [3](#page-2-0)
- <span id="page-8-20"></span>[11] Ting Chen, Xiaohua Zhai, Marvin Ritter, Mario Lucic, and Neil Houlsby. Self-supervised gans via auxiliary rotation loss. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2019. [5](#page-4-3)
- <span id="page-8-15"></span>[12] Xinlei Chen and Kaiming He. Exploring simple siamese representation learning, 2020. [3](#page-2-0)
- <span id="page-8-17"></span>[13] Zhiyuan Fang, Jianfeng Wang, Lijuan Wang, Lei Zhang, Yezhou Yang, and Zicheng Liu. SEED: Self-supervised distillation for visual representation. In *International Conference on Learning Representations*, 2021. [3,](#page-2-0) [5](#page-4-3)

- <span id="page-8-12"></span>[14] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Modelagnostic meta-learning for fast adaptation of deep networks. In *Proceedings of the International Conference on Machine Learning*, 2017. [3](#page-2-0)
- <span id="page-8-14"></span>[15] Jean-Bastien Grill, Florian Strub, Florent Altche, Corentin ´ Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Remi ´ Munos, and Michal Valko. Bootstrap your own latent: A new approach to self-supervised learning, 2020. [3](#page-2-0)
- <span id="page-8-11"></span>[16] Gunshi Gupta, Karmesh Yadav, and Liam Paull. Look-ahead meta learning for continual learning. In *Advances in Neural Information Processing Systems*, 2020. [3](#page-2-0)
- <span id="page-8-13"></span>[17] Michael Gutmann and Aapo Hyvarinen. Noise-contrastive ¨ estimation: A new estimation principle for unnormalized statistical models. In *Proceedings of the International Conference on Machine Learning*, 2010. [3](#page-2-0)
- <span id="page-8-4"></span>[18] Raia Hadsell, Sumit Chopra, and Yann LeCun. Dimensionality reduction by learning an invariant mapping. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2006. [1](#page-0-1)
- <span id="page-8-19"></span>[19] Hui Han, Wen-Yuan Wang, and Bing-Huan Mao. Borderline-SMOTE: A new over-sampling method in imbalanced data sets learning. In *International Conference on Intelligent Computing*, 2005. [5](#page-4-3)
- <span id="page-8-7"></span>[20] Kaiming He, Haoqi fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2020. [1,](#page-0-1) [3](#page-2-0)
- <span id="page-8-26"></span>[21] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. *arXiv preprint arXiv:1512.03385*, 2015. [6](#page-5-2)
- <span id="page-8-10"></span>[22] Khurram Javed and Martha White. Meta-learning representations for continual learning. In *Advances in Neural Information Processing Systems*, 2019. [3](#page-2-0)
- <span id="page-8-6"></span>[23] Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. Supervised contrastive learning. In *Advances in Neural Information Processing Systems*, 2020. [1,](#page-0-1) [3,](#page-2-0) [4](#page-3-7)
- <span id="page-8-2"></span>[24] James Kirkpatrick, Razvan Pascanu, Neil C. Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, and Raia Hadsell. Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences of the United States of America*, 2017. [1](#page-0-1)
- <span id="page-8-21"></span>[25] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009. [6](#page-5-2)
- <span id="page-8-23"></span>[26] Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick ´ Haffner. Gradient-based learning applied to document recognition. In *Proceedings of the IEEE*, 1998. [6](#page-5-2)
- <span id="page-8-16"></span>[27] Zhizhong Li and Derek Hoiem. Learning without forgetting. In *European Conference on Computer Vision*, 2016. [3](#page-2-0)
- <span id="page-8-0"></span>[28] David Lopez-Paz and Marc'Aurelio Ranzato. Gradient episodic memory for continual learning. In *International Conference on Learning Representations*, 2017. [1,](#page-0-1) [3,](#page-2-0) [6,](#page-5-2) [7](#page-6-2)

- <span id="page-9-5"></span>[29] Arun Mallya and Svetlana Lazebnik. PackNet: Adding multiple tasks to a single network by iterative pruning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2018. [1](#page-0-1)
- <span id="page-9-0"></span>[30] Michael McCloskey and Neal J. Cohen. Catastrophic interference in connectionist networks: The sequential learning problem. *Psychology of Learning and Motivation*, 1989. [1](#page-0-1)
- <span id="page-9-8"></span>[31] Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, and Christoph H. Lampert. icarl: Incremental classifier and representation learning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2017. [3,](#page-2-0) [6,](#page-5-2) [7](#page-6-2)
- <span id="page-9-2"></span>[32] Matthew Riemer, Ignacio Cases, Robert Ajemian, Miao Liu, Irina Rish, Yuhai Tu, and Gerald Tesauro. Learning to learn without forgetting by maximizing transfer and minimizing interference. In *International Conference on Learning Representations*, 2019. [1,](#page-0-1) [3,](#page-2-0) [6,](#page-5-2) [7](#page-6-2)
- <span id="page-9-1"></span>[33] Anthony Robins. Catastrophic forgetting, rehearsal, and pseudorehearsal. *Connection Science*, 1995. [1](#page-0-1)
- <span id="page-9-7"></span>[34] Joshua David Robinson, Ching-Yao Chuang, Suvrit Sra, and Stefanie Jegelka. Contrastive learning with hard negative samples. In *International Conference on Learning Representations*, 2021. [2](#page-1-1)
- <span id="page-9-6"></span>[35] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. ImageNet large scale visual recognition challenge. *International Journal of Computer Vision*, 2015. [1](#page-0-1)
- <span id="page-9-4"></span>[36] Andrei A. Rusu, Neil C. Rabinowitz, Guillaume Desjardins, Hubert Soyer, James Kirkpatrick, Koray Kavukcuoglu, Razvan Pascanu, and Raia Hadsell. Progressive neural networks. *arXiv preprint 1606.04671*, 2016. [1](#page-0-1)
- <span id="page-9-3"></span>[37] Joan Serra, Didac Suris, Marius Miron, and Alexandros Karatzoglou. Overcoming catastrophic forgetting with hard attention to the task. In *Proceedings of the International Conference on Machine Learning*, 2018. [1](#page-0-1)
- <span id="page-9-10"></span>[38] Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding. *arXiv preprint arXiv:1906.05849*, 2019. [3](#page-2-0)
- <span id="page-9-11"></span>[39] Gido M. van de Ven and Andreas S Tolias. Three scenarios for continual learning. *arXiv preprint arXiv:1904.07734*, 2019. [3,](#page-2-0) [6](#page-5-2)
- <span id="page-9-9"></span>[40] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Repre- ¨ sentation learning with contrastive predictive coding. *arXiv preprint arXiv:1807.03748*, 2018. [3](#page-2-0)

#### A. Training Details

#### A.1. Augmentation

We follow the data augmentation scheme introduced in [42] for representation learning and linear evaluation. We describe the default set of augmentations following the Py-Torch [46] notations in what follows.

- RandomResizedCrop. We crop Seq-CIFAR-10, Tiny-ImageNet, and R-MNIST datasets with the scale in [0.2, 1.0], [0.1, 1.0], and [0.7, 1.0], respectively. The cropped images are resized to  $32 \times 32$  for Seq-CIFAR-10,  $64 \times 64$  for Tiny-ImageNet, and  $28 \times 28$  for R-MNIST.
- RandomHorizontalFlip. Images are flipped horizontally with probability 0.5.
- ColorJitter. The maximum strengths of {brightness, contrast, saturation, hue} are {0.4, 0.4, 0.4, 0.1} with probability 0.8.
- RandomGrayScale. Images are grayscaled with probability 0.2.
- GaussianBlur. For the Tiny-ImageNet dataset, blur augmentation is applied with Gaussian kernel. Kernel size is  $7 \times 7$  and the standard deviation is randomly drawn from [0.1, 2.0]. This operation is randomly applied with probability 0.5.

#### A.2. Architecture

For Seq-CIFAR-10 and Tiny-ImageNet datasets, we use ResNet-18 (not pretrained) as a base encoder for representation learning followed by a 2-layer projection MLP which maps representations to a 128-dimensional latent space [44]. The hidden layer of projection MLP consists of 512 hidden units.

For R-MNIST, we use two convolutional layers and one fully connected layer for the base encoder. We use 20 and 50 filters with  $5\times 5$  kernel and stride 1 for two convolutional layers, respectively. Each feature map is followed by a max pooling operation with stride 2. The base encoder for R-MNIST is also followed by a 2-layer projection MLP for representation learning. The output dimensions of the last fully connected layer of encoder and the following 2-layer MLP's all hidden/output neuron sizes are equally 500.

#### A.3. Hyperparameter

The hyperparameters for Section 5 are selected by performing a grid search using the validation set consisting of randomly drawn 10% of the training samples, and chosen hyperparameters are given in Table 4. We consider following hyperparameters for  $\text{Co}^2\text{L}$ : learning rate  $(\eta)$ , batch size (bsz), temperature for asymmetric supervised contrastive learning loss  $(\tau)$ , temperatures for instance-wise relation distillation loss  $(\kappa, \kappa^*)$ , and the number of epochs of t-th task  $(E_t)$ . The hyperparameter search space for  $\text{Co}^2\text{L}$  on benchmark

| Method            | Buffer            | Parameters                                                               |  |  |  |
|-------------------|-------------------|--------------------------------------------------------------------------|--|--|--|
|                   | R-MNIST           |                                                                          |  |  |  |
| ER                | 200               | η: 0.1                                                                   |  |  |  |
|                   | 500               | $\eta$ : 0.1                                                             |  |  |  |
| GEM               | 200               | $\eta$ : 0.1, $\gamma$ : 0.5                                             |  |  |  |
|                   | 500               | $\eta$ : 0.3, $\gamma$ : 0.5                                             |  |  |  |
| A-GEM             | 200               | $\eta$ : 0.1                                                             |  |  |  |
|                   | 500               | $\eta$ : 0.1                                                             |  |  |  |
| FDR               | 200               | $\eta$ : 0.1, $\alpha$ : 1.0                                             |  |  |  |
|                   | 500               | $\eta$ : 0.2, $\alpha$ : 0.3                                             |  |  |  |
| GSS               | 200               | η: 0.2, gmbs: 128, nb: 1                                                 |  |  |  |
|                   | 500               | η: 0.2, gmbs: 128, nb: 1                                                 |  |  |  |
| HAL               | 200               | $\eta$ : 0.03, $\lambda$ :0.1, $\beta$ : 0.3, $\alpha$ :0.1              |  |  |  |
|                   | 500               | $\eta$ : 0.03, $\lambda$ :0.1, $\beta$ : 0.5, $\alpha$ :0.1              |  |  |  |
| DER               | 200               | $\eta$ : 0.1, $\alpha$ : 0.5                                             |  |  |  |
|                   | 500               | $\eta$ : 0.1, $\alpha$ : 0.5                                             |  |  |  |
| DER++             | 200               | $\eta$ : 0.1, $\alpha$ : 1.0, $\beta$ : 0.5                              |  |  |  |
|                   | 500               | $\eta$ : 0.2, $\alpha$ : 1.0, $\beta$ : 1.0                              |  |  |  |
| $Co^2L$           | 200               | 0.010.1 0.2*. 0.01 20                                                    |  |  |  |
|                   | 500               | $\eta$ : 0.01, $\tau$ :0.1, $\kappa$ : 0.2, $\kappa^*$ : 0.01, epoch: 20 |  |  |  |
|                   | Seq-CIFAR-10      |                                                                          |  |  |  |
| Co <sup>2</sup> L | 200               |                                                                          |  |  |  |
|                   | 500               | $\eta$ : 0.5, $\tau$ :0.5, $\kappa$ : 0.2, $\kappa^*$ : 0.01, epoch: 100 |  |  |  |
|                   | Seq-Tiny-ImageNet |                                                                          |  |  |  |
| Co <sup>2</sup> L | 200               |                                                                          |  |  |  |
|                   | 500               | $\eta$ : 0.1, $\tau$ :0.5, $\kappa$ : 0.1, $\kappa^*$ : 0.1, epoch: 50   |  |  |  |
|                   |                   |                                                                          |  |  |  |

<span id="page-10-0"></span>Table 4. Hyperparameters chosen in our experiments

| Dataset           | Parameter  | Values                |
|-------------------|------------|-----------------------|
|                   | η          | {0.1, 0.5, 1.0}       |
|                   | $\tau$     | $\{0.1, 0.5, 1.0\}$   |
|                   | $\kappa$   | $\{0.1, 0.2\}$        |
| Seq-CIFAR-10      | $\kappa^*$ | $\{0.01, 0.05, 0.1\}$ |
|                   | $E_0$      | {500}                 |
|                   | $E_{t>0}$  | {50, 100}             |
|                   | bsz        | {256, 512, 1024}      |
|                   | η          | {0.1, 0.5, 1.0}       |
|                   | $\tau$     | $\{0.1, 0.5, 1.0\}$   |
|                   | $\kappa$   | $\{0.1, 0.2\}$        |
| Seq-Tiny-ImageNet | $\kappa^*$ | $\{0.01, 0.05, 0.1\}$ |
|                   | $E_0$      | {500}                 |
|                   | $E_{t>0}$  | {50, 100}             |
|                   | bsz        | {256, 512, 1024}      |
|                   | $\eta$     | {0.01, 0.05, 0.1}     |
|                   | $\tau$     | $\{0.1, 0.5, 1.0\}$   |
|                   | $\kappa$   | $\{0.1, 0.2\}$        |
| R-MNIST           | $\kappa^*$ | $\{0.01, 0.05, 0.1\}$ |
|                   | $E_0$      | {100}                 |
|                   | $E_{t>0}$  | {10, 20}              |
|                   | bsz        | {256, 512, 1024}      |
|                   |            | 2 2 2 7               |

<span id="page-10-1"></span>Table 5. Hyperparameter space for Co<sup>2</sup>L

datasets are provided in Table 5. In a combined grid search for Class-IL and Task-IL, we select the best hyperparameters that achieve the highest final accuracy averaged over both settings. For R-MNIST, we conduct a grid search for all

baselines since the architecture for R-MNIST changes. We follow the hyperparameter search space (and its notations) for R-MNIST given in [\[41\]](#page-13-3). For all experiments of Co<sup>2</sup>L, we use distillation power (λ in eq. [9\)](#page-3-5) as 1.0.

#### A.4. Training Details for Co<sup>2</sup>L

For representation learning, we use a linear warmup for the first 10 epochs and decay the learning rate with the cosine decay schedule [\[45\]](#page-13-4). The learning rate scheduling is restarted at every task is introduced. We use SGD with momentum 0.9 and weight decay 0.0001 for all experiments.

For linear evaluation, we train a linear classifier for 100 epochs using SGD with momentum 0.9 and no weight decay. We decay the learning rate exponentially at 60, 75, and 90 epoch with decay rate 0.2. We use {1.0, 0.1, 1.0} learning rate for {Seq-CIFAR-10, Seq-Tiny-ImageNet, R-MNIST}.

#### **B.** Experiments on IRD Alternatives

We propose  $\text{Co}^2\text{L}$  that learns representations and preserves learned representations using  $\mathcal{L}_{\text{sup}}^{\text{asym}}$  and  $\mathcal{L}^{\text{IRD}}$ , respectively. In this section, we explore alternatives for IRD loss to preserve learned representations, and verify its effectiveness. More specifically, we consider following baselines.

**Embedding distillation**. IRD can be viewed as distilling the representations from the past self, similar to how SEED [43] distills the representation from the teacher model to the student model. However, there is a slight difference: IRD distills the instance-wise similarity of the outputs from the joint encoder-projector, where the projector is introduced for contrastive learning. SEED directly distills the output of the encoder. Specifically, for each sample  $\tilde{x}_i$  in a batch  $\mathcal{B}$ , the similarity score with respect to an encoder  $f_{\vartheta}$  is defined as:

$$\mathbf{p}\left(\tilde{\mathbf{x}}_{i}; \vartheta, \gamma\right) = \left[p_{i,1}, \dots, p_{i,2N}\right],\tag{14}$$

where  $p_{i,j}$  denotes the normalized similarity

$$p_{i,j} = \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_j / \gamma)}{\sum_{k \neq i}^{2N} \exp(\mathbf{z}_i \cdot \mathbf{z}_k / \gamma)},$$
 (15)

and  $\mathbf{z}_i$  denotes the normalized feature vector representations of  $\tilde{x}_i$  from the encoder  $f_{\vartheta}$ , *i.e.*,  $\mathbf{z}_i = f_{\vartheta}(\tilde{x}_i)/\|f_{\vartheta}(\tilde{x}_i)\|_2$ . Denoting the parameters of the teacher/student encoder and temperature as  $\vartheta^{\mathrm{T}}$ ,  $\vartheta^{\mathrm{S}}$  and  $\gamma^{\mathrm{T}}$ ,  $\gamma^{\mathrm{S}}$ , the SEED loss is defined as

$$\mathcal{L}^{\text{SEED}} = \sum_{i=1}^{2N} -\mathbf{p}\left(\tilde{\mathbf{x}}_i; \vartheta^{\text{T}}, \gamma^{\text{T}}\right) \cdot \log \mathbf{p}\left(\tilde{\mathbf{x}}_i; \vartheta^{\text{S}}, \gamma^{\text{S}}\right) \quad (16)$$

**Logit matching**. Buzzega *et al.* [41] shows matching the logit, *i.e.*, pre-softmax outputs, of the past and current model is effective for mitigating forgetting. Similarly, we replace IRD loss with the one that directly matches representation maps. Specifically, for each sample  $\tilde{x}_i$  in batch  $\mathcal{B}$ , two types of matching loss are defined as

$$\mathcal{L}_{\text{embedding}}^{\text{MSE}} = \frac{1}{2N} \sum_{i=1}^{2N} \left( f_{\vartheta}(\tilde{x}_i) - f_{\vartheta^*}(\tilde{x}_i) \right)^2 \tag{17}$$

$$\mathcal{L}_{\text{projection}}^{\text{MSE}} = \frac{1}{2N} \sum_{i=1}^{2N} \left( (g \circ f)_{\psi}(\tilde{x}_i) - (g \circ f)_{\psi^*}(\tilde{x}_i) \right)^2$$
(18)

where  $f_{\vartheta}$  is encoder and  $(g \circ f)_{\psi}$  is the feature map which maps augmented batch to an unnormalized d-dimensional Euclidean sphere. The difference between  $\mathcal{L}^{\text{MSE}}_{\text{embedding}}$  and  $\mathcal{L}^{\text{MSE}}_{\text{projection}}$  is the choice of representation maps to be matched; one defined on embedding space and the other defined on the projection space.

As shown in Table 6, we find that distilling the projector output (and thereby applying both  $\mathcal{L}^{IRD}$  and  $\mathcal{L}^{sup}_{asym}$  at the

same layer) significantly outperforms distilling at the encoder output ( $\mathcal{L}^{\text{SEED}}$ ,  $\mathcal{L}^{\text{MSE}}_{\text{embedding}}$ ) and the projection output ( $\mathcal{L}^{\text{MSE}}_{\text{projection}}$ ). Since we learn representations continually in constrastive learning schemes, where similarity is defined on a unit d-dimensional Euclidean sphere, regulating the relation drifts in the projection space can be more effective to preserve learned representations than other alternatives.

| Buffer | Objective                                                                                                                                                                                                                                                                                                                                                                                                     | Space                | Seq-CIFAR-10                                                             |                                                                      | Seq-Tiny-ImageNet                                                    |                                                              |
|--------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|--------------------------------------------------------------------------|----------------------------------------------------------------------|----------------------------------------------------------------------|--------------------------------------------------------------|
|        |                                                                                                                                                                                                                                                                                                                                                                                                               |                      | Class-IL                                                                 | Task-IL                                                              | Class-IL                                                             | Task-IL                                                      |
| 200    | $ \begin{array}{l} \mathcal{L}_{\text{sup}}^{\text{asym}} + \mathcal{L}^{\text{SEED}}\left[43\right] \\ \mathcal{L}_{\text{sup}}^{\text{asym}} + \mathcal{L}_{\text{embedding}}^{\text{MSE}} \\ \mathcal{L}_{\text{sup}}^{\text{asym}} + \mathcal{L}_{\text{projection}}^{\text{MSE}} \\ \mathcal{L}_{\text{sup}}^{\text{asym}} + \mathcal{L}_{\text{IRD}}^{\text{IRD}}\left(\text{ours}\right) \end{array} $ | Embedding Projection | $53.42\pm1.07$<br>$56.31\pm2.30$<br>$53.10\pm1.52$<br><b>65.57</b> ±1.37 | 85.79±0.91<br>86.12±0.94<br>85.05±0.95<br><b>93.43</b> ± <b>0.78</b> | 9.23±0.65<br>11.03±0.26<br>11.45±0.33<br><b>13.88</b> ± <b>0.40</b>  | 27.02±1.70<br>33.15±0.55<br>34.38±0.66<br><b>42.37</b> ±0.74 |
| 500    | $ \begin{array}{l} \mathcal{L}_{\text{sup}}^{\text{asym}} + \mathcal{L}^{\text{SEED}}\left[43\right] \\ \mathcal{L}_{\text{sup}}^{\text{asym}} + \mathcal{L}_{\text{embedding}}^{\text{MSE}} \\ \mathcal{L}_{\text{sup}}^{\text{asym}} + \mathcal{L}_{\text{projection}}^{\text{MSE}} \\ \mathcal{L}_{\text{sup}}^{\text{asym}} + \mathcal{L}_{\text{IRD}}^{\text{IRD}}\left(\text{ours}\right) \end{array} $ | Embedding Projection | 61.65±3.24<br>62.83±2.92<br>57.47±1.07<br><b>74.26</b> ± <b>0.7</b> 7    | 88.40±2.44<br>88.63±2.05<br>86.29±0.31<br><b>95.90</b> ± <b>0.26</b> | 12.04±0.40<br>14.89±0.40<br>14.73±0.39<br><b>20.12</b> ± <b>0.42</b> | 34.91±0.57<br>42.25±0.51<br>41.85±1.22<br><b>53.04</b> ±0.69 |

<span id="page-13-6"></span>Table 6. Classification accuracies for Seq-CIFAR-10 and Seq-Tiny-ImageNet on our algorithm and three alternatives. All results are averaged over ten independent trials.

#### References

- <span id="page-13-3"></span>[41] Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, and Simone Calderara. Dark experience for general continual learning: a strong, simple baseline. In *Advances in Neural Information Processing Systems*, 2020.
- <span id="page-13-0"></span>[42] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In *Proceedings of the International Conference on Machine Learning*, 2020.
- <span id="page-13-5"></span>[43] Zhiyuan Fang, Jianfeng Wang, Lijuan Wang, Lei Zhang, Yezhou Yang, and Zicheng Liu. SEED: Self-supervised distillation for visual representation. In *International Conference on Learning Representations*, 2021.
- <span id="page-13-2"></span>[44] Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. Supervised contrastive learning. In *Advances in Neural Information Processing Systems*, 2020.
- <span id="page-13-4"></span>[45] Ilya Loshchilov and Frank Hutter. SGDR: stochastic gradient descent with warm restarts. In *International Conference on Learning Representations*, 2017.
- <span id="page-13-1"></span>[46] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems.