### 第五届“达观杯” 基于大规模预训练模型的风险事件标签识别赛后总结

##### 比赛链接

https://www.datafountain.cn/competitions/512/ranking?isRedance=0&sch=1804&stage=A

本项目是该比赛的代码和思路整理，许多代码可以多次复用，为了日后面试或项目需要，从而整理出来

#### 数据处理

```
# train dataset and test dataset
/datasets

# train dataset label1 and label2 freq count
/datasets/label_vocab_process.py

# train dataset sentences length count
/datasets/sample_length_stats.py

# split train dataset
/datasets/split_datasets.py
```

#### 词频对应

```
# get word freq count from corpus
src/bert_models/vocak_process.py

# transform train dataset to plaintext according to word freq count
src/bert_models/get_vocab_mapping.py
```

#### 思路整理

比赛过程中，幸得老师的指导，开拓了许多思路，然而因为比赛的时间比较紧，没有来得及所有思路都尝试一遍，下面仅仅列出思路以及paper及相关的代码片段。

- pooling层变化

  **pool** 其实就是bert模型概念中的cls，即对于一个句子他由多个word组成的，然后每个word具有其对应的embedding，那么cls之前的一层的对应的维度就应该为：

  **batch_size * sentence_length * embedding_length**

  那个如果将一个句子用一个向量来表示了（cls），那么就需要使用pool层，将embedding的dim压缩到1

  ```
  # 根据无监督数据训练word2vec
  /src/classic_models/word2vec/train_word2vec.py
  
  # max-pool
  /src/classic_models/modules/max_pool.py
  
  # self_attn_pool
  # reference paper: https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf
  # 论文中提出了一种计算word embedding到sentence embedding的思想：
  # 其实就是随机初始化一个向量，把该向量做为一个标杆同word embedding做点积得到这个字同这个标杆的相关性，然后利用这个相关性做
  # softmax得到每个word embedding所对应的系数，然后利用每个word embedding的系数得到sentence embedding
  /src/classic_model/modules/self_attn_pool.py
  
  # dynamic_routing_pool
  # reference paper: https://arxiv.org/pdf/1806.01501.pdf
  # 论文中提出了一种胶囊网络，这里主要借鉴的是胶囊网络的pooling，即为dynamic_pooling
  # dynamic_pooling中的主要思想是将word embedding通过多个全联接层，得到对应的多个映射的之后的表示，然后通过特定系数的计算
  # 方式得到这几个映射的系数，最终得到sentence embedding
  /src/classic_model/modules/chid_dynamic_routing.py
  ```

- Bert改进模型使用

  Bert的原始模型中是先切词，使用MLM来mask句子中的一个单词，但是这样会有一个问题就是单纯mask句子中的一个单词，当模型训练的时候周围的单词，会给mask的单词很多提示，这样就会弱化MLM语言模型的锻炼

  **WWM**是Whole Word Masking Models，即mask all of the tokens correspond to a word，剩下的没有什么变化

  比赛中尝试了很多WWM模型，其中效果最好的模型是**nezha-large-bert**

- 样本不均衡问题解决

  统计了train dataset中的标签的频率，可以明显看出标签存在的明显的不均衡的情况

  - Class weights/ Sample weights

    - Class weights

      对于ce loss添加系数，来控制少数类别的样本对于总loss的贡献

      ```python
      loss_fct = nn.CrossEntropyLoss(weight=weight)
      ```

    - Sample weights

      是关于如何将train dataset中不均衡的数据放入到batch中，其中抽样的方法一共有3种：

      - Instance-balanced sampling：每个图片被抽到的概率相同，即我们一般使用的抽样的方式

      - Class balanced sampling: 每个类别被抽到的概率相同，即先确定类别，然后从对应的类别中的获取样本

      - Re-sampling: 假设$q \in(0,1)$$， $$p_i=\frac{n^q_i}{\sum_{j=1}^Cn_j^q}$$，其中$$C$$表示数据集的类别的数目，$$n_i$$是类别的$$i$$的样本的数量。

        ```python
        # 对应main中的train的设定
        parser.add_argument("--use_weighted_sampler", action="store_true", help="use weighted sampler")                        
        ```

  - Focal loss

    Reference paper: https://arxiv.org/pdf/1708.02002.pdf

    Reference: https://zhuanlan.zhihu.com/p/32423092

    专门针对于样本不均衡的情况下的loss，其主要的思想就是在二分类问题中，当正负样本不均衡的时候，如果正样本的数目过多，如果是普通的交叉熵损失函数，那边模型在训练过程只会越来越关注正样本的分类是否正确，所以focal loss设计的理念就是增大如上情况的负样本分类正确与否对于模型的贡献

  - dice loss

    **to do**

- Muliti-sample dropout

  Reference paper: https://arxiv.org/pdf/1905.09788.pdf

  该方法是提分效果最明显，但是思想很简单。original dropout是作用在pooling层之后，分类器之前，相当于把对于分类器的输入按照一定的概率dropout掉，而multi-Sample Dropout则是增加多个dropout，这样相当于扩大了输入到分类器中的样本的数量，其实就是数据扩充。

  在比赛中的使用则是对于bert预训练模型pooling的输出通过多个dropout，将通过dropout的结果求和之后（根据参数**ms-average**决定是否做平均）作为logits来计算loss。

- Contrastive learning

  Contrastive learning是之前我并没有接触过的learning的方法，本次比赛中通过contrastive learning来提升sentence embedding。

  我们知道深度学习的本质就是表示学习，既然是表示学习，那么核心就是学习一个映射函数$f$，把样本$x$编码成其表示$f(x)$，Contrastive learning的核心就是使得$f$满足公式：

  $$s(f(x),f(x^+)) >> s(f(x), f(x^-))$$

  公式中的$x^+$就是和$x$类似的样本，$x^-$就是和$x$不相似的样本，$S(\cdot, \cdot)$这是一个度量样本之间相似程度的函数，一个比较典型的score函数就是向量内积。即优化下面这一期望：

  $$E=[-log(\frac{e^{f(x)^Tf(x^+)}}{e^{f(x)^Tf(x^+)}+e^{f(x)^Tf(x^-)}})]$$

  如果对于一个$x$，有1个正例和N-1个负例，那么这个loss就可以看做一个N分类问题，那么这个loss就可以看做一个2分类问题，实际就是一个交叉熵。

  Contrastive learning优化策略：

  - 增大batch
  - 提升encoder模型能力
  - 提升数据增加技术

  reference paper: [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722)

  ![image-20210927232808402](/Users/zhuyue/Code/Python/data_grand_competition_review/contrastive_learning.png)

  (a) 图是普通的对比学习流程，输入正负样本计算contrastive loss进行学习，此时负样例也会为loss产生贡献，因此也就会有梯度回传给对应的负样本的encoder，那么这样在实现的时候，样本的数量必然会受到batch size的限制，从而影响到表示的质量。

  所以Contrastive learning的优化策略其中之一就是**增大batch size**，从而出现(b)图memory bank，将所有的样本存起来，然后每次随机采样，这样就可以认为负例的样本理论上可以达到所有样本的数量。具体的做法就是在train的过程中每一次epoch，encoder所有的负样本，但是这样很吃内存。

  - 生成式自监督（AutoEncoder）

    输入一个图片到Encoder中，然后学习出能够表征出这幅图片的特征，然后再输入一个模糊的图片，然后这个模糊的图片能够利用之前学习到的特征重构出原始的图片，这里学习到的能够提取出表征图片特征的encoder就叫做autoEncoder。但是这一类方法往往太集中表示像素级的特征，整体的语义建模上不够好。

  - 判别式自监督

    学习能够帮助我们判别两个物体是否是同一个标签的特征。

    - Info NCE（最大互信息目标）**to do**

      $$L_y = -log\frac{exp(sim(y, y^{\prime})\\/\tau)}{\sum_{i=0}^{N}exp(sim(y, y^{\prime})\\/\tau)}$$

  - Supervised CL

- Adversarial training

- Multi-exit