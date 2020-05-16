# Self-Attention and Convolution

The code accompanies the paper [On the Relationship between Self-Attention and Convolutional Layers](https://openreview.net/pdf?id=HJlnC1rKPB) by [Jean-Baptiste Cordonnier](http://jbcordonnier.com/), [Andreas Loukas](https://andreasloukas.blog/) and [Martin Jaggi](https://m8j.net/) that appeared in ICLR 2020.

### Abstract

Recent trends of incorporating attention mechanisms in vision have led researchers to reconsider the supremacy of convolutional layers as a primary building block. Beyond helping CNNs to handle long-range dependencies, Ramachandran et al. (2019) showed that attention can completely replace convolution and achieve state-of-the-art performance on vision tasks. This raises the question: do learned attention layers operate similarly to convolutional layers? This work provides evidence that attention layers can perform convolution and, indeed, they often learn to do so in practice. Specifically, we prove that a multi-head self-attention layer with sufficient number of heads is at least as powerful as any convolutional layer. Our numerical experiments then show that the phenomenon also occurs in practice, corroborating our analysis. Our code is publicly available.

### Interact with Attention

Check out our [interactive website](https://epfml.github.io/attention-cnn/).

### Reproduce

To run our code on a Ubuntu machine with a GPU, install the Python packages in a fresh Anaconda environment:

```bash
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
pip install -r requirements.txt
```

All experiments presented in the paper are reproducible by running the scripts in `runs/`, for example:

```
bash runs/quadratic/run.sh
```

### Reference

If you use this code, please cite the following [paper](https://openreview.net/pdf?id=HJlnC1rKPB):

```
@inproceedings{
    Cordonnier2020On,
    title={On the Relationship between Self-Attention and Convolutional Layers},
    author={Jean-Baptiste Cordonnier and Andreas Loukas and Martin Jaggi},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=HJlnC1rKPB}
}
```
