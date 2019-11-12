# Self-Attention and Convolution

[On the Relationship between Self-Attention and Convolutional Layers](https://arxiv.org/pdf/1911.03584.pdf)<br/>
Jean-Baptiste Cordonnier, Andreas Loukas and Martin Jaggi.

### Abstract

Recent trends of incorporating attention mechanisms in vision have led researchers to reconsider the supremacy of convolutional layers as a primary building block. Beyond helping CNNs to handle long-range dependencies, Ramachandran et al. (2019) showed that attention can completely replace convolution and achieve state-of-the-art performance on vision tasks. This raises the question: do learned attention layers operate similarly to convolutional layers? This work provides evidence that attention layers can perform convolution and, indeed, they often learn to do so in practice. Specifically, we prove that a multi-head self-attention layer with sufficient number of heads is at least as powerful as any convolutional layer. Our numerical experiments then show that the phenomenon also occurs in practice, corroborating our analysis. Our code is publicly available.

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

If you use this code, please cite the following [paper](https://arxiv.org/pdf/1911.03584.pdf):

```
@misc{cordonnier2019relationship,
    title={On the Relationship between Self-Attention and Convolutional Layers},
    author={Jean-Baptiste Cordonnier and Andreas Loukas and Martin Jaggi},
    year={2019},
    eprint={1911.03584},
    archivePrefix={arXiv}
}
```
