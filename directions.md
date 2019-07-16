# Directions for Bert on Images

1. Scale up to bigger images: ImageNet, MIT Places, COCO

2. Unsupervised training:

- Same Image Prediction (SIP): in the same idea as next sentence prediction,
  give two patches, either from the same image or different and predict if
  patches come from the same image or not,
- Evaluate visually the reconstruction loss, _from last ablation study it seemed useless,
  maybe classification loss is too strong in supervised setting_.

3. Improve other task than classification (already done by [1]),

- VQA (Visual Question Answering) or Image Inference (CLEVR)
- Few shot learning with pretraining on ImageNet and finetuning on subset of ImageNet?

## Related Work

[Attention Augmented Convolutional Networks](https://arxiv.org/pdf/1904.09925.pdf) [1]
made meaningful contribution in using Attention on images (BERT style).
They show that replacing CNN layers by attention performs well and allow to reduce the number of layers
because receptive field is unbounded thanks to attention.
Their positional encoding does not perform well in practice.
Resource constrained setting.

[Generative Modeling with Sparse Transformers](https://arxiv.org/pdf/1904.10509.pdf) [2](OpenAI)
used sparse (row/column) transformer for inpainting.

## Improvements

- Accuracy curve is bad at the beginning of the training, i.e. going down for 20 epochs then only starting to improve, possible solutions:
  - we could use the [learning rate schedule](https://github.com/google-research/bert/issues/425) from BERT (linear warmup).
  - improve initialization (either of BERT or of positional encoding)?

### Misc

- [ ] Print number of parameters in the model

### Refs

(1) [Attention Augmented Convolutional Networks](https://arxiv.org/pdf/1904.09925.pdf)

(2) [Generative Modeling with Sparse Transformers](https://arxiv.org/pdf/1904.10509.pdf)
