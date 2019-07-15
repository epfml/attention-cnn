# Directions for Bert on Images

1. Scale up to bigger images: ImageNet, MIT Places
2. Unsupervised training:

- Same Image Prediction (SIP): in the same idea as next sentence prediction,
  give two patches, either from the same image or different and predict if
  patches come from the same image or not,
- Evaluate visually the reconstruction loss, _from last ablation study it seemed useless,
  maybe classification loss is too strong in supervised setting_.

3. What should we evaluate against?

- Few shot learning with pretraining on ImageNet and finetuning on subset of ImageNet?
- Restricted number of parameters?
- VQA (Visual Question Answering) or Image Inference

## Improvements

- Accuracy curve is bad at the beginning of the training, i.e. going down for 20 epochs then only starting to improve, possible solutions:
  - we could use the [learning rate schedule](https://github.com/google-research/bert/issues/425) from BERT (linear warmup).
  - improve initialization (either of BERT or of positional encoding)?

### Misc

- [ ] Print number of parameters in the model
