# Self-Supervised Learning for OOD Detection

A Simplified Pytorch implementation of *Using Self-Supervised Learning Can Improve Model Robustness and Uncertainty(NeurIPS 2019)*

## Introduction
A number of **Self-supervised learning** have been proposed, each exploring a different pretext task. 
![image](https://user-images.githubusercontent.com/37788686/98618879-6807f680-2345-11eb-8a9a-39842d581add.png)

For example, the above image is unsupervised representation learning by predicting image rotations. Predicting rotation requires modeling shape. Texture alone is not sufficient for determining whether the zebra is flipped, although it may be sufficient for classification under ideal conditions. Thus, training with self-supervised auxiliary rotations may improve robustness.

## Method
CE Loss + auxiliary self-supervised loss(predicting rotations)
![image](https://user-images.githubusercontent.com/37788686/98619112-e6fd2f00-2345-11eb-8fed-05b653d0218a.png)

## Results
![image](https://user-images.githubusercontent.com/37788686/98619356-6a1e8500-2346-11eb-9190-53e7ce039329.png)

ROT(rotation method) shows better performance than MSP(Maximum Softmax Probability). So auxiliary self-supervised loss improves model robustness.

**The code supports only Multi-class OOD Detection experiment(in-dist: CIFAR-10, Out-of-dist: CIFAR-100/SVHN)** 


## Command 
  - RotNet-OOD
  
    python test.py --method=rot --ood_dataset=cifar100
  
  - baseline
  
    python test.py --method=msp --ood_dataset=svhn

## Code Explanation
* train.py
```python
    for x_tf_0, x_tf_90, x_tf_180, x_tf_270, batch_y in tqdm(train_loader):  
        batch_size = x_tf_0.shape[0]

        batch_x = torch.cat([x_tf_0, x_tf_90, x_tf_180, x_tf_270], 0).cuda()    # batch_x: [bs*4, 3, 32, 32]
        batch_y = batch_y.cuda()                                                # batch_y: [bs]        
        batch_rot_y = torch.cat((                                               # batch_rot_y: [bs*4]
            torch.zeros(batch_size),
            torch.ones(batch_size),
            2 * torch.ones(batch_size),
            3 * torch.ones(batch_size)
        ), 0).long().cuda()

        optimizer.zero_grad()

        logits, pen = model(batch_x)
        
        classification_logits = logits[:batch_size]
        rot_logits  = model.rot_head(pen)

        # classification loss(only using not rotated images)
        classification_loss = F.cross_entropy(classification_logits, batch_y)
        # rotation loss
        rot_loss = F.cross_entropy(rot_logits, batch_rot_y)  
        
        # use self-supervised rotation loss 
        if args.method == 'rot':
            loss = classification_loss + args.rot_loss_weight * rot_loss 
        # baseline, maximum softmax probability
        elif args.method == 'msp':
            loss = classification_loss
```
x_tf_0, x_tf_90, x_tf_180, x_tf_270 are 0, 90, 180, 270-degree rotated images and have 0, 1, 2, 3 as label, respectively. At method=='rot', rotation loss is added to loss and method=='mlp' uses only classification loss. 

## Reference
  - full code(by authors): https://github.com/hendrycks/ss-ood
  - Unsupervised Representation Learning by Predicting Image Rotations: https://arxiv.org/abs/1803.07728
  - Using Self-Supervised Learning Can Improve Model Robustness and Uncertainty(NeurIPS 2019): https://arxiv.org/abs/1906.12340
  - A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks(ICLR 2017): https://arxiv.org/abs/1610.02136

