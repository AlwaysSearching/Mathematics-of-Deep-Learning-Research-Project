Modified code for the CNN experiments from https://github.com/edouardoyallon/lazy-training-CNN

Fixed issues that prevented the original repo from running

# changes: 
1. set num_workers=0 in dataloader (>0 num_workers might work in unix environments, has issue with windows)
2. reduced kernel size of conv layers by 1 (from 3 to 2)
3. changed log file naming format (removed ':')
4. added truncate_shape() to ensure shapes match in lazy_net() and net_activation()

# NOTE: 
1. change 'py' to 'python' in script.sh if your environment invoke python using 'python' 
2. script.sh simply runs similar commands with varying scaling factors and learning rates
3. 'length' is number of epochs
4. Some configurations (e.g. lr=1, scaling=10000000) will learn reasonably well under double precision but would barely learn anything at all under float precision
5. the double precision experiments requires substanitally more computataional resources to run at at reasonable speed

# example for running a single experiment: 

py train.py --scaling_factor 10.0  --lr 0.01 --gain 1.0 --schedule 'b' --loss 'mse' --length 100 --precision 'float'

output for example (first 3 epochs):

Epoch: 0
[99.90087121212126, 51.788640136718726, 99.73974589100344, 50.18372802734373, 99.5557412229938, 99.122844921875, 98.69636406249998, 98.82523980034722, 98.29132932079084, 49.159322916666675, 97.86321899414062, 96.99761015624999, 47.9044482421875, 93.66130859374998]
epoch 0, log train loss:-7.04988, train acc:23.762, log test loss:-2.33889, log test loss scaled:-7.08806 , test acc:29.41, log loss lazy: -6.252284965071187, test lazy acc:9.47;

Epoch: 1
[99.86485379361798, 51.787092285156234, 99.64549145761247, 50.15415893554687, 99.39911795910497, 98.85959296874998, 98.267828125, 98.47344509548611, 97.78067881058675, 49.054082031250005, 97.2618615722656, 96.24414296874996, 47.66470703125002, 92.1078515625]
epoch 1, log train loss:-7.08432, train acc:28.892, log test loss:-2.34181, log test loss scaled:-7.10496 , test acc:30.91, log loss lazy: -6.2516984490962155, test lazy acc:9.73;

Epoch: 2
[99.83875946969695, 51.78722167968751, 99.58352751946366, 50.13650634765626, 99.2959587191358, 98.70264843749997, 97.99263750000001, 98.26731987847221, 97.49288185586734, 48.96636501736111, 96.9471459960938, 95.85166484375002, 47.5190966796875, 91.35755859375001]
epoch 2, log train loss:-7.09601, train acc:30.212, log test loss:-2.34372, log test loss scaled:-7.11630 , test acc:32.44, log loss lazy: -6.244377700643538, test lazy acc:9.82;

#link to paper: https://arxiv.org/pdf/1812.07956.pdf


README for original repo:
# lazy-training-code

This code was based on https://github.com/kuangliu/pytorch-cifar . 

## Reproducing CNNs experiments

If you want to obtain CNN experiments accuracies and loss from the paper, simply run:

```
cd cnn
sh script.sh
```

The __double__ precision experiments require a Tesla or Volta GPUs for handling this numerical precision at a reasonable speed...

## Reproducing shallow experiments

All the codes necessary to reproduce the results from the paper as located in `shallow-nn`

## Contributions

All contributions are welcome.
