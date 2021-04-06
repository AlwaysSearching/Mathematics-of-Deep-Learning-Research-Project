# uses xavier_normal_ initialization
from torch.nn.init import xavier_normal_ as xavier
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data,gain=args.gain)
        m.bias.data.zero_()

net.apply(weights_init)
FC1 = nn.Linear(512*k, 10).cuda()
FC2 = nn.Linear(512*k, 10).cuda()
xavier(FC1.weight.data, gain=args.gain)
# set gradients manually to zero, might be a pytorch specific thing to do
# https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903
FC1.bias.data.zero_()
net2=copy.deepcopy(net)

# this appears to be important
# FC1 and FC2 appears to be two separate last layer for two neural nets (net and net2, which is a copy of net), as shown in train() 
# Symmetrize!
FC2.weight.data.copy_(FC1.weight.data)
FC2.bias.data.copy_(FC1.bias.data)

# trains the combined model (two identical parallel models merged by subtracting the outputs of the dense tops)
# the resulting difference of outputs from the two networks is then scaled by alpha (scaling factor) before being passed to loss function (for mse loss)
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        if args.precision == 'double':
            inputs = inputs.double()
        optimizer.zero_grad()
        outputs_ = net(inputs)
        outputs2_ = net2(inputs)
        # the two parallel models are merged here by subtracting first model's dense top by the other
        outputs = FC1(outputs_)-FC2(outputs2_)
        loss = None
        if args.loss == 'ce':
            loss = criterion_train(alpha*outputs, targets)/alpha**2
        elif args.loss== 'mse':
            targets_=targets.unsqueeze(1)
            targets_embed=torch.zeros(targets_.size(0),10).cuda()
            targets_embed.scatter_(1, targets_, 1)
            loss = criterion_train(outputs, targets_embed/alpha)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return train_loss/(1+len(trainloader)),100.*correct/total

# we can use keras mse loss, just need to scale by target by alpha
    pytorch documentation:
    >>> loss = nn.MSELoss()
    >>> input = torch.randn(3, 5, requires_grad=True)
    >>> target = torch.randn(3, 5)
    >>> output = loss(input, target)
    >>> output.backward()

    keras documentation:
    tf.keras.losses.MSE(
        y_true, y_pred
    )

    

# stack_hook is a list (one element per layer) of some kind of trained layer attributes (which I suspect is layer activation) that plays a key role in test time for the lazy_net()
# stack_hook elements are tensors of boolean values denoting whether each element in the output tensor of a layer is > 0 DURING TRAINING TIME
# stack_hook will be populated during training time
net_clone = copy.deepcopy(net)
stack_hook=[]

# iterate each layer and add a hook to access the layer outputs during training, similar to callbacks in tf's fit() function
# https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks
# The hook can be a forward hook or a backward hook. The forward hook will be executed when a forward call is executed.
for i in range(len(net_clone.features)):
    if net_clone.features[i].__class__.__name__=='ReLU':
        net_clone.features[i].register_forward_hook(hook_extract_relu)
    elif net_clone.features[i].__class__.__name__ == 'MaxPool2d':
        net_clone.features[i].register_forward_hook(hook_extract_maxpool)
    elif net.features[i].__class__.__name__ == 'BasicBlock':
        net_clone.features[i].register_forward_hook(hook_extract_basicblock)

# forward hook (callback) methods to extract layer outputs, separately implemented for each type of layers
# in tf (at least with the high level apis) should be able to use model.weights in a callback function to achieve the same
# here the stack_hook entry p appears to be True if out element > 0 else False (relu activation?)

# example:
# >>> a
# tensor([[-0.7113],
#         [ 1.1290],
#         [ 0.1133],
#         [ 1.5920]])
# >>> a>0
# tensor([[False],
#         [ True],
#         [ True],
#         [ True]])

# args.precision is an command line input argument that specifies whether to use float or double precision for the experiment
def hook_extract_relu(module, input, out):
    global stack_hook
    p = out > 0
    if args.precision=='float':
        p = p.float()
    else:
        p=p.double()
    stack_hook.append(p)

# separate hook method for maxpool layer, appears to apply unpooling to output to extract the desired attributes
# (pooling the input to obtain the indices needed for unpooling of output)
# here the stack_hook entry p appears to be True if unpooled out element > 0 else False
def hook_extract_maxpool(module, inp, outp):
    global stack_hook
    inp = inp[0]

    _,idx=pooling_layer(inp)
    out = unpooling_layer(outp,idx)
    p = out > 0

    if args.precision == 'float':
        p = p.float()
    else:a
        p = p.double()
    stack_hook.append(p)

# module here is the block, which is a composition of layers
# the block contains 2 conv2d layers and something called shortcut, which is a conditional conv2d layer
def hook_extract_basicblock(module, inp, outp):
    global stack_hook
    inp = inp[0]
    # run input through the block's first conv2d layer
    a = F.relu(module.conv1(inp))
    # True for positive elements of a and False for elements <=0
    # a is the result of only running through the first conv2d layer
    p = a > 0
    if args.precision == 'float':
        p = p.float()
    else:
        p = p.double()
    # True for positive elements of outp and False for elements <=0
    # outp is the result of running input through both conv2d layers and the 'shortcut' layer
    q = outp>0
    if args.precision == 'float':
        q = q.float()
    else:
        q = q.double()
    # append what appears to be the activation after the first conv2d layer and the activation after the entire block
    stack_hook.append([p,q])



# in the below example epoch output, a list of floats is printed. This list is proportion_lazy, which appears to be related to stack_hook and populated during test time
# proportion_lazy entry appears to be the proportion of layer/block activations/outputs that are the same between lazy_net output and regular output for each layer/block of conv2d layers
Epoch: 55
[64.04078368916439, 49.75936401367187, 54.47552551903115, 61.380059814453155, 51.45770447530865, 50.023481249999996, 72.657065625, 51.90329372829859, 50.58404217155612, 72.99224609375, 50.900621337890655, 51.48249609374997, 96.73110839843756, 91.81488281249996]
epoch 55, log train loss:-4.73640, train acc:94.598, log test loss:-3.93251, log test loss scaled:-3.93251 , test acc:87.05, log loss lazy: 2.1512772487987473, test lazy acc:11.43;


for epoch in range(args.length):
    lr = args.lr /(1.0+100.0*epoch/300.0)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    for i in range(len(stack_hook)):
        stack_hook[i]=None

    loss_train, acc_train = train(epoch)
    loss_test, acc_test, loss_scaled = test()

    proportion_lazy = [0] * len(stack_hook)

    # uses a SEPARATE function to test with lazy_net
    # produces metrics of running lazy_net, which appears to use values stored in stack_hook at training time to perform forward pass
    # it appears that the purpose of this is also to populate proportion_lazy during calls to net_activation() within test_lazy() after lazy_net()
    loss_test_lazy, acc_test_lazy = test_lazy()
    print(proportion_lazy)

    # formatted metrics and saving to log file, straight forward
    print(
        "epoch {}, log train loss:{:.5f}, train acc:{}, log test loss:{:.5f}, log test loss scaled:{:.5f} , test acc:{}, log loss lazy: {}, test lazy acc:{};"
        .format(epoch, np.log(loss_train), acc_train, np.log(loss_test), np.log(loss_scaled), acc_test,
                np.log(loss_test_lazy), acc_test_lazy))
    with open(name_log_txt, "a") as text_file:
        print("epoch {}, log train loss:{:.5f}, train acc:{}, log test loss:{:.5f}, log test loss scaled:{:.5f} , test acc:{}, log loss lazy: {}, test lazy acc:{};"
              .format(epoch, np.log(loss_train), acc_train, np.log(loss_test), np.log(loss_scaled), acc_test, np.log(loss_test_lazy), acc_test_lazy), file=text_file)
        print(proportion_lazy, file=text_file)


# net_clone is a copy of the first network, used in testing lazy_net
net_clone = copy.deepcopy(net)

# before testing, this part runs a random input through net_clone, not sure why this is done
stack_hook = []
x=torch.randn(1,3,32,32).cuda()
if args.precision=='double':
    x=x.double()
net_clone(x)

# proportion_lazy is initialized with 0s, len is the same as the len of stack_hook (i.e. same as the number of layers/blocks in the network)
# proportion_lazy entry appears to be the proportion of layer/block activations/outputs that are the same between lazy_net output and regular output for each layer/block of conv2d layers
proportion_lazy = [0] * len(stack_hook)
del x


# function to perform standard testing, producing test-time loss and accuracy metrics
# mirrors the architecture in train(), but for inference, also scaling loss by alpha here
# torch.no_grad() disables training
def test():
    global best_acc
    test_loss = 0
    test_loss_scaled = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device).float(), targets.to(device)
            if args.precision=='double':
                inputs=inputs.double()
            outputs_ = net(inputs)
            outputs2_ = net2(inputs)
            outputs = FC1(outputs_) - FC2(outputs2_)

            loss = 0
            loss_scaled = 0
            if args.loss == 'ce':
                loss = criterion(outputs, targets)
                loss_scaled = criterion(alpha * outputs, targets) / alpha ** 2
            elif args.loss == 'mse':
                targets_ = targets.unsqueeze(1)
                targets_embed = torch.zeros(targets_.size(0), 10).cuda()
                targets_embed.scatter_(1, targets_, 1)
                loss =criterion(outputs, targets_embed)
                loss_scaled = criterion(outputs, targets_embed / alpha)

            test_loss_scaled += loss_scaled.item()
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.*correct/total
    return test_loss / (1 + len(testloader)), acc, test_loss_scaled/ (1 + len(testloader))

# separate testing function that, instead of calling net for inference of the first network, runs input through lazy_net()
# net2 is still run through net2() as in test()
# net_activation is called after lazy_net()
# otherwise mirrors the architecture in train(), but for inference, also scaling loss by alpha here
def test_lazy():
    global best_acc
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # device = 'cuda'
            inputs, targets = inputs.to(device), targets.to(device)
            if args.precision == 'double':
                inputs = inputs.double()
            # runs batch input through lazy_net to obtain output
            outputs_ = lazy_net(inputs)
            net_activation(inputs)
            outputs2_ = net2(inputs)
            outputs = FC1(outputs_) - FC2(outputs2_)
            loss = 0
            if args.loss == 'ce':
                loss = criterion_train(alpha * outputs, targets) / alpha ** 2
            elif args.loss == 'mse':
                targets_ = targets.unsqueeze(1)
                targets_embed = torch.zeros(targets_.size(0), 10).cuda()
                targets_embed.scatter_(1, targets_, 1)
                loss = criterion_train(outputs, targets_embed / alpha)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.*correct/total
    return test_loss / (1 + len(testloader)), acc

# change: for reasons unknown sometimes the multiplications in lazy_net and net_activation 
# have inputs with shapes that are off by 1 in dim 2 and 3, this checks for that and truncates if necessary
def truncate_shape(x, y):
    # print('xshape', x.shape)
    # print('yshape', y.shape)
    if x.shape[-1] == y.shape[-1]+1:
        # print('truncating x in dim 2 and 3 to match y\'s shape')
        x = x[:,:,:-1,:-1]
    return x

# called during test time by test_lazy()
# appears to use what is saved in stack_hook during training to implement inference on input
# I added truncate_shape because when I was running the original version, there seem to be shape mismatch (off by 1) in some of the layers
# possible explaination of the shape mismatch is that I changed kernel_size for conv2d layers in the model from 3 to 2 because kernel_size=3 produces Output size is too small Runtime Error
def lazy_net(x):
    global stack_hook
    z = x.clone()
    stack_hook = []
    # pass the input through a copy of net for some reason
    net_clone(x)

    # j in the index of layer, incremented each loop, used to access what's stored in stack_hook during training
    # although the author could have just used i instead, unsure of the reason for using a separate variable
    j = 0
    # again, net.features is a list of layers in net
    for i in range(len(net.features)):
        if net.features[i].__class__.__name__ == 'ReLU':
            p = stack_hook[j]
            # x is input, z is x.clone(), which makes z also the batched input
            # here a copy of the input is tensor-multiplied with the corresponding stack_hook content for this layer
            z_ = torch.mul(z , p)
            # appears to be done with the layer after the tensor multiplicaton (assuming relu stack_hook is just {0,1}, this would make sense)
            z = z_
            j = j + 1
        elif net.features[i].__class__.__name__ == 'MaxPool2d':
            p = stack_hook[j]
            z = truncate_shape(z, p)
            # z * p is the same as torch.mul(z , p)
            z = z * p
            # after multiplication run through a pooling layer
            z,_ =pooling_layer(z)
            j = j + 1
        elif net.features[i].__class__.__name__ == 'BasicBlock':
            p,q = stack_hook[j]
            # run input through first conv2d layer in the block (defined below)
            z_ = net.features[i].conv1(z)
            z_ = truncate_shape(z_, p)
            # multiply with stack_hook[i][0]
            z_ = z_*p
            # run that through seconds conv2 layer within the block
            z_ = net.features[i].conv2(z_)
            # not exactly sure what shortcut layer does, see BasicBlock class for more details
            z = net.features[i].shortcut(z)+z_
            # multiply with stack_hook[i][1]
            z = z*q
            # j == i (I think)
            j = j + 1
        else:
            # if layer is neither ReLU, Maxpool, or conv block, run through the layer in net in a normal way without the above shenanigans with stack_hook
            z = net.features[i](z)
    z = z.view(z.size(0), -1)
    return z

# from models.py, a BasicBlock contains 2 conv2d layers and another conv2d layer under certain circumstances
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut =nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride))#, bias=False),
             #   nn.BatchNorm2d(self.expansion*planes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# what happens here appears to a bit similar to lazy_net(), except that proportion_lazy is populated with some 
def net_activation(x):
    global stack_hook
    global proportion_lazy
    z = x.clone()
    stack_hook = []
    # pass the input through a copy of net for some reason
    net_clone(x)

    j = 0
    for i in range(len(net.features)):
        if net.features[i].__class__.__name__ == 'ReLU':
            # the stack_hook saved from training
            p = stack_hook[j]
            # result of passing input through this layer normally
            # tensors of boolean values denoting whether an element in the output of a layer is > 0 DURING TRAINING TIME
            z = net.features[i](z)
            # True if the above > 0 else False, 
            # similar to how stack_hook entries are tensors of boolean values denoting whether an element in the output of a layer is > 0
            p_=z>0
            if args.precision == 'float':
                p_ = p_.float()
            else:
                p_ = p_.double()
            # torch.sum will sum the number of True in a boolean tensor
            # example:
            # >>> a>0
            # tensor([[False],
            #         [ True],
            #         [ True],
            #         [ True]])
            # >>> torch.sum(a>0)
            # tensor(3)

            # numel returns the total number of elements in a tensor.

            # hence, I think proportion_lazy entry appears to be the proportion of layer/block activations/outputs that are the same between lazy_net output and regular output
            proportion_lazy[j]+= float(torch.sum(p_ == p)) / float(p.numel())
            j = j + 1
        # the below repeat the same but for other types of layers
        elif net.features[i].__class__.__name__ == 'MaxPool2d':
            # this part does the same of what is done in lazy_net()
            p = stack_hook[j]
            z = truncate_shape(z, p)
            z_ = z * p
            z_, _ = pooling_layer(z_)
            # output of normal pass of input
            z = net.features[i](z)
            # proportion of the output tensor elements that are the same between lazy_net and normal pass
            proportion_lazy[j]+=float(torch.sum(z == z_)) / float(z.numel())
            j = j + 1
        elif net.features[i].__class__.__name__ == 'BasicBlock':
            p, q = stack_hook[j]
            z = net.features[i](z)
            q_ = z>0
            if args.precision == 'float':
                q_ = q_.float()
            else:
                q_ = q_.double()
            proportion_lazy[j] += float(torch.sum(q_ == q)) / float(q.numel())
            j = j + 1
        else:
            z = net.features[i](z)
    z = z.view(z.size(0), -1)
    return z


