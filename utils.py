def get_mnist(Binarized = True, batch_size = 64, semi_supervised = None, classes = [0,1,2,3,4,5,6,7,8,9], labels_per_class = 10):
    '''
    Give the dataset MNIST from torch-vision, by default returns the binarized version;
    can also return a splitted dataset in case of semi-supervised learning (choose which class and how many):
    - use "dataloader"(default) to return a dataloader in which the dataset is splitted (uniform stratified sampling)
    - use "numpy" for the dataset to be contained in a numpy array
    '''
    import torch
    from torch.utils.data import DataLoader, Subset
    from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor, Lambda, Compose

    def uniform_stratified_sampler(labels, classes, n=None):
        """
        Stratified sampler that distributes labels uniformly by
        sampling at most n data points per class
        """
        from functools import reduce
        import numpy as np

        # Only choose digits in n_labels
        (indices,) = np.where(reduce(lambda x, y: x | y, [labels.numpy() == i for i in classes]))

        # Ensure uniform distribution of labels
        np.random.shuffle(indices)
        indices = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n] for i in classes])

        indices = torch.from_numpy(indices)
        sampler = SubsetRandomSampler(indices)
        return sampler
    
    flatten = Lambda(lambda x: ToTensor()(x).view(28**2))
    binarize = Lambda(lambda x: torch.bernoulli(x))

    if Binarized: 
        transform = Compose([flatten,binarize])
    else:
        transform = flatten
    
    trainset = MNIST("./", train=True, download=True, transform=transform)
    testset = MNIST("./", train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=2,shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=2,shuffle=True)

    if not semi_supervised:
        return train_loader, test_loader
    
    elif semi_supervised == "dataloader":
        #uniform sampling
        uniform_sampler = uniform_stratified_sampler(trainset.targets, classes=classes, n=labels_per_class)

        #split the trainset without repetition of samples
        labelled_indices = uniform_sampler.indices.numpy().tolist()
        unlabelled_indices = list(set(range(len(trainset))) - set(labelled_indices))
        unlabelled_subset = Subset(trainset,unlabelled_indices)
        
        #create Dataloader
        labelled = DataLoader(trainset, batch_size=batch_size, sampler=uniform_sampler)
        unlabelled = DataLoader(unlabelled_subset, batch_size=batch_size, shuffle=True)
        
        return labelled, unlabelled, test_loader
        
    elif semi_supervised == "all":
        #uniform sampling
        uniform_sampler = uniform_stratified_sampler(trainset.targets, classes=classes, n=labels_per_class)
        
        #split the trainset without repetition of samples
        labelled_indices = uniform_sampler.indices.numpy().tolist()
        unlabelled_indices = list(set(range(len(trainset))) - set(labelled_indices))
        unlabelled_subset = Subset(trainset,unlabelled_indices)

        #create Dataloader with batch size as total number of samples
        labelled_all = DataLoader(trainset, batch_size=labels_per_class*len(classes), sampler=uniform_sampler)
        test_loader_all = DataLoader(testset, batch_size=len(testset),shuffle=True)
        
        #get the samples from the dataloader
        x_train,y_train = next(iter(labelled_all))
        x_test,y_test = next(iter(test_loader_all))

        return x_train, y_train, x_test, y_test
    else:
        raise NotImplementedError("Wrong argument for semi_supervised")

# Weight initialization function
'''
Initialization function for the weights of dense layers
'''
def init_weights(net, init_method="Xavier"):
    from torch import nn
    if type(net)== nn.Linear:
        if init_method == "Xavier":
            torch.nn.init.xavier_uniform_(net.weight)
            net.bias.data.zero_()
        elif init_method == "Kaiming-He":
            torch.nn.init.kaiming_uniform_(net.weight)
            net.bias.data.zero_()

import torch
import numpy as np
from collections import  defaultdict

def training(model,dataloader,optimizer, cuda_available:bool,training_data:defaultdict, verbose=True):
    '''
    Training function using the entire training set, pass a default dict by reference to store the values of the diagnostics.
    '''
    training_epoch_data = defaultdict(list)
    model.train()
    if cuda_available: model = model.cuda()
    for input,target in dataloader:
        if cuda_available: input = input.cuda()

        loss, diagnostics, _, _ = model(input)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for k, v in diagnostics.items():
            training_epoch_data[k] += [v.mean().item()]
           

    # this should be in the other loop
    for k, v in training_epoch_data.items():
        training_data[k] += [np.mean(training_epoch_data[k])]

    if verbose:
        print("elbo : {}, BCE : {}, KL : {}".format(round(training_data["elbo"][-1],3),
                                                    round(training_data["likelihood"][-1],3),
                                                    round(training_data["KL"][-1],3)))

    pass

def validate(model,dataloader,cuda_available:bool,validation_data:defaultdict, verbose=True):
    '''
    Validation function using a single batch of the test set, pass a default dict by reference to store the values of the diagnostics
    '''
    with torch.no_grad():
        model.eval()
        input,target =next(iter(dataloader))
        if cuda_available:
            model = model.cuda()
            input = input.cuda()
        
        _, diagnostics, _, _ = model(input)

        for k, v in diagnostics.items():
            validation_data[k] += [v.mean().item()]
    if verbose:
        print("Validation loss = ", round(validation_data["elbo"][-1],3), "\n")
    pass

def test(model,dataloader,cuda_available:bool,test_data:defaultdict,verbose=True):
    '''
    Test function using the entire test set, pass a default dict by reference to store the values of the diagnostics
    '''
    testing_epoch_data = defaultdict(list)
    with torch.no_grad():
        model.eval()
        for input,target in dataloader:
            if cuda_available:
                model = model.cuda()
                input = input.cuda()
            
            _, diagnostics, _, _ = model(input)

            for k, v in diagnostics.items():
                testing_epoch_data[k] += [v.mean().item()]
        
        # this should be in the other loop
        for k, v in testing_epoch_data.items():
            test_data[k] += [np.mean(testing_epoch_data[k])]
    if verbose:
        print("Test loss = ", round(test_data["elbo"][-1],3), "\n")
    
    pass


def training_M2(model, dataloader_labelled, dataloader_unlabelled, optimizer, cuda_available:bool, training_data:defaultdict, training_data_labelled:defaultdict, training_data_unlabelled:defaultdict, verbose=True):
    from itertools import cycle
    import torch.nn.functional as F
    #init 
    training_epoch_data_labelled = defaultdict(list)
    training_epoch_data_unlabelled = defaultdict(list)
    correct = 0
    total = 0
    loss_class=0
    running_loss=0  
    model.train()

    # Go through both labelled and unlabelled data
    for i, ((x, y), (u, _))  in enumerate(zip(cycle(dataloader_labelled), dataloader_unlabelled)):

        if cuda_available:
            x, y, u = x.cuda(), y.cuda(), u.cuda()
            model.cuda()
        
        # Send labelled data through autoencoder
        loss_labelled, diagnostics_labelled, _, _ = model(x, y)
        loss_unlabelled, diagnostics_unlabelled, _, _ = model(u)
        output = model.classifier(x)
        
        

        loss =  loss_unlabelled + loss_labelled 

        optimizer.zero_grad()
        loss.backward()       
        optimizer.step()

        
        ##diagnostic VAE batch

        for k, v in diagnostics_labelled.items():
            if k == "classifier_loss":
                training_epoch_data_labelled[k] += [v]
            else:
                training_epoch_data_labelled[k] += [v.mean().item()]


        for k, v in diagnostics_unlabelled.items():            
            training_epoch_data_unlabelled[k] += [v.mean().item()]

        ##diagnostic classifier batch
        _, predicted = torch.max(output.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum()
        
        # loss_class += classifier_loss.item()
        running_loss += loss.item()
    
    ##diagnostic VAE epoch
    for k, v in training_epoch_data_labelled.items():
        training_data_labelled[k] += [np.mean(training_epoch_data_labelled[k])]

    for k, v in training_epoch_data_unlabelled.items():
        training_data_unlabelled[k] += [np.mean(training_epoch_data_unlabelled[k])]
    
    
    ##diagnostic model epoch
    training_data["Tot_loss"] += [running_loss / i+1]
    training_data["classifier_loss"] += [loss_class / i+1]
    training_data["classifier_accuracy"] += [100 * correct.true_divide(total).item()]
    
    if verbose:
        print("Trainig :")
        print("Labelled, elbo(L) : {}, BCE : {}, KL : {}, Classifier loss : {},".format(round(training_data_labelled["elbo"][-1],3),
                                                                 round(training_data_labelled["likelihood"][-1],3),
                                                                 round(training_data_labelled["KL"][-1], 4),
                                                                 round(training_data_labelled["classifier_loss"][-1], 7)))
    
        print("Unlabelled; elbo(U) : {}, BCE : {}, KL : {}, H : {}".format(round(training_data_unlabelled["elbo"][-1],3),
                                                                           round(training_data_unlabelled["likelihood"][-1],3),
                                                                           round(training_data_unlabelled["KL"][-1], 4),
                                                                           round(training_data_unlabelled["Entropy H"][-1],3)))
        
        print("Total loss : {},  Classifier accuracy : {}".format(round(training_data["Tot_loss"][-1],3), round(training_data["classifier_accuracy"][-1], 3)))

def training_M2_v2(model, dataloader_labelled, dataloader_unlabelled, optimizer, frequency:int, cuda_available:bool, training_data:defaultdict, training_data_labelled:defaultdict, training_data_unlabelled:defaultdict, verbose=True):
    from itertools import cycle
    import torch.nn.functional as F
    #init 
    training_epoch_data_labelled = defaultdict(list)
    training_epoch_data_unlabelled = defaultdict(list)
    correct = 0
    total = 0
    loss_class=0
    running_loss=0  
    steps_unlabelled = len(dataloader_unlabelled)
    steps_labelled = len(dataloader_labelled)
    batches_per_epoch = steps_labelled + steps_unlabelled
    periodic_interval_batches = int(frequency)

    if cuda_available:
        model.cuda()
    
    model.train()
    
    batch_labelled_proc = 0
    for i in range(batches_per_epoch):

        # whether this batch is supervised or not
        is_supervised = (i % periodic_interval_batches == 1) #and batch_labelled_proc < steps_labelled
        
        # extract the corresponding batch
        if is_supervised:
            x, y = next(iter(dataloader_labelled))
            if cuda_available:
                x, y = x.cuda(), y.cuda()
                model.cuda()
            loss_labelled, diagnostics_labelled, _, _ = model(x, y)
            output = model.classifier(x)
            
            loss = loss_labelled

            batch_labelled_proc += 1

            for k, v in diagnostics_labelled.items():
                if k == "classifier_loss":
                    training_epoch_data_labelled[k] += [v]
                else:
                    training_epoch_data_labelled[k] += [v.mean().item()]


            ##diagnostic classifier batch
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum()
            
            running_loss += loss.item()

        else:
            u, _ = next(iter(dataloader_unlabelled))
            if cuda_available:
                u = u.cuda()
                model.cuda()
            loss_unlabelled, diagnostics_unlabelled, _, _ = model(u)

            loss =  loss_unlabelled
            running_loss += loss.item()

            for k, v in diagnostics_unlabelled.items():            
                training_epoch_data_unlabelled[k] += [v.mean().item()]
        
        #backward step
        optimizer.zero_grad()
        loss.backward()        
        optimizer.step()


    ##diagnostic VAE epoch
    for k, v in training_epoch_data_labelled.items():
        training_data_labelled[k] += [np.mean(training_epoch_data_labelled[k])]

    for k, v in training_epoch_data_unlabelled.items():
        training_data_unlabelled[k] += [np.mean(training_epoch_data_unlabelled[k])]
    
    
    ##diagnostic model epoch
    training_data["Tot_loss"] += [running_loss / i+1]
    training_data["classifier_accuracy"] += [100 * correct.true_divide(total).item()]
    
    if verbose:
        print("Trainig :")
        print("Labelled, elbo(L) : {}, BCE : {}, KL : {}, Classifier loss : {},".format(round(training_data_labelled["elbo"][-1],3),
                                                                 round(training_data_labelled["likelihood"][-1],3),
                                                                 round(training_data_labelled["KL"][-1], 4),
                                                                 round(training_data_labelled["classifier_loss"][-1], 7)))
    
        print("Unlabelled; elbo(U) : {}, BCE : {}, KL : {}, H : {}".format(round(training_data_unlabelled["elbo"][-1],3),
                                                                           round(training_data_unlabelled["likelihood"][-1],3),
                                                                           round(training_data_unlabelled["KL"][-1], 4),
                                                                           round(training_data_unlabelled["Entropy H"][-1],3)))
        
        print("Total loss : {},  Classifier accuracy : {}".format(round(training_data["Tot_loss"][-1],3), round(training_data["classifier_accuracy"][-1], 2)))




def test_M2(model,dataloader, alpha, cuda_available:bool,test_data:defaultdict, verbose=True):
    import torch.nn.functional as F
    #init
    running_loss = 0
    correct = 0
    total = 0
    loss_class=0
    saved = False
    with torch.no_grad():
        model.eval()
        for i, (input,target) in enumerate(dataloader):
            if cuda_available:
                model = model.cuda()
                input = input.cuda()
                target = target.cuda()
            
            loss_lab , _, _, _ = model(input,target)
            loss_unlab, _,_,_, = model(input)
            
            output = model.classifier(input)
            classifier_loss = F.cross_entropy(output,target)

            loss_class += classifier_loss.item()
            loss = loss_lab + alpha* classifier_loss + loss_unlab

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum()
      
            running_loss += loss.item()
        
        try: max_acc = max(test_data["test_accuracy"])
        except : max_acc = 0

        test_data["Tot_loss"] += [running_loss / i+1]
        test_data["classifier_loss"] += [loss_class / i+1]
        test_data["test_accuracy"] += [100*correct.true_divide(total).item()]
   
        current_acc = test_data["test_accuracy"][-1] 
        if current_acc >= max_acc:
            torch.save(model.classifier.state_dict(),"./state_dict_classifier.pt")
            max_acc = current_acc
            saved = True

    if verbose: 
        print("Test :")
        print("Tot loss : {}, Classifier loss : {}, Classifier accuracy : {}".format(round(test_data["Tot_loss"][-1],3),
                                                                                     round(test_data["classifier_loss"][-1],3),
                                                                                     round(test_data["test_accuracy"][-1], 2)))
        if saved:
            print(f"Saved Checkpoint with accuracy {max_acc}")     
