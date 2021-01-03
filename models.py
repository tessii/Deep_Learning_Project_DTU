import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, OneHotCategorical


#Net prototipe
class FFNN(nn.Module):
    '''
    Prototipe class for Feed-Forward Neural Network (dense layer), as initialization is it possible to pass a list with the number of hidden units and choose activation function;
    It is also possible to use batchnorm and dropout (this will be true and costant for each layer)
    '''
    def __init__(self, layers_, num_output_, activation ="ReLU",batchnorm = False, dropout:float = None):
        super().__init__() 
        self.layers = []
        # layer construction
        for i in range(len(layers_)-1):#more than 2 layers
            self.layers.append(nn.Linear(layers_[i], layers_[i+1]))
            if activation == "ReLU":
                self.layers.append(nn.ReLU())
            elif activation == "LeakyReLU":
                self.layers.append(nn.LeakyReLU())
            elif activation == "Softplus":
                self.layers.append(nn.Softplus())
            else:
                raise NotImplementedError("Wrong activation function")
            if batchnorm:
                self.layers.append(nn.BatchNorm1d(layers_[i+1]))
            if dropout:
                self.layers.append(nn.Dropout(dropout))

        # output layer
        self.layers.append(nn.Linear(layers_[-1], num_output_))

        #finalize net
        self.net = nn.Sequential(*self.layers)
        

    def forward(self, x):
        return self.net(x)

# define network
class Encoder(nn.Module):
    '''
    Encoder class that similar to FFNN class. 
    '''
    def __init__(self, layers_, latent_features, activation ="ReLU", batchnorm:bool = False, dropout:float = None):
        super().__init__() 
        self.layers = []
        # layer construction
        for i in range(len(layers_)-1):#more than 2 layers
            self.layers.append(nn.Linear(layers_[i], layers_[i+1]))
            if activation == "ReLU":
                self.layers.append(nn.ReLU())
            elif activation == "LeakyReLU":
                self.layers.append(nn.LeakyReLU())
            elif activation == "Softplus":
                self.layers.append(nn.Softplus())
            else:
                raise NotImplementedError("Wrong activation function")
            if batchnorm:
                self.layers.append(nn.BatchNorm1d(layers_[i+1]))
            if dropout:
                self.layers.append(nn.Dropout(dropout))

        #finalize net
        self.net = nn.Sequential(*self.layers)
        
        self.mu_dense = torch.nn.Linear(layers_[-1],latent_features)
        self.log_var_dense = nn.Linear(layers_[-1],latent_features)


    def forward(self, x):
        x= self.net(x)
        mu = self.mu_dense(x)
        log_var = self.log_var_dense(x)
        z = Normal(mu, (0.5*log_var).exp()).rsample()  
        # return tensor mu and sigma or return directly the distribution using the gaussian class
        return z, mu, log_var
        # return 


# define network
class Decoder(nn.Module):
    '''
    Decoder class that similar to FFNN class.
    '''
    def __init__(self, layers_, num_output_, activation ="ReLU",batchnorm = False, dropout:float = None):
        super().__init__() 
        self.layers = []
        # layer construction
        for i in range(len(layers_)-1):#more than 2 layers
            self.layers.append(nn.Linear(layers_[i], layers_[i+1]))
            if activation == "ReLU":
                self.layers.append(nn.ReLU())
            elif activation == "LeakyReLU":
                self.layers.append(nn.LeakyReLU())
            elif activation == "Softplus":
                self.layers.append(nn.Softplus())
            else:
                raise NotImplementedError("Wrong activation function")
            if batchnorm:
                self.layers.append(nn.BatchNorm1d(layers_[i+1]))
            if dropout:
                self.layers.append(nn.Dropout(dropout))

        # output layer
        self.layers.append(nn.Linear(layers_[-1], num_output_))

        #finalize net
        self.net = nn.Sequential(*self.layers)

            

    def forward(self, x):
        return self.net(x)


class VAE(nn.Module):
    '''
    Beta Variational Auto-Encoder that is built upon the Encoder-Decor class;
    The initialization is the same as the FFNN class, moreover the Reconstruction loss is chosen to be "Binary Cross Entropy" 
    and the KL divergence is computed in analytical form (both prior and posterior are Gaussion)
    '''
    def __init__(self, enc_layers,dec_layers,latent_features,num_output,beta = 1.0, activation ="ReLU",batchnorm:bool = False, dropout:float = None):
        super().__init__() 
        self.beta = beta      
        self.encoder = Encoder(enc_layers, latent_features, activation,batchnorm,dropout)
        self.decoder = Decoder(dec_layers,num_output,activation,batchnorm,dropout)



    def encode(self,x):
        # encoding from z-batch, must return 2 parameter [mu,sigma],
        z, mu, log_var = self.encoder(x)
        # extract mu and sigma from encoder result
        # mu = self.mu_dense(x)
        # log_var = self.log_var_dense(x)
        # return tensor mu and sigma or return directly the distribution using the gaussian class
        return z , mu, log_var

    
    def decode(self,z):
        # decoding 
        z = self.decoder(z)
        z = torch.sigmoid(z)
        return z

    def sample(self,  n:int, z = None):
        #generation of image = sampling from random noise + decode

        if not z:
            return self.decode(torch.randn(n, self.latent_features))
        else:
            return self.decode(z)


    def elbo(self, x, mu, log_var, z, rec):
        #loss function = reconstruction error + KL-divergence
        BCE_new = - torch.sum(F.binary_cross_entropy(rec, x, reduction="none"),dim = 1) #mean /none 
        KL_analyt = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(),dim=1) # analytical KL
        
        #ELBO
        ELBO = BCE_new - self.beta * KL_analyt
        # ELBO = BCE - self.beta * KL

        loss = -ELBO.mean()
        
        with torch.no_grad():
            diagnostics = {'elbo': ELBO, 'likelihood': BCE_new, 'KL': KL_analyt}
            # diagnostics = {'elbo': ELBO, 'likelihood': BCE, 'KL': KL}

        return loss, diagnostics

    def forward(self,x):
        #posterior param
        z, mu, log_var = self.encode(x)

        #posterior dist (force sigma to be positive with log) + reparametrization
        # post_dist = Normal(mu, (0.5*log_var).exp())
        # z = post_dist.rsample()        
        
        #reconstruction -> log prob with sigmoid
        rec = self.decode(z)
        
        #ELBO
        loss, diagnostic = self.elbo(x, mu, log_var, z, rec)
        
        return [loss, diagnostic, z, rec]

class Classifier(nn.Module):
#just a fast net to do some testing
    def __init__(self,):
        super().__init__()
        
        self.net= nn.Sequential(
            nn.Linear(784,500),
            nn.ReLU(),
            nn.Linear(500,10))

    def forward(self, x):
        x = self.net(x)
        return x


# class M2(nn.Module):
#     '''
#     Beta Variational Auto-Encoder that is built upon the Encoder-Decor class, plus a Classifier as specified in the paper "Deep Generative" from Kingma 2014;
#     '''
#     def __init__(self,enc_layers,dec_layers,latent_features,dec_num_output,beta, layer_classifier, num_classes,activation="ReLU", batchnorm = False, dropout:float = None,activation_classifier="ReLU", batchnorm_classifier = False, dropout_classifier:float = None):
#         super().__init__() 
#         self.beta = beta
#         self.latent_features = latent_features
#         self.num_classes=num_classes
#         # self.classifier = FFNN(layer_classifier,self.num_classes, activation_classifier, batchnorm_classifier, dropout_classifier)
#         self.encoder = Encoder(enc_layers, activation,batchnorm,dropout)
#         self.mu_dense = nn.Linear(enc_layers[-1],self.latent_features)
#         self.log_var_dense = nn.Linear(enc_layers[-1],self.latent_features)
#         self.classifier = FFNN(layer_classifier, self.num_classes,activation_classifier, batchnorm_classifier, dropout_classifier)      
#         self.decoder = Decoder(dec_layers,dec_num_output,activation,batchnorm,dropout)



#     def encode(self,x):
#         # encoding from z-batch, must return 2 parameter [mu,sigma],
#         x = self.encoder(x)
#         # extract mu and sigma from encoder result
#         mu = self.mu_dense(x)
#         log_var = self.log_var_dense(x)

#         # return tensor mu and sigma or return directly the distribution using the gaussian class
#         return mu, log_var

    
#     def decode(self,z):
#         # decoding 
#         z = self.decoder(z)
#         z = torch.sigmoid(z)
#         return z

#     def classify(self,x):
#         probs =  F.softmax(self.classifier(x), dim=-1)
#         return probs
    
#     def elbo(self):
#         raise NotImplementedError("Old method without distinction between labelled and unlabelled")
    
#     def conditional_sample(self, y:int, n:int):
#     #generation of image = sampling from posterior (I believe) + decode
#     # torch.rand(n, self.latent_features)
#         z = torch.randn(n, self.latent_features)
#         y = F.one_hot(torch.tensor(y),self.num_classes).expand(n,-1)
#         z = torch.cat((z,y),dim=1).to(next(self.parameters()).device)
#         return self.decode(z)
    
#     def L_(self, x, mu, log_var, z, rec, cat_prior):
#         #loss function = reconstruction error + KL-divergence + categorical prior
#         BCE = - torch.sum(F.binary_cross_entropy(rec, x, reduction="none"),dim = 1) #mean /none 
#         KL_analyt = (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(),dim=1)) # analytical KL
        
#         #Elbo computation
#         L = BCE + cat_prior - self.beta * KL_analyt


#         with torch.no_grad():
#             diagnostics = {'elbo': L, 'likelihood': BCE, 'KL': KL_analyt, "cat_prior": cat_prior}
            
#         return L, diagnostics


#     def U_(self, L, diagnostics, categorical_post):
#         #classifier loss
#         # H = -torch.sum(torch.mul(probs, torch.log(probs + 1e-8)), dim=-1)
#         H = categorical_post.entropy() #size=batch
#         L_unlabelled = torch.sum(torch.mul(categorical_post.probs, L.view(-1,1)),dim=1)
#         U = L_unlabelled + H
#         with torch.no_grad():
#             diagnostics["Entropy H"] = H

            
#         return U, diagnostics
        
    
#     def forward(self,x,y=None):
#         labelled = False if y is None else True

#         if labelled:
#             y = F.one_hot(y,num_classes=self.num_classes).to(next(self.parameters()).device)

#         elif not labelled:
#             probs = self.classify(x)
#             categorical_post = Categorical(probs)
#             y = probs

#         #posterior param
#         x_new = torch.cat((x,y),dim=1).to(next(self.parameters()).device)
#         mu, log_var = self.encode(x_new)    


#         #posterior dist (force sigma to be positive with log) + reparametrization
#         post_dist = Normal(mu,  (0.5*log_var).exp())
#         z = post_dist.rsample()
        
#         #prepare input
#         z_cond = torch.cat((z,y),dim=1).to(next(self.parameters()).device)
#         # if not labelled: print(z_cond.shape)

        
#         #reconstruction -> log prob with sigmoid
#         rec = self.decode(z_cond)

        
#         #cateorical prior always constant
#         CAT_prior = Categorical(torch.ones(self.num_classes, device=next(self.parameters()).device))
#         categorical_prior = CAT_prior.log_prob(torch.ones(x.shape[0], device=next(self.parameters()).device))

#         L, diagnostics = self.L_(x, mu, log_var, z, rec, categorical_prior)
        
        
#         if labelled:
#             loss = -L.mean()
#             return [loss, diagnostics, z, rec]
        
#         elif not labelled:
#             U, diagnostics = self.U_(L, diagnostics, categorical_post)
#             loss = - U.mean()
#             return [loss, diagnostics, z, rec]


class M2(VAE):
    '''
    Beta Variational Auto-Encoder that is built upon the Encoder-Decor class, plus a Classifier as specified in the paper "Deep Generative" from Kingma 2014;
    '''

    def __init__(self,enc_layers,dec_layers,latent_features,dec_num_output, alpha, beta, layer_classifier, num_classes,activation="ReLU", batchnorm = False, dropout:float = None,activation_classifier="ReLU", batchnorm_classifier = False, dropout_classifier:float = None):
        super().__init__(enc_layers,dec_layers,latent_features,dec_num_output, beta, activation, batchnorm, dropout) 
        self.num_classes=num_classes
        self.classifier = FFNN(layer_classifier, self.num_classes,activation_classifier, batchnorm_classifier, dropout_classifier)
        self.latent_features = latent_features
        self.alpha = alpha

    def elbo(self):
        raise NotImplementedError("Old method without distinction between labelled and unlabelled")

    def conditional_sample(self, y:int, n:int):
    #generation of image = sampling from posterior (I believe) + decode
    # torch.rand(n, self.latent_features)
        z = torch.randn(n, self.latent_features)
        y = F.one_hot(torch.tensor(y),self.num_classes).expand(n,-1)
        z = torch.cat((z,y),dim=1).to(next(self.parameters()).device)
        return self.decode(z)
    
    def classify(self,x):
        probs =  F.softmax(self.classifier(x), dim=-1)
        return probs

    def L_(self, x, mu, log_var, z, rec, cat_prior):
        #loss function = reconstruction error + KL-divergence + categorical prior
        BCE = - torch.sum(F.binary_cross_entropy(rec, x, reduction="none"),dim = 1) #mean /none 
        KL_analyt = (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(),dim=1)) # analytical KL
        
        #Elbo computation
        L = BCE + cat_prior - self.beta * KL_analyt


        with torch.no_grad():
            diagnostics = {'elbo': L, 'likelihood': BCE, 'KL': KL_analyt, "cat_prior": cat_prior}
            
        return L, diagnostics


    def U_(self, L, diagnostics, categorical_post):
        #classifier loss
        # H = -torch.sum(torch.mul(probs, torch.log(probs + 1e-8)), dim=-1)
        H = categorical_post.entropy() #size=batch
        L_unlabelled = torch.sum(torch.mul(categorical_post.probs, L.view(-1,1)),dim=1)
        U = L_unlabelled + H
        with torch.no_grad():
            diagnostics["Entropy H"] = H

            
        return U, diagnostics
        
    
    def forward(self,x,y=None):
        
        probs = self.classify(x)
        categorical_post = Categorical(probs)
        
        labelled = False if y is None else True

        if labelled:
            ys = F.one_hot(y,num_classes=self.num_classes).to(next(self.parameters()).device)

        elif not labelled:
            ys = probs

        #posterior param
        # x_new = torch.cat((x,ys),dim=1).to(next(self.parameters()).device)
        z, mu, log_var = self.encode(x)    


        #prepare input
        z_cond = torch.cat((z,ys),dim=1).to(next(self.parameters()).device)
       
        #reconstruction -> log prob with sigmoid
        rec = self.decode(z_cond)

        
        #cateorical prior always constant
        CAT_prior = Categorical(torch.ones(self.num_classes, device=next(self.parameters()).device))
        categorical_prior = CAT_prior.log_prob(torch.ones(x.shape[0], device=next(self.parameters()).device))

        L, diagnostics = self.L_(x, mu, log_var, z, rec, categorical_prior)
        
        
        if labelled:
            # class_loss = F.cross_entropy(self.classifier(x),y)
            class_loss = - categorical_post.log_prob(y)
            diagnostics["classifier_loss"] = class_loss.mean().item()
            loss = (-L + self.alpha*class_loss).mean()

            return [loss, diagnostics, z, rec]
        
        elif not labelled:
            U, diagnostics = self.U_(L, diagnostics, categorical_post)
            loss = - U.mean()
            return [loss, diagnostics, z, rec]


class M2_stacked(M2):
    '''
    Beta Variational Auto-Encoder that is built upon the Encoder-Decor class, plus a Classifier as specified in the paper "Deep Generative" from Kingma 2014;
    '''
    def __init__(self, M1_model, enc_layers,dec_layers,latent_features,dec_num_output,beta, layer_classifier, num_classes,activation="ReLU", batchnorm = False, dropout:float = None,activation_classifier="ReLU", batchnorm_classifier = False, dropout_classifier:float = None):
        super().__init__(enc_layers,dec_layers,latent_features,dec_num_output,beta, layer_classifier, num_classes,activation, batchnorm, dropout,activation_classifier, batchnorm_classifier , dropout_classifier) 
        
        self.m1_model = M1_model
        self.m1_model.train(False)

        for param in self.m1_model.parameters():
            param.requires_grad = False

    def embedding(self,x):
        # encoding from z-batch, must return 2 parameter [mu,sigma],
        x, _,_ = self.m1_model.encoder(x)
        # extract mu and sigma from encoder result
        return x

    def classify(self,x):
        probs =  F.softmax(self.classifier(x), dim=-1)
        return probs
    
    def L_(self, x, mu, log_var, z, rec, cat_prior):
        #loss function = reconstruction error + KL-divergence + categorical prior
        BCE = - torch.sum(F.binary_cross_entropy(rec, x, reduction="none"),dim = 1) #mean /none 
        KL_analyt = (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(),dim=1)) # analytical KL
        
        #Elbo computation
        L = BCE + cat_prior - self.beta * KL_analyt


        with torch.no_grad():
            diagnostics = {'elbo': L, 'likelihood': BCE, 'KL': KL_analyt, "cat_prior": cat_prior}
            
        return L, diagnostics

    def forward(self,x,y=None):
        labelled = False if y is None else True

        if labelled:
            y = F.one_hot(y,num_classes=self.num_classes).to(next(self.parameters()).device)

        elif not labelled:
            probs = self.classify(x)
            categorical_post = Categorical(probs)
            y = probs

        # obtain embedding from M1
        z_1 = self.embedding(x) 
        
        #posterior param
        z_2, mu, log_var = self.encode(z_1)    


        
        #prepare input
        z_cond = torch.cat((z_2,y),dim=1).to(next(self.parameters()).device)
        # if not labelled: print(z_cond.shape)

        
        #reconstruction -> log prob with sigmoid
        rec = self.decode(z_cond)

        
        #cateorical prior always constant
        CAT_prior = Categorical(torch.ones(self.num_classes, device=next(self.parameters()).device))
        categorical_prior = CAT_prior.log_prob(torch.ones(x.shape[0], device=next(self.parameters()).device))

        L, diagnostics = self.L_(x, mu, log_var, z_2, rec, categorical_prior)
        
        
        if labelled:
            loss = -L.mean()
            return [loss, diagnostics, z_1, rec]
        
        elif not labelled:
            U, diagnostics = self.U_(L, diagnostics, categorical_post)
            loss = - U.mean()
            return [loss, diagnostics, z_1, rec]