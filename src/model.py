import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class AttentionModule(nn.Module):
    def __init__(self, L, D, K):
        super(AttentionModule, self).__init__()
        self.L = L
        self.D = D
        self.K = K

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )


    def forward(self, H):
        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        # A = F.softmax(A, dim=1)  # softmax over N
        return A
    
class GatedAttentionModule(nn.Module):
    def __init__(self, L, D, K):
        super(GatedAttentionModule, self).__init__()
        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, H):
        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        # A = F.softmax(A, dim=1)  # softmax over N
        return A



class TeacherBranch(nn.Module):
  def __init__(self, input_dims, latent_dims, attention_module, 
               num_classes=2, 
               activation_function=nn.Tanh):
    super().__init__()
    self.input_dims = input_dims
    self.L = latent_dims
    self.K = 1
    self.D = latent_dims
    self.attention_module = attention_module
    self.num_classes = num_classes
    
    self.bagNN = nn.Sequential(
        nn.Linear(self.input_dims, self.L),
        activation_function(),
        nn.Linear(self.L, self.L),
        activation_function(),
        nn.Linear(self.L, self.num_classes ),
    )
    self.initialize_weights()
      
  def forward(self, input, replaceAS=None):  
    if replaceAS is not None:
      attention_weights = F.softmax(replaceAS,dim=1)
    else:
      attention_weights = self.attention_module(input)
      attention_weights = F.softmax(attention_weights,dim=1)
    
    aggregated_instance = torch.mm(attention_weights, input)
    output = aggregated_instance.squeeze()
    output = self.bagNN(output)
    return output
  
  def initialize_weights(self):
      for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)

class StudentBranch(nn.Module):
  def __init__(self, input_dims, latent_dims, 
               num_classes=2, 
               activation_function=nn.Tanh):
    super().__init__()
    self.input_dims = input_dims
    self.L = latent_dims
    self.K = 1
    self.D = latent_dims
    self.num_classes = num_classes 
    
    self.instanceNN = nn.Sequential(
        nn.Linear(self.input_dims, self.L),
        activation_function(),
        nn.Linear(self.L, self.L),
        activation_function(),
        nn.Linear(self.L, self.num_classes )
      )
    self.initialize_weights()
  def forward(self, input):  
    NN_out = input
    output = self.instanceNN(NN_out)
    
    return output 
  
  def initialize_weights(self):
      for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)

class EncoderBranch(nn.Module):
  def __init__(self, proto_vae, output_dims, activation_function = nn.Tanh):
    super().__init__()
    self.proto_vae = proto_vae
    self.activation_function = activation_function
    self.output_dims = output_dims
    self.encoder_layer = nn.Sequential(
      nn.Linear(self.proto_vae.latent_dim, self.output_dims),
      activation_function(),
      nn.Linear(self.output_dims, self.output_dims),
      activation_function(),
      nn.Linear(self.output_dims, self.output_dims)
    )
    self.initialize_weights()
  def forward(self, input):
    with torch.no_grad():
      vae_latent = self.proto_vae.features(input)
      mu = vae_latent[:,:self.proto_vae.latent_dim]
      logVar = vae_latent[:,self.proto_vae.latent_dim:].clamp(np.log(1e-8), - np.log(1e-8))
      z = self.proto_vae.reparameterize(mu, logVar)
    
    encoded_vector = self.encoder_layer(z)
    return encoded_vector
  def initialize_weights(self):
    for m in self.encoder_layer.modules():
      if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
          nn.init.zeros_(m.bias.data)
### AENB
class AENB(nn.Module):
    def __init__(self, input_dim, latent_dim, device, hidden_layers, activation_function=nn.ReLU):
        super(AENB, self).__init__()
        self.device= device
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.epsilon = 1e-4
        self.hidden_layers = hidden_layers
        self.activation_function = activation_function

        
        feature_layers = []
        previous_dim = input_dim 
        for layer_dim in self.hidden_layers:
            feature_layers.append(nn.Linear(previous_dim, layer_dim))
            feature_layers.append(self.activation_function())
            # feature_layers.append(nn.BatchNorm1d(layer_dim))
            previous_dim = layer_dim
        feature_layers.append(nn.Linear(previous_dim, latent_dim))
        self.features = nn.Sequential(*feature_layers)
        
        decoder_layers = []
        for layer_dim in reversed(self.hidden_layers):
            decoder_layers.append(nn.Linear(previous_dim, layer_dim))
            decoder_layers.append(self.activation_function())
            # decoder_layers.append(nn.BatchNorm1d(layer_dim))
            previous_dim = layer_dim
        decoder_layers.append(nn.Linear(previous_dim, input_dim * 2))
        self.decoder_layers = nn.Sequential(*decoder_layers)

        self._initialize_weights()

    def decoder(self, z):
        decoded = self.decoder_layers(z)
        mu_recon = torch.exp(decoded[:, :self.input_dim]).clamp(1e-6, 1e6) 
        theta_recon = F.softplus(decoded[:, self.input_dim:]).clamp(1e-4, 1e4)  
        return mu_recon, theta_recon

    def forward(self, x, y=None, is_train=True):
        encoded_features = self.features(x)
        z = encoded_features
        mu_recon, theta_recon = self.decoder(z)


        return mu_recon, theta_recon
    
    def _initialize_weights(self):
        for m in self.features.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


        for m in self.decoder_layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)