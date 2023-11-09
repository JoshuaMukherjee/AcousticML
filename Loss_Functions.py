import torch
from Gorkov import gorkov_autograd, gorkov_fin_diff
from Utilities import add_lev_sig

def max_loss(pressure, true):
  if len(true.shape) > 1:
    m = torch.max((true-pressure)**2,1).values
  else:
    m = torch.max((true-pressure)**2,0).values
  return torch.sum(m)

def mse_loss(expected, found):
  return torch.nn.MSELoss()(expected,found)

def mse_loss_print(expected, found):
  l = torch.nn.MSELoss()(expected,found)
  return l

def mean_std(output,alpha=0.01):
  if len(output.shape) > 1:
    dim = 1
  else:
    dim = 0
  m = -1 * (torch.mean(output,dim) - alpha*torch.std(output,dim) )
  return torch.sum(m,0)

def l1Loss(expected, found):
  return torch.nn.L1Loss()(expected,found)
  

def cosine_accuracy(target, output):
  """
  From Deep learning-based framework for fast and accurate acoustic hologram generation
  """
  def bottom(mat):
    return torch.sqrt(torch.sum(torch.square(mat)))
  batch = target.shape[0]
  return 1 - (torch.sum(torch.bmm(target.view(batch, 1, -1),output.view(batch, -1, 1))) / (bottom(target) * bottom(output)))

def mean_cosine_similarity(target, output, **loss_params):
  cos = torch.nn.CosineSimilarity(**loss_params)
  loss = cos(target, output)
  return torch.mean(loss)

def log_pressure(output):
  return torch.mean(torch.log(output**-1))

def cos_log(target,output,alpha=0.1,**cos_loss_params):
  """
  From Deep learning-based framework for fast and accurate acoustic hologram generation
  """
  return mean_cosine_similarity(target,output,**cos_loss_params) + alpha*log_pressure(output)

def cos_mean(target,output,alpha=0.1,**cos_loss_params):
  return mean_cosine_similarity(target,output,**cos_loss_params) - alpha*torch.mean(target)

def max_mean_pressure(output):
  return -1 * torch.mean(output)

def max_mean_min_pressure(output):
  # Bx4
  return torch.mean(-1 * torch.min(output,dim=1).values)

def max_min(output, max_fist_N, alpha=1):
  # BxN
  to_max = output[:,0:max_fist_N]
  to_min = output[:,max_fist_N:]
  # print(to_max)
  # print(to_min)
  loss = torch.mean(to_min) - alpha*torch.mean(to_max)
  return loss

def holo_cos_pressure_max(activation_out,target,field,target_pressure,alpha=1,**loss_params):
  holo = mean_cosine_similarity(target,activation_out)
  press = mse_loss(target_pressure,field)
  # print(holo,alpha*press)

  return holo + alpha*press

def AcousNetLoss(output, target, **params):
  l = 1 - torch.cos(2*torch.pi * (output - target))
  loss  = torch.sum(l)
  return loss

def cos_cos_loss(activation_out,target,field,target_pressure,alpha=1,**loss_params):
  holo = cosine_accuracy(target, activation_out)
  press = cosine_accuracy(field,target_pressure)
  # print(holo,alpha*press)

  return torch.sum(holo+alpha*press)

def cos_cos_max_pressure(activation_out,target,field,target_pressure,alpha=1,beta=1,**loss_params):
    holo = cosine_accuracy(target, activation_out)
    press = cosine_accuracy(field,target_pressure)
    max_p = torch.min(field)
    # print(field)
    # print(beta, max_p)
    # print(holo,alpha*press, beta*max_p)

    return torch.sum(holo+alpha*press-beta*max_p)

def cos_cos_log_pressure(activation_out,target,field,target_pressure,alpha=1,beta=1,**loss_params):
    holo = cosine_accuracy(target, activation_out)
    press = cosine_accuracy(field,target_pressure)
    max_p = log_pressure(field)
    # print(field)
    # print(beta, max_p)
    # print(holo,alpha*press, beta*max_p)

    return torch.sum(holo+alpha*press+beta*max_p)

def cos_cos_mean_pressure(activation_out,target,field,target_pressure,alpha=1,beta=1,**loss_params):
    holo = cosine_accuracy(target, activation_out)
    press = cosine_accuracy(field,target_pressure)
    max_p = max_mean_pressure(field)
    # print(field)
    # print(beta, max_p)
    # print(holo,alpha*press, beta*max_p)

    return torch.sum(holo+alpha*press+beta*max_p)

def gorkov_loss(activation, points):
  activation = add_lev_sig(activation)
  gorkov = gorkov_autograd(activation,points,retain_graph=True)
  return torch.sum(gorkov)

def gorkov_FD_loss(activation, points, axis="XYZ",stepsize = 0.000135156253,K1=None, K2=None):
  activation = add_lev_sig(activation)
  U = gorkov_fin_diff(activation,points,axis=axis,stepsize=stepsize,K1=K1,K2=K2)
  return torch.sum(U)

def gorkov_FD_mean_loss(activation, points, axis="XYZ",stepsize = 0.000135156253,K1=None, K2=None):
  activation = add_lev_sig(activation)
  U = gorkov_fin_diff(activation,points,axis=axis,stepsize=stepsize,K1=K1,K2=K2)
  return torch.mean(U)

def gorkov_FD_maxsum_loss(activation, points, axis="XYZ",stepsize = 0.000135156253,K1=None, K2=None):
  activation = add_lev_sig(activation)
  U = gorkov_fin_diff(activation,points,axis=axis,stepsize=stepsize,K1=K1,K2=K2)
  return torch.sum(torch.max(U,dim=1).values)

def gorkov_FD_maxmean_loss(activation, points, axis="XYZ",stepsize = 0.000135156253,K1=None, K2=None):
  activation = add_lev_sig(activation)
  U = gorkov_fin_diff(activation,points,axis=axis,stepsize=stepsize,K1=K1,K2=K2)
  return torch.mean(torch.max(U,dim=1).values)


if __name__ == "__main__":
  output = torch.Tensor([[9000,9000,3000,4000],[1000,8000,1000,4000]])
  print(max_min(output,2))


