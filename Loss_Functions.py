import torch

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

def log_pressure(target, output):
  return torch.mean(torch.log(output**-1))

def cos_log(target,output,alpha=0.1,**cos_loss_params):
  """
  From Deep learning-based framework for fast and accurate acoustic hologram generation
  """
  return mean_cosine_similarity(target,output,**cos_loss_params) + alpha*log_pressure(target,output)

def cos_mean(target,output,alpha=0.1,**cos_loss_params):
  return mean_cosine_similarity(target,output,**cos_loss_params) - alpha*torch.mean(target)