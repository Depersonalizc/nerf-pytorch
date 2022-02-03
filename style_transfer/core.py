# Style Transfer
import copy

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision import models
from torchvision import transforms


class ImageNetNormalize(nn.Module):
  """Module which normalizes inputs using the ImageNet mean and stddev."""

  def __init__(self):
    super().__init__()
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    self.register_buffer('_mean', mean)
    self.register_buffer('_std', std)

  def forward(self, input):
    return (input - self._mean.to(input.device)) / self._std.to(input.device)

class ContentLoss(nn.Module):
  """The content loss module.
  
  Computes the L1 loss between the target and the input.
  """

  def __init__(self, target):
    """Initializes a new ContentLoss instance.
    
    Args:
      target: Take the L1 loss with respect to this target.
    """
    super().__init__()
    # Detach target since we do not want to use it for gradient computation.
    self.update_target(target)
    self.loss = None

  def forward(self, input):
    self.loss = F.l1_loss(input, self._target)
    return input

  def update_target(self, new_target):
    self._target = new_target.detach()


class StyleLoss(nn.Module):
  """The style loss module.
  
  Computes the L1 loss between the gram matricies of the target feature and the
  input.
  """

  def __init__(self, target_feature):
    """Initializes a new StyleLoss instance.
    
    Args:
      target_feature: Take the L1 loss with respect to this target feature.
    """
    super().__init__()
    # Detach target since we do not want to use it for gradient computation.
    self._target = self._gram_matrix(target_feature.detach()).detach()
    self.loss = None

  def _gram_matrix(self, input):
    """Returns the normalized Gram matrix of the input."""
    n, c, w, h = input.size()
    features = input.view(n * c, w * h)
    G = torch.mm(features, features.t())
    return G.div(n * c * w * h)

  def forward(self, input):
    G = self._gram_matrix(input)
    self.loss = F.l1_loss(G, self._target)
    return input


# TODO(eugenhotaj): This function also replaces the model's ReLU and Pooling
# layers. Therefore, the name of this method is misleading. We should either
# rename the method to something more appropriate, or move the replacement logic
# to get_style_model_and_losses.
def rename_vgg_layers(model):
  """Renames VGG model layers to match those in the paper."""
  block, number = 1, 1
  renamed = nn.Sequential()
  for layer in model.children():
    if isinstance(layer, nn.Conv2d):
      name = f'conv{block}_{number}'
    elif isinstance(layer, nn.ReLU):
      name = f'relu{block}_{number}'
      # The inplace ReLU version doesn't play nicely with NST.
      layer = nn.ReLU(inplace=False)
      number += 1
    elif isinstance(layer, nn.MaxPool2d):
      name = f'pool_{block}'
      # Average pooling was found to generate images of higher quality than
      # max pooling by Gatys et al.
      layer = nn.AvgPool2d(layer.kernel_size, layer.stride)
      block += 1
      number = 1
    else:
      raise RuntimeError(f'Unrecognized layer "{layer.__class__.__name__}""') 
    renamed.add_module(name, layer)
  return renamed


def init_nst_model_and_losses(
    model, style_imgs, content_layers, style_layers):
  """Creates the Neural Style Transfer model and losses. 

  We assume the model was pretrained on ImageNet and normalize all inputs using
  the ImageNet mean and stddev.
  We also assume style image is not changed over the course of optimization

  Args:
    model: The model to use for Neural Style Transfer. ContentLoss and StyleLoss
      modules will be inserted after each layer in content_layers and 
      style_layers respectively.
    style_imgs: The list of style images to use when creating the StyleLosses.
    content_layers: The name of the layers after which a ContentLoss module will
      be inserted.
    style_layers: The name of the layers after which a StyleLoss module will be
      inserted.
  Returns: A three item tuple of the NST model with ContentLoss and StyleLoss 
    modules inserted, the ContentLosses modules, and the StyleLosses modules.
  """
  nst_model = nn.Sequential(ImageNetNormalize())
  content_losses, style_losses, last_layer = [], [], 0
  for i, (name, layer) in enumerate(model.named_children()):
    nst_model.add_module(name, layer)
    if name in content_layers:
      content_loss = ContentLoss(nst_model(torch.zeros([1, 3, 224, 224])))
      nst_model.add_module(f'ContentLoss_{name}', content_loss)
      content_losses.append(content_loss)
      last_layer = i
    if name in style_layers:
      for j, style_img in enumerate(style_imgs):
        style_loss = StyleLoss(nst_model(style_img))
        nst_model.add_module(f'StyleLoss_{j}_{name}', style_loss)
        style_losses.append(style_loss)
        last_layer = i + j
  # Sanity check that we have the desired number of style and content layers.
  assert len(content_losses) == len(content_layers), 'Not all content layers found.'
  assert len(style_losses) / len(style_imgs) == len(style_layers), 'Not all style layers found.'
  # Remove the layers after the last StyleLoss and ContentLoss since they will
  # not be used for style transfer. To get the correct last_layer index, we 
  # take into account the ImageNetNormalization layer at the front and the
  # ContentLoss and StyleLoss layers.
  last_layer += 1 + len(content_losses) + len(style_losses)
  nst_model = nst_model[:last_layer+1]
  return nst_model, content_losses, style_losses 


class NST_VGG19(nn.Module):
  def __init__(self, style_imgs, content_layers, style_layers):
    super().__init__()
    vgg_model = models.vgg19(pretrained=True).features
    vgg_model = rename_vgg_layers(vgg_model)
    self.vgg_model, self.content_losses, self.style_losses = \
        init_nst_model_and_losses(
            vgg_model, style_imgs, content_layers, style_layers
        )
    # self.content_layers = content_layers
    # self.style_layers = style_layers
    self.eval()
  
  def forward(self, x):
    # Got style and content target
    return self.vgg_model(x)

  def update_content(self, new_content):
    # feed new content image into the model
    # and update all content targets
    with torch.no_grad():
      x = new_content
      for i, (name, layer) in enumerate(self.vgg_model.named_children()):
        # if name.startswith('Content'):
        if isinstance(layer, ContentLoss):
          layer.update_target(x)
          break
        else:
          x = layer(x)



