import torch
import torch.nn as nn
import copy
import time

def fuse_conv_and_bn(conv, bn):
	#
	# init
	fusedconv = torch.nn.Conv2d(
		conv.in_channels,
		conv.out_channels,
		kernel_size=conv.kernel_size,
		stride=conv.stride,
		padding=conv.padding,
		bias=True
	)
	#
	# prepare filters
	w_conv = conv.weight.clone().view(conv.out_channels, -1)
	w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_var)))
	fusedconv.weight.copy_( torch.mm(w_bn, w_conv).view(fusedconv.weight.size()) )
	#
	# prepare spatial bias
	if conv.bias is not None:
		b_conv = conv.bias
	else:
		b_conv = torch.zeros( conv.weight.size(0) )
	b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
	fusedconv.bias.copy_( torch.matmul(w_bn, b_conv) + b_bn )
	#
	# we're done
	return fusedconv

def extract_conv_bn_layers(model, model_name):
  modules_to_fuse = []
  single_module = []
  for ith, info in enumerate(model.named_parameters()):
      name, param = info

      name = ".".join(name.split(".")[:-1])
      last_name = name.split(".")[-1]
      if "conv" in last_name and len(single_module) == 0:
          single_module.append(model_name  + "." + name)
      if ("bn" in last_name or "batch_norm" in last_name) and len(single_module)==1:
          single_module.append(model_name + "." + name)
          modules_to_fuse.append(single_module)
          single_module = []
  return modules_to_fuse

def fuse_bisenet1(model):
  fused_model = copy.deepcopy(model)
  model_name = "fused_model"
  modules_to_fuse = extract_conv_bn_layers(fused_model, model_name)

  for module in modules_to_fuse:
    conv_layer_name = module[0]
    bn_name = module[1]

    conv_layer_name1 = ".".join(module[0].split(".")[:-2]) + "[" + module[0].split(".")[-2] + "]" + "." + module[0].split(".")[-1]
    bn_name1 = ".".join(module[1].split(".")[:-2]) + "[" + module[1].split(".")[-2] + "]" + "." + module[1].split(".")[-1]

    try:
      conv_layer = eval(conv_layer_name)
      batchnorm_layer = eval(bn_name)
      setattr(eval(".".join(conv_layer_name.split(".")[:-1])), conv_layer_name.split(".")[-1], fuse_conv_and_bn(conv_layer, batchnorm_layer))
      setattr(eval(".".join(bn_name.split(".")[:-1])), bn_name.split(".")[-1], nn.Identity())
    
    except:
      conv_layer = eval(conv_layer_name1)
      batchnorm_layer = eval(bn_name1)

      setattr(eval(".".join(conv_layer_name1.split(".")[:-1])), conv_layer_name1.split(".")[-1], fuse_conv_and_bn(conv_layer, batchnorm_layer))
      setattr(eval(".".join(bn_name1.split(".")[:-1])), bn_name1.split(".")[-1], nn.Identity())
    

  fused_model.context_path.features.layer2[0].downsample = fuse_conv_and_bn(fused_model.context_path.features.layer2[0].downsample[0], fused_model.context_path.features.layer2[0].downsample[1])

  fused_model.context_path.features.layer3[0].downsample = fuse_conv_and_bn(fused_model.context_path.features.layer3[0].downsample[0], fused_model.context_path.features.layer3[0].downsample[1])

  fused_model.context_path.features.layer4[0].downsample = fuse_conv_and_bn(fused_model.context_path.features.layer4[0].downsample[0], fused_model.context_path.features.layer4[0].downsample[1])
  
  return fused_model

