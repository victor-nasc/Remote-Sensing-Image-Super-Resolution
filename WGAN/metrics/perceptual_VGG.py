import torch.nn as nn
import torchvision.models as models


class PerceptualVGG(nn.Module):
    def __init__(self, layers=['relu_1', 'relu_2', 'relu_3']):
        super(PerceptualVGG, self).__init__()
        
        vgg = models.vgg19(pretrained=True).features
        
        # freeze the VGG model
        for param in vgg.parameters():
            param.requires_grad = False
        
        self.selected_layers = layers
        self.vgg = nn.Sequential()
        self.layer_map = {}
        
        layer_counter = 0
        for i, layer in enumerate(vgg):
            if isinstance(layer, nn.ReLU):
                layer = nn.ReLU(inplace=False)  # turn off in-place to avoid backpropagation issues
            self.vgg.add_module(str(i), layer)
            
            # store layers
            layer_name = f"relu_{layer_counter}" if isinstance(layer, nn.ReLU) else None
            if layer_name:
                self.layer_map[layer_name] = i
                layer_counter += 1

        # keep only the selected layers
        self.layer_ids = [self.layer_map[layer] for layer in self.selected_layers]
        
    def forward(self, x):
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if int(name) in self.layer_ids:
                features[self.layer_ids.index(int(name))] = x
        return features