import torch
import torch.nn as nn
import torchvision.models as models


# Custom VGG19 model architecture
class CustomAlexNet(nn.Module):
    def __init__(self, input_channels=1, pretrained=False, num_classes=1):
        super(CustomAlexNet, self).__init__()
        self.model = models.alexnet(pretrained=pretrained)
        if input_channels < 3:
            self.model.features[0] = nn.Conv2d(input_channels, 64, kernel_size=11, stride=4, padding=2)
        num_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


# Custom VGG19 model architecture
class CustomVGG19(nn.Module):
    def __init__(self, input_channels=1, pretrained=False, num_classes=1):
        super(CustomVGG19, self).__init__()
        self.model = models.vgg19(pretrained=pretrained)
        if input_channels < 3:
            self.model.features[0] = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        num_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


# Custom ResNet152 model architecture
class CustomResNet152(nn.Module):
    def __init__(self, input_channels=1, pretrained=False, num_classes=1):
        super(CustomResNet152, self).__init__()
        self.model = models.resnet152(pretrained=pretrained)
        if input_channels < 3:
            self.model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


# Custom Inception model architecture
class CustomInception(nn.Module):
    def __init__(self, input_channels=1, pretrained=False, num_classes=1):
        super(CustomInception, self).__init__()
        self.model = models.inception_v3(pretrained=pretrained)
        self.model.aux_logits = False
        if input_channels != 3:
            self.model.transform_input = False
            self.model.Conv2d_1a_3x3.conv = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, bias=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


# Custom DenseNet121 model architecture
class CustomDenseNet121(nn.Module):
    def __init__(self, input_channels=1, pretrained=False, num_classes=1):
        super(CustomDenseNet121, self).__init__()
        self.model = models.densenet121(pretrained=pretrained)
        if input_channels < 3:
            self.model.features.conv0 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


# Two-stream VGG19 model architecture
class TwoStreamVGG(nn.Module):
    def __init__(self, pretrained=False, num_classes=1):
        super(TwoStreamVGG, self).__init__()

        # Load pretrained vgg19
        self.vgg_nir = models.vgg19(pretrained=pretrained)
        self.vgg_rgb = models.vgg19(pretrained=pretrained)

        # Modify the nir model
        self.vgg_nir.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        num_nir_features =  self.vgg_nir.classifier[-1].in_features
        self.vgg_nir.classifier[-1] = nn.Linear(num_nir_features, 4096)

        # Modify the rgb model
        num_rgb_features =  self.vgg_rgb.classifier[-1].in_features
        self.vgg_rgb.classifier[-1] = nn.Linear(num_rgb_features, 4096)

        # Final regression layer
        self.regression_layer = nn.Sequential(
            nn.Linear(4096*2, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, nir_input, rgb_input):
        # Forward pass for NIR stream
        nir_features = self.vgg_nir(nir_input)

        # Forward pass for RGB stream
        rgb_features = self.vgg_rgb(rgb_input)

        # Concatenate RGB and NIR features
        combined_features = torch.cat((rgb_features, nir_features), dim=1)

        # Final regression layer
        output = self.regression_layer(combined_features)

        return output
    

# Two-stream ResNet152 model architecture
class TwoStreamResNet(nn.Module):
    def __init__(self, pretrained=False, num_classes=1):
        super(TwoStreamResNet, self).__init__()

        # Load pretrained resnet152
        self.resnet_nir = models.resnet152(pretrained=pretrained)
        self.resnet_rgb = models.resnet152(pretrained=pretrained)

        # Modify the nir model
        self.resnet_nir.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_nir_features = self.resnet_nir.fc.in_features
        self.resnet_nir.fc = nn.Linear(num_nir_features, 2048)

        # Modify the rgb model
        num_rgb_features = self.resnet_rgb.fc.in_features
        self.resnet_rgb.fc = nn.Linear(num_rgb_features, 2048)

        # Final regression layer
        self.regression_layer = nn.Sequential(
            nn.Linear(2048*2, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_classes)
        )

    def forward(self, nir_input, rgb_input):
        # Forward pass for NIR stream
        nir_features = self.resnet_nir(nir_input)

        # Forward pass for RGB stream
        rgb_features = self.resnet_rgb(rgb_input)

        # Concatenate RGB and NIR features
        combined_features = torch.cat((rgb_features, nir_features), dim=1)

        # Final regression layer
        output = self.regression_layer(combined_features)

        return output
    

# Two-stream Inception model architecture
class TwoStreamInception(nn.Module):
    def __init__(self, pretrained=False, num_classes=1):
        super(TwoStreamInception, self).__init__()

        # Load pretrained inception_v3
        self.inception_nir = models.inception_v3(pretrained=pretrained)
        self.inception_rgb = models.inception_v3(pretrained=pretrained)

        # Modify the nir model
        self.inception_nir.transform_input = False
        self.inception_nir.aux_logits = False 
        self.inception_nir.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False)
        num_nir_features = self.inception_nir.fc.in_features
        self.inception_nir.fc = nn.Linear(num_nir_features, 2048)

        # Modify the rgb model
        self.inception_rgb.aux_logits = False 
        num_rgb_features = self.inception_rgb.fc.in_features
        self.inception_rgb.fc = nn.Linear(num_rgb_features, 2048)

        # Final regression layer
        self.regression_layer = nn.Sequential(
            nn.Linear(2048*2, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_classes)
        )

    def forward(self, nir_input, rgb_input):
        # Forward pass for NIR stream
        nir_features = self.inception_nir(nir_input)

        # Forward pass for RGB stream
        rgb_features = self.inception_rgb(rgb_input) 

        # Concatenate RGB and NIR features
        combined_features = torch.cat((rgb_features, nir_features), dim=1)

        # Final regression layer
        output = self.regression_layer(combined_features)

        return output
    
# Two-stream DenseNet121 model architecture
class TwoStreamDenseNet(nn.Module):
    def __init__(self, pretrained=False, num_classes=1):
        super(TwoStreamDenseNet, self).__init__()

        # Load pretrained DenseNet121
        self.densenet_nir = models.densenet121(pretrained=pretrained)
        self.densenet_rgb = models.densenet121(pretrained=pretrained)

        # Modify the nir model
        self.densenet_nir.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_nir_features = self.densenet_nir.classifier.in_features
        self.densenet_nir.classifier = nn.Linear(num_nir_features, 1024)

        # Modify the rgb model
        num_rgb_features = self.densenet_rgb.classifier.in_features
        self.densenet_rgb.classifier = nn.Linear(num_rgb_features, 1024)

        # Final regression layer
        self.regression_layer = nn.Sequential(
            nn.Linear(1024*2, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, nir_input, rgb_input):
        # Forward pass for NIR stream
        nir_features = self.densenet_nir(nir_input)

        # Forward pass for RGB stream
        rgb_features = self.densenet_rgb(rgb_input)

        # Concatenate RGB and NIR features
        combined_features = torch.cat((rgb_features, nir_features), dim=1)

        # Final regression layer
        output = self.regression_layer(combined_features)

        return output


# Two-stream AlexNet model architecture
class TwoStreamAlexNet(nn.Module):
    def __init__(self, pretrained=False, num_classes=1):
        super(TwoStreamAlexNet, self).__init__()

        # NIR stream: Load AlexNet and modify the first layer for single-channel input
        self.alexnet_nir = models.alexnet(pretrained=pretrained)
        self.alexnet_nir.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        num_nir_features = self.alexnet_nir.classifier[-1].in_features
        self.alexnet_nir.classifier[-1] = nn.Linear(num_nir_features, 4096)

        # RGB stream: Load AlexNet as is for three-channel input
        self.alexnet_rgb = models.alexnet(pretrained=pretrained)
        num_rgb_features = self.alexnet_rgb.classifier[-1].in_features
        self.alexnet_rgb.classifier[-1] = nn.Linear(num_rgb_features, 4096)

        # Final regression layer
        self.regression_layer = nn.Sequential(
            nn.Linear(4096 * 2, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, nir_input, rgb_input):
        # Forward pass for NIR stream
        nir_features = self.alexnet_nir(nir_input)

        # Forward pass for RGB stream
        rgb_features = self.alexnet_rgb(rgb_input)

        # Concatenate RGB and NIR features
        combined_features = torch.cat((nir_features, rgb_features), dim=1)

        # Final regression layer
        output = self.regression_layer(combined_features)

        return output



class CustomViT(nn.Module):
    def __init__(self, input_channels, num_classes=1):
        super(CustomViT, self).__init__()

        self.vit = models.vit_b_16(weights=None)

        self.vit.conv_proj = nn.Conv2d(
            in_channels=input_channels,
            out_channels=self.vit.conv_proj.out_channels,
            kernel_size=self.vit.conv_proj.kernel_size,
            stride=self.vit.conv_proj.stride,
            padding=self.vit.conv_proj.padding,
            bias=False
        )

        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)

    def forward(self, x):
        x = self.vit(x)
        return x


class TwoStreamViT(nn.Module):
    def __init__(self, num_classes=1):
        super(TwoStreamViT, self).__init__()

        self.vit_nir = models.vit_b_16(weights=None)
        self.vit_rgb = models.vit_b_16(weights=None)

        self.vit_nir.conv_proj = nn.Conv2d(
            in_channels=1,
            out_channels=self.vit_nir.conv_proj.out_channels,
            kernel_size=self.vit_nir.conv_proj.kernel_size,
            stride=self.vit_nir.conv_proj.stride,
            padding=self.vit_nir.conv_proj.padding,
            bias=False
        )

        self.vit_nir.heads.head = nn.Linear(self.vit_nir.heads.head.in_features, 768)
        self.vit_rgb.heads.head = nn.Linear(self.vit_rgb.heads.head.in_features, 768)

        self.regression_layer = nn.Sequential(
            nn.Linear(768 * 2, 768),
            nn.ReLU(),
            nn.Linear(768, num_classes)
        )

    def forward(self, nir_input, rgb_input):
        nir_features = self.vit_nir(nir_input)
        rgb_features = self.vit_rgb(rgb_input)
        combined_features = torch.cat((nir_features, rgb_features), dim=1)
        output = self.regression_layer(combined_features)
        return output
    

# Custom ResNet152 model architecture
class DomainResNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=1, weight_path=None):
        super(DomainResNet, self).__init__()
        
        # Load ResNet152 model
        self.resnet = models.resnet152(weights=None)

        # Load domain-specific weights if provided
        if weight_path is not None:
            pretrained_dict = torch.load(weight_path)
            # Remove the final layer weights to avoid mismatches
            pretrained_dict.pop('fc.weight', None)
            pretrained_dict.pop('fc.bias', None)
            self.resnet.load_state_dict(pretrained_dict, strict=False)

        if input_channels < 3:
            self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
    

class TwoStreamDomainResNet(nn.Module):
    def __init__(self, num_classes=1, weight_path=None):
        super(TwoStreamDomainResNet, self).__init__()

        # Initialize separate ResNet152 models for NIR and RGB
        self.resnet_nir = models.resnet152(weights=None)
        self.resnet_rgb = models.resnet152(weights=None)

        # Load domain-specific weights if provided
        if weight_path is not None:
            pretrained_dict = torch.load(weight_path)
            # Remove the final layer weights to avoid mismatches
            pretrained_dict.pop('fc.weight', None)
            pretrained_dict.pop('fc.bias', None)
            self.resnet_nir.load_state_dict(pretrained_dict, strict=False)
            self.resnet_rgb.load_state_dict(pretrained_dict, strict=False)

        # Modify the NIR model for 1-channel input
        self.resnet_nir.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_nir_features = self.resnet_nir.fc.in_features
        self.resnet_nir.fc = nn.Linear(num_nir_features, 2048)

        # Modify the RGB model for 3-channel input
        num_rgb_features = self.resnet_rgb.fc.in_features
        self.resnet_rgb.fc = nn.Linear(num_rgb_features, 2048)

        # Final regression layer
        self.regression_layer = nn.Sequential(
            nn.Linear(2048 * 2, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_classes)
        )

    def forward(self, nir_input, rgb_input):
        # Forward pass for NIR stream
        nir_features = self.resnet_nir(nir_input)

        # Forward pass for RGB stream
        rgb_features = self.resnet_rgb(rgb_input)

        # Concatenate RGB and NIR features
        combined_features = torch.cat((rgb_features, nir_features), dim=1)

        # Final regression layer
        output = self.regression_layer(combined_features)

        return output


    


# Example Usage
if __name__ == "__main__":
    # Create dummy data for NIR and RGB inputs
    nir_input = torch.randn(1, 1, 224, 224)
    rgb_input = torch.randn(1, 3, 224, 224)

    # Initialize model
    model = TwoStreamAlexNet(pretrained=False, num_classes=1)

    # Forward pass
    output = model(nir_input, rgb_input)
    print(output)