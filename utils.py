import torch
import torch.nn as nn

# Functions to check Conv2d and ConvTranspose2d output size
def convt_output_size(input_size, kernel_size, stride=1, padding=0, output_padding=0, dilation=1):
    return int((input_size-1)*stride - 2*padding + dilation*(kernel_size-1) + output_padding + 1)

def conv_output_size(input_size, kernel_size, stride=1, padding=0, dilation=1):
    output_size = (input_size + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1
    if not output_size.is_integer():
        print(f'Fractional output size: {output_size}')
    return int(output_size)

# Functions to check Conv2d and ConvTranspose2d architecture output size
def check_Conv2d_architecture(input_size, conv_layers):
    print(f'# In: {input_size} x {input_size}')
    for conv_layer in conv_layers:
        output_size = conv_output_size(input_size, *conv_layer)
        print(f"nn.Conv2d(..., {str(conv_layer).strip('()')}), # {output_size} x {output_size}")
        input_size = output_size
    print(f'# Out: {output_size} x {output_size}')

def check_ConvTranspose2d_architecture(input_size, convt_layers):
    print(f'# In: {input_size} x {input_size}')
    for convt_layer in convt_layers:
        output_size = convt_output_size(input_size, *convt_layer)
        print(f"nn.ConvTranspose2d(..., {str(convt_layer).strip('()')}), # {output_size} x {output_size}")
        input_size = output_size
    print(f'# Out: {output_size} x {output_size}')

# Function to compute gradient penalty
def gradient_penalty(critic, images_real, images_fake, device):
    # Initialize epsilon
    N, C, H, W = images_real.shape
    epsilon = torch.rand((N, 1, 1, 1)).repeat(1, C, H, W).to(device)
    
    # Create interpolated images
    images_mixed = images_real * epsilon + images_fake * (1 - epsilon)
    
    # Compute critic score for interpolated images
    outputs_mixed = critic(images_mixed)
    
    # Compute gradient of critic score
    grad_mixed = torch.autograd.grad(
        outputs=outputs_mixed,
        inputs=images_mixed,
        grad_outputs=torch.ones_like(outputs_mixed),
        create_graph=True,
        retain_graph=True,
    )[0].flatten(1)
    
    # Compute gradient penalty
    grad_norm = grad_mixed.norm(2, dim=1)
    grad_penalty = torch.mean((grad_norm - 1) ** 2)
    
    return grad_penalty