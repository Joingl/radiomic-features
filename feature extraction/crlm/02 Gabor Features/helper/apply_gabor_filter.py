import torch
import pytorch_gabor as pygab

#Needs working installation of the pytorch_gabor package
class NetworkFreq(torch.nn.Module):
    def __init__(self, resolution, stride=(1, 1)):
        super().__init__()
        self.gwt = pygab.GaborFilterFrequency(in_channels=1, resolution=resolution, number_of_scales=3, number_of_directions=8, sigma=torch.pi/3, k_max=torch.pi/6, k_fac=2**(-0.5))
        self.grid = pygab.GridExtract(stride=stride)
        self.act = pygab.GaborFilterAct(out_type='abs', stack_abs_phase=True, normalize=False)

    def forward(self, img):
        img = self.act(self.grid(self.gwt(img)))

        return img

network = NetworkFreq(resolution=(512, 512), stride=(1, 1))

def format_tensor(tensor):
    tensor = tensor.unsqueeze(0)  # [B=1, C=1, H=300, W=300]
    tensor = tensor.unsqueeze(0)
    return tensor

def apply_filter(img):
    img_tensor = torch.from_numpy(img.astype('int16'))
    img_tensor = format_tensor(img_tensor)  # [B=1, C=1, H=300, W=300] are the dimensions the network needs
    res = network(img_tensor)[0]

    filtered_images = res.numpy()
    return filtered_images