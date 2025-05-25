import os
import pdb
import torch
import torch.nn as nn


class GpuDataParallel(object):
    def __init__(self):
        self.gpu_list = []
        self.output_device = None

    def set_device(self, device):
        device = str(device).lower()  # Convert to lowercase for safety
        
        if device in ["none", "cpu"]:
            self.output_device = "cpu"
            self.gpu_list = []
            return

        if device == "cuda" and torch.cuda.is_available():
            self.gpu_list = [0]  # Use the first GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            self.output_device = self.gpu_list[0]
            self.occupy_gpu(self.gpu_list)
            return

        # Handling specific GPU IDs like "0,1,2"
        try:
            self.gpu_list = [int(i) for i in device.split(',')]
            os.environ["CUDA_VISIBLE_DEVICES"] = device
            self.output_device = self.gpu_list[0] if self.gpu_list else "cpu"
            self.occupy_gpu(self.gpu_list)
        except ValueError:
            print(f"Invalid device format: {device}. Using CPU instead.")
            self.output_device = "cpu"
            self.gpu_list = []

            
    def model_to_device(self, model):
        model = model.to(self.output_device)
        if len(self.gpu_list) > 1:  # Only wrap in DataParallel if multiple GPUs exist
            model = nn.DataParallel(
                model,
                device_ids=self.gpu_list,
                output_device=self.output_device
            )
        return model


    def data_to_device(self, data):
        if isinstance(data, torch.FloatTensor):
            return data.to(self.output_device)
        elif isinstance(data, torch.DoubleTensor):
            return data.float().to(self.output_device)
        elif isinstance(data, torch.ByteTensor):
            return data.long().to(self.output_device)
        elif isinstance(data, torch.LongTensor):
            return data.to(self.output_device)
        elif isinstance(data, list) or isinstance(data, tuple):
            return [self.data_to_device(d) for d in data]
        else:
            raise ValueError(data.shape, "Unknown Dtype: {}".format(data.dtype))

    def criterion_to_device(self, loss):
        return loss.to(self.output_device)

    def occupy_gpu(self, gpus=None):
        """
            make program appear on nvidia-smi.
        """
        if len(gpus) == 0:
            torch.zeros(1).cuda()
        else:
            gpus = [gpus] if isinstance(gpus, int) else list(gpus)
            for g in gpus:
                torch.zeros(1).cuda(g)
