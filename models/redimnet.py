from huggingface_hub import hf_hub_download
import torch

class Redimnet(torch.nn.Module):
    def __init__(self, model_name_or_path = 'IDRnD/ReDimNet', model_location = 'cpu', model_name = 'b6', train_type = 'ft_lm', dataset ='vox2', cache_dir=None):
        super(Redimnet, self).__init__()
        self.cache_dir = cache_dir
        self.model_location = model_location
        self.redimnet_model = torch.hub.load(
            model_name_or_path,
            "ReDimNet",
            model_name=model_name,
            train_type=train_type,
            dataset=dataset,
        ).to(model_location)
        self.redimnet_model.eval()
    def forward(self, audio):
        audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.model_location)
        with torch.no_grad():
            embeddings = self.redimnet_model(audio)
        return embeddings
