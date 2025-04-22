from huggingface_hub import hf_hub_download
import torch

class Redimnet(torch.nn.Module):
    def __init__(self, model_name_or_path = 'IDRnD/ReDimNet', model_location = 'cpu', model_name = 'b6', train_type = 'ft_lm', dataset ='vox2', cache_dir=None):
        super(Redimnet, self).__init__()
        self.model_name_or_path = model_name_or_path
        self.cache_dir = cache_dir
        self.model_location = model_location
        self.train_type = train_type
        self.model_name = model_name
        self.dataset = dataset
    def forward(self, audio):
        redimnet_model = torch.hub.load(
            self.model_name_or_path,
            "ReDimNet",
            model_name=self.model_name,
            train_type=self.train_type,
            dataset=self.dataset,
        ).to(self.model_location)
        redimnet_model.eval()
        audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.model_location)
        with torch.no_grad():
            embeddings = redimnet_model(audio)
        return embeddings
