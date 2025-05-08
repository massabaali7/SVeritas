import torch
import numpy as np
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier

class ASVWrapper(torch.nn.Module):
    def __init__(self, model, score_fn):
        super().__init__()
        self.model = model
        self.model.eval()
        self.score_fn = score_fn

    def forward(self, x):
        
        # Split batch
        N = x.shape[0]
        half = N//2

        # Get test and enrollment utterances
        test_utts = x[:half]
        enroll_utts = x[half:]

        # Forward pass through model for enrollment w/o grad
        with torch.no_grad():
            enroll_embs = self.model(enroll_utts.squeeze())
            
        # Forward pass through model for test with grad
        test_embs = self.model(test_utts.squeeze())

        # Compute scores (e.g. cosine similarity)
        scores = self.score_fn(test_utts, enroll_utts)

        # Duplicate scores to match size of batch - check if this creates issues or not with ART
        scores = scores.repeat(2)
        return scores

class ASVLoss(torch.nn.Module):
    def __init__(self, threshold=0, margin=0.2):
        super().__init__()
        self.threshold = threshold
        self.margin = torch.tensor(margin)

    def forward(self, s, y=None):
        if y.numel() == 4:
            label = y[0][1]
        else:
            label = int(s[0] >= self.threshold)
            
        if label == 1:
            loss = torch.maximum(1-s, self.margin).mean(dim=-1)
        elif label == 0:
            loss = torch.maximum(1+s, self.margin).mean(dim=-1)
        return loss
                
class AbsAdvAttack(torch.nn.Module):

    def __init__(self, model, attack_config, targeted=False, attack=None, **kwargs):
        super().__init__(**kwargs)

        self.attack_config = attack_config
        self.targeted = targeted
        self.mask = np.array([[1.0],[0.0]])
        self.classifier = ASVWrapper(model, torch.nn.functional.cosine_similarity)

        if attack:
            self.set_attack(attack)
            self.set_name()

    def set_attack(self, attack):
        self.attack = attack(estimator=self.classifier,
                             targeted=self.targeted,
                             **self.attack_config)

    def set_name(self):
        self.name = attack.__name__

    def __repr__(self):
        return f"AdversarialAttack({self.name})"

    def forward(self, model, x, y=None):
        if y == 0:
            y = torch.zeros(x.shape[0])
        else:
            y = torch.ones(x.shape[0])

        x_adv = attack.generate(x=x.numpy(), y=y, mask=mask)
        return x_adv[0,:]

class PGD(AbsAdvAttack):

    def __init__(self, model, attack_config, targeted=False, **kwargs):
        super().__init__(model, attack_config, targeted, **kwargs)
        self.eps = attack_config["eps"]
        self.set_name()
        self.set_attack()

    def set_name(self):
        self.name = "pgd-"+str(self.eps)

    def set_attack(self):
        self.attack = ProjectedGradientDescentPyTorch(
                        estimator=self.classifier,
                        targeted=self.targeted,
                        **self.attack_config)

class FGSM(AbsAdvAttack):

    def __init__(self, model, attack_config, targeted=False, **kwargs):
        super().__init__(model, attack_config, targeted, **kwargs)
        self.eps = attack_config["eps"]
        self.set_name()
        self.set_attack()

    def set_name(self):
        self.name = "fgsm-"+str(self.eps)

    def set_attack(self)
        self.attack = FastGradientMethod(
                estimator=self.classifier,
                targeted=self.targeted,
                **self.attack_config)

