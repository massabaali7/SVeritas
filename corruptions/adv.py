import torch
import numpy as np
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescentPyTorch, CarliniL2Method, CarliniLInfMethod, HopSkipJump, BoundaryAttack
from art.estimators.classification import PyTorchClassifier

class ASVWrapper(torch.nn.Module):
    def __init__(self, model, score_fn):
        super().__init__()
        self.model = model
        self.model.eval()
        self.score_fn = score_fn

    def set_target_sample(self, x):
        with torch.no_grad():
            self.tgt = self.model(x.squeeze(0))
        self.tgt_set = True

    def forward(self, x):    
        
        # Get test and enrollment utterances
        test_utts = x
        enroll_embs = self.tgt

        # Forward pass through model for test with grad
        test_embs = self.model(test_utts.squeeze())

        # Compute scores (e.g. cosine similarity)
        scores = self.score_fn(test_embs, enroll_embs)

        # Duplicate scores to match size of batch - check if this creates issues or not with ART
        return scores

class ASVLoss(torch.nn.Module):
    def __init__(self, threshold=0, loss_threshold=1.0, margin=0.2):
        super().__init__()
        self.threshold = threshold
        self.loss_threshold = loss_threshold
        self.margin = torch.tensor(margin)

    def forward(self, s, y=None):
        if y.numel() == 4:
            label = y[0][1]
        else:
            label = int(s[0] >= self.threshold)
            
        if label == 1:
            loss = torch.maximum(self.loss_threshold - s, self.margin).mean(dim=-1)
        elif label == 0:
            loss = torch.maximum(self.loss_threshold + s, self.margin).mean(dim=-1)
        return loss
                
class AbsAdvAttack(torch.nn.Module):

    def __init__(self, model, attack_config, targeted=False, attack=None, **kwargs):
        super().__init__(**kwargs)

        self.attack_config = attack_config
        self.targeted = targeted
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

    def forward(self, x, x_tgt, y=None):
        if y == 0:
            y = torch.zeros(x.shape[0])
        else:
            y = torch.ones(x.shape[0])
        self.classifier.set_target_sample(x_tgt)
        x_adv = attack.generate(x=x.numpy(), y=y)
        return x_adv

class PGD(AbsAdvAttack):

    def __init__(self, model, attack_config, targeted=False, **kwargs):
        super().__init__(model, attack_config, targeted, **kwargs)
        self.eps = attack_config["eps"]
        self.max_iter = attack_config["max_iter"]
        self.set_name()
        self.set_attack()

    def set_name(self):
        self.name = "pgd-"+str(self.eps)+"-"+str(self.max_iter)

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

class BIM(AbsAdvAttack):

    def __init__(self, model, attack_config, targeted=False, **kwargs):
        super().__init__(model, attack_config, targeted, **kwargs)
        self.eps = attack_config["eps"]
        self.max_iter = attack_config["max_iter"]
        self.set_name()
        self.set_attack()

    def set_name(self):
        self.name = "bim-"+str(self.eps)+"-"+str(self.max_iter)

    def set_attack(self)
        self.attack = FastGradientMethod(
                estimator=self.classifier,
                targeted=self.targeted,
                **self.attack_config)

class CWL2(AbsAdvAttack):

    def __init__(self, model, attack_config, targeted=False, **kwargs):
        super().__init__(model, attack_config, targeted, **kwargs)
        self.confidence = attack_config["confidence"]
        self.max_iter = attack_config["max_iter"]
        self.set_name()
        self.set_attack()

    def set_name(self):
        self.name = "cw-l2-"+str(self.confidence)+"-"+str(self.max_iter)

    def set_attack(self)
        self.attack = CarliniL2Method(
                classifier=self.classifier,
                targeted=self.targeted,
                **self.attack_config)

class CWLInf(AbsAdvAttack):

    def __init__(self, model, attack_config, targeted=False, **kwargs):
        super().__init__(model, attack_config, targeted, **kwargs)
        self.confidence = attack_config["confidence"]
        self.max_iter = attack_config["max_iter"]
        self.set_name()
        self.set_attack()

    def set_name(self):
        self.name = "cw-linf-"+str(self.confidence)+"-"+str(self.max_iter)

    def set_attack(self)
        self.attack = CarliniLInfMethod(
                classifier=self.classifier,
                targeted=self.targeted,
                **self.attack_config)


class BlackHopSkipJump(AbsAdvAttack):

    def __init__(self, model, attack_config, targeted=False, **kwargs):
        super().__init__(model, attack_config, targeted, **kwargs)
        self.max_iter = attack_config["max_iter"]
        self.set_name()
        self.set_attack()

    def set_name(self):
        self.name = "hsj-"+str(self.max_iter)

    def set_attack(self)
        self.attack = HopSkipJump(
                classifier=self.classifier,
                targeted=self.targeted,
                **self.attack_config)

class BlackBoundaryAttack(AbsAdvAttack):

    def __init__(self, model, attack_config, targeted=False, **kwargs):
        super().__init__(model, attack_config, targeted, **kwargs)
        self.eps = attack_config["eps"]
        self.max_iter = attack_config["max_iter"]
        self.set_name()
        self.set_attack()

    def set_name(self):
        self.name = "ba-"+str(self.eps)+"-"+str(self.max_iter)

    def set_attack(self)
        self.attack = BoundaryAttack(
                estimator=self.classifier,
                targeted=self.targeted,
                **self.attack_config)
