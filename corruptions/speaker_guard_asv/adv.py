import torch
import numpy as np

from .asv_wrapper import ASVWrapper
from .white_box import FGSM, PGD, CW2, CWinf
from .black_box import FAKEBOB, SirenAttack

class AbsAdvAttack(torch.nn.Module):

    def __init__(self, model, config, attack=None, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.device = config["device"]
        self.attack_config = config["attack"]
        self.classifier = ASVWrapper(model, torch.nn.functional.cosine_similarity, 
                                     config["threshold"], config["device"])

        if attack:
            self.set_attack(attack)
            self.set_name()

    def set_attack(self, attack):
        self.attack = attack(self.classifier, task="SV",
                             **self.attack_config)

    def set_name(self):
        self.name = attack.__name__

    def __repr__(self):
        return f"AdversarialAttack({self.name})"

    def forward(self, x, x_tgt, y):
        x = x.unsqueeze(dim=1).to(self.device)
        x_tgt = x_tgt.unsqueeze(dim=1).to(self.device)
        self.attack.model.set_tgt(x_tgt.squeeze(dim=1))
        x_adv = self.attack.attack(x, x_tgt, torch.LongTensor([y]))
        return x_adv

class FGSMAttack(AbsAdvAttack):

    def __init__(self, model, config, **kwargs):
        super().__init__(model, config, **kwargs)
        self.eps = self.attack_config["epsilon"]
        self.set_name()
        self.set_attack()

    def set_name(self):
        self.name = "fgsm-"+str(self.eps)

    def set_attack(self):
        self.attack = FGSM(self.classifier, task="SV", 
                           loss="Margin", **self.attack_config)

class PGDAttack(AbsAdvAttack):

    def __init__(self, model, config, **kwargs):
        super().__init__(model, config, **kwargs)
        self.eps = self.attack_config["epsilon"]
        self.max_iter = self.attack_config["max_iter"]
        self.set_name()
        self.set_attack()

    def set_name(self):
        self.name = "pgd-"+str(self.eps)+"-"+str(self.max_iter)

    def set_attack(self):
        self.attack = PGD(self.classifier, task="SV", 
                          loss="Margin", **self.attack_config)

class CW2Attack(AbsAdvAttack):

    def __init__(self, model, config, **kwargs):
        super().__init__(model, config, **kwargs)
        self.confidence = self.attack_config["confidence"]
        self.max_iter = self.attack_config["max_iter"]
        self.set_name()
        self.set_attack()

    def set_name(self):
        self.name = "cw-l2-"+str(self.confidence)+"-"+str(self.max_iter)

    def set_attack(self):
        self.attack = CW2(self.classifier, task="SV",
                          **self.attack_config)

class CWInfAttack(AbsAdvAttack):

    def __init__(self, model, config, **kwargs):
        super().__init__(model, config, **kwargs)
        self.epsilon = self.attack_config["epsilon"]
        self.max_iter = self.attack_config["max_iter"]
        self.set_name()
        self.set_attack()

    def set_name(self):
        self.name = "cw-linf-"+str(self.epsilon)+"-"+str(self.max_iter)

    def set_attack(self):
        self.attack = CWinf(self.classifier, task="SV", 
                            loss="Margin", **self.attack_config)

class BlackFakeBobAttack(AbsAdvAttack):

    def __init__(self, model, config, **kwargs):
        super().__init__(model, config, **kwargs)
        self.epsilon = self.attack_config["epsilon"]
        self.max_iter = self.attack_config["max_iter"]
        self.set_name()
        self.set_attack()

    def set_name(self):
        self.name = "fkbob-"+str(self.epsilon)+"-"+str(self.max_iter)

    def set_attack(self):
        self.attack = FAKEBOB(self.classifier, task="SV",
                              **self.attack_config)

class BlackSirenAttack(AbsAdvAttack):

    def __init__(self, model, config, **kwargs):
        super().__init__(model, config, **kwargs)
        self.eps = self.attack_config["epsilon"]
        self.max_iter = self.attack_config["max_iter"]
        self.set_name()
        self.set_attack()

    def set_name(self):
        self.name = "siren-"+str(self.eps)+"-"+str(self.max_iter)

    def set_attack(self):
        self.attack = SirenAttack(self.classifier, task="SV",
                                  **self.attack_config)

