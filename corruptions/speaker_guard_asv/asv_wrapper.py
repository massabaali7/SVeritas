import torch

class ASVWrapper(torch.nn.Module):
    def __init__(self, model, score_fn, threshold, device):
        super().__init__()
        self.model = model
        self.model.eval()
        self.device = device
        self.score_fn = score_fn
        self.threshold = threshold
        self.tgt_set = False

    def set_tgt(self, x):
        with torch.no_grad():
            enroll_embs = self.model(x)
        self.enroll_embs = enroll_embs
        self.tgt_set = True

    def forward(self, x, x_tgt):    
 
        # Get rid of channel dimension
        x = x.squeeze(dim=1)
        x_tgt = x_tgt.squeeze(dim=1)
        if x.ndim > 1 and x.shape[0] > 1:
            return self.batched_forward(x, x_tgt)

        # Get test and enrollment utterances
        test_utts = x
        if not self.tgt_set:
            with torch.no_grad():
                enroll_utts = x_tgt.to(x.device)
                enroll_embs = self.model(enroll_utts)
        else:
            enroll_embs = self.enroll_embs

        # Forward pass through model for test with grad
        test_embs = self.model(test_utts)

        # Compute scores (e.g. cosine similarity)
        if test_embs.ndim > 1:
            scores = self.score_fn(test_embs, enroll_embs, dim=1)
            scores = scores.unsqueeze(dim=1)
        else:
            scores = self.score_fn(test_embs, enroll_embs, dim=0)
            scores = scores.unsqueeze(dim=0).unsqueeze(dim=1)

        decisions = scores >= self.threshold
        return decisions, scores

    def batched_forward(self, x, x_tgt):    
        # Get test and enrollment utterances
        test_utts = x
        if not self.tgt_set:
            with torch.no_grad():
                enroll_utts = x_tgt.to(x.device)
                enroll_embs = []
                for utt in enroll_utts:
                    enroll_embs.append(self.model(utt.unsqueeze(dim=0)))
                enroll_embs = torch.cat(enroll_embs, dim=0)
        else:
            enroll_embs = self.enroll_embs.squeeze().unsqueeze(dim=0).repeat(test_utts.shape[0],1)

        # Forward pass through model for test with grad
        test_embs = []
        for utt in test_utts:
            test_embs.append(self.model(utt.unsqueeze(dim=0)).unsqueeze(dim=0))
        test_embs = torch.cat(test_embs, dim=0)

        # Compute scores (e.g. cosine similarity)
        if test_embs.ndim > 1:
            scores = self.score_fn(test_embs, enroll_embs, dim=1)
            scores = scores.unsqueeze(dim=1)
        else:
            scores = self.score_fn(test_embs, enroll_embs, dim=0)
            scores = scores.unsqueeze(dim=0).unsqueeze(dim=1)

        decisions = scores >= self.threshold
        return decisions, scores

