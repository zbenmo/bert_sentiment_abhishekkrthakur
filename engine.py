import torch
import torch.nn as nn
from tqdm import tqdm


def loss_fn(output, targets):
    return nn.BCEWithLogitsLoss()(output, targets.view(-1, 1))


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()

    for _, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d['ids']
        mask = d['mask']
        token_type_ids = d['token_type_ids']
        targets = d['targets']

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        output = model(
            ids = ids,
            mask = mask,
            token_type_ids = token_type_ids
        )

        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()


def eval_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d['ids']
            mask = d['mask']
            token_type_ids = d['token_type_ids']
            targets = d['targets']

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(
                ids = ids,
                mask = mask,
                token_type_ids = token_type_ids
            )
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    return fin_outputs, fin_targets