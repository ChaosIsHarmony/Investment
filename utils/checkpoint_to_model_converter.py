'''
Used to convert checkpoints into models.
'''

import torch

chkpt = input("file suffix [after '...CryptoSoothsayer_']: ")

saved_chkpt = torch.load(f"models/checkpoint_CryptoSoothsayer_{chkpt}.pt")
model = saved_chkpt["model"]
model.optimizer_state = saved_chkpt["optimizer_state"]
model.load_state_dict(saved_chkpt["state_dict"])
model.device = saved_chkpt["device"]

torch.save(model.state_dict(), f"models/CryptoSoothsayer_{chkpt}.pt")
