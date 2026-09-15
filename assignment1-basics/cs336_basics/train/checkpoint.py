import torch

def save_checkpoint(model, optimizer, iteration, out):
    check_point = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    return torch.save(check_point,out)

def load_checkpoint(src, model, optimizer):
    check_point = torch.load(src)
    model.load_state_dict(check_point["model"])
    optimizer.load_state_dict(check_point["optimizer"])
    return check_point["iteration"]