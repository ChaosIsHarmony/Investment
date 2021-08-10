import torch
from . import neural_nets as nn
from . import common
from typing import List, Tuple



def save_model(model: nn.CryptoSoothsayer, filepath: str) -> None:
    torch.save(model.state_dict(), filepath)



def load_model(model_to_load: nn.CryptoSoothsayer, filepath: str) -> nn.CryptoSoothsayer:
    model = model_to_load
    model.load_state_dict(torch.load(filepath))

    return model



def load_model_by_params(filepath: str, params: dict) -> nn.CryptoSoothsayer:
    nn.set_model_parameters(dropout = params["dropout"], eta = params["eta"], eta_decay = params["decay"])
    nn.set_model(params["architecture"])
    nn.set_pretrained_model(load_model(nn.get_model(), filepath))
    nn.set_model_props(nn.get_model())

    return nn.get_model()



def print_evaluation_status(model_accuracy: List[float]) -> str:
    '''
    Prints summary for model evaluation.
    '''
    report = f"""
        POSITIVE:
            [+] Perfect accuracy: {model_accuracy[0]:>10.4f}
        NEGATIVE:
            [-] Told to hodl but should have sold/bought rate: {model_accuracy[1]:>10.4f}
            [--] Should have hodled but told to sell/buy rate: {model_accuracy[2]:>10.4f}
            [---] Told to do the opposite of correct move rate: {model_accuracy[3]:>10.4f}
        """

    print(report)

    return report



def evaluate_model(model: nn.CryptoSoothsayer, test_data: Tuple[List[float], float]) -> List[float]:
    model.eval()
    correct = 0
    safe_fail = 0
    nasty_fail = 0
    catastrophic_fail = 0
    for feature, target in test_data:
        feature_tensor, target_tensor = common.convert_to_tensor(feature, target)

        with torch.no_grad():
            output = model(feature_tensor)

        decision = torch.argmax(output, dim=1)

        # flawless
        if decision == target_tensor:
            correct += 1
            # catastrophic failure (e.g., told to buy when should have sold)
        elif (target_tensor > 1 and decision < 1) or (target_tensor < 1 and decision > 1):
            catastrophic_fail += 1
        # severe failure (e.g., should have hodled but was told to buy or sell
        elif target_tensor == 1 and (decision < 1 or decision > 1):
            nasty_fail += 1
        # decision was to hodl, but should have sold or bought
        else:
            safe_fail += 1

    model_accuracy = [correct/len(test_data), safe_fail/len(test_data), nasty_fail/len(test_data), catastrophic_fail/len(test_data)]


    return model_accuracy



def validate_model(model: nn.CryptoSoothsayer, valid_data: Tuple[List[float], float], lowest_valid_loss: float, filepath: str) -> Tuple[float, float]:
    '''
    Validates the model on the validation dataset.
    Saves model if validation loss is lower than the current lowest.
    Returns the average validation loss and lowest validation loss.
    '''
    # set to evaluate mode to turn off components like dropout
    model.eval()
    valid_loss = 0.0
    for features, target in valid_data:
        # make data pytorch compatible
        feature_tensor, target_tensor = common.convert_to_tensor(features, target)
        # model makes prediction
        with torch.no_grad():
            model_output = model(feature_tensor)
            loss = nn.get_criterion()(model_output, target_tensor)
            valid_loss += loss.item()

    avg_valid_loss = valid_loss/len(valid_data)

    return avg_valid_loss, lowest_valid_loss



def convert_to_tensor(features: List[float], target: float) -> Tuple[torch.tensor, torch.tensor]:
    '''
    Converts the feature vector and target into pytorch-compatible tensors.
    '''
    feature_tensor = torch.tensor([features], dtype=torch.float32)
    feature_tensor = feature_tensor.to(nn.get_device())
    target_tensor = torch.tensor([target], dtype=torch.int64)
    target_tensor = target_tensor.to(nn.get_device())

    return feature_tensor, target_tensor



