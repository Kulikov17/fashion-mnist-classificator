import torch
import torchvision

from .transforms import get_image_transform


def inference(dataset_path, model, device, batch_size):
    model.to(device)

    transform = get_image_transform()

    test_set = torchvision.datasets.FashionMNIST(
        dataset_path, train=False, transform=transform
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )

    y_true = []
    y_pred = []

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        labels_cpu = labels.cpu().numpy()
        outputs = model(images)
        predicts = torch.max(outputs, 1)[1]
        predicts_cpu = predicts.cpu().numpy()
        y_true.extend(labels_cpu)
        y_pred.extend(predicts_cpu)

    return y_true, y_pred
