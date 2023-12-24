import torch
import torchvision
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .transforms import TRANSFORMS
from .visualizers import plot_losses


class Fitter(object):
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        n_epochs=1,
        batch_size=128,
        logging=True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.logging = logging

        self.model.to(device)

    def fit(self, dataset_path, plot=False):
        train_set = torchvision.datasets.FashionMNIST(dataset_path, transform=TRANSFORMS)

        test_set = torchvision.datasets.FashionMNIST(
            dataset_path, train=False, transform=TRANSFORMS
        )

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False
        )

        count = 0
        train_losses = []
        test_losses = []
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        iteration_list = []

        for _ in range(self.n_epochs):
            # Training the model
            self.model.train()

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                train_loss = self.criterion(outputs, labels)
                train_loss.backward()
                self.optimizer.step()

                count += 1

                # Validating the model
                if not (count % 50):
                    self.model.eval()

                    accuracy = 0
                    precision = 0
                    recall = 0
                    f1 = 0
                    total = 0

                    with torch.no_grad():
                        for images, labels in test_loader:
                            images = images.to(self.device)
                            labels = labels.to(self.device)
                            labels_cpu = labels.cpu().numpy()

                            outputs = self.model(images)
                            test_loss = self.criterion(outputs, labels)

                            predicts = torch.max(outputs, 1)[1]
                            predicts_cpu = predicts.cpu().numpy()

                            accuracy += accuracy_score(labels_cpu, predicts_cpu)
                            precision += precision_score(
                                labels_cpu, predicts_cpu, average="macro", zero_division=0
                            )
                            recall += recall_score(
                                labels_cpu, predicts_cpu, average="macro", zero_division=0
                            )
                            f1 += f1_score(labels_cpu, predicts_cpu, average="macro")

                            total += len(labels)

                    iteration_list.append(count)
                    train_losses.append(train_loss.detach().cpu().numpy())
                    test_losses.append(test_loss.detach().cpu().numpy())
                    accuracy_scores.append(accuracy * 100 / total)
                    precision_scores.append(precision * 100 / total)
                    recall_scores.append(recall * 100 / total)
                    f1_scores.append(f1 * 100 / total)

        if plot:
            plot_losses(
                iteration_list,
                train_losses,
                test_losses,
                accuracy_scores,
                precision_scores,
                recall_scores,
                f1_scores,
            )
