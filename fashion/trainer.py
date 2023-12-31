import mlflow
import torch
import torchvision
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .transforms import get_image_transform
from .visualizers import plot_losses


class Trainer(object):
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
        transform = get_image_transform()

        train_set = torchvision.datasets.FashionMNIST(dataset_path, transform=transform)

        test_set = torchvision.datasets.FashionMNIST(
            dataset_path, train=False, transform=transform
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

                    train_loss = train_loss.detach().cpu().numpy()
                    test_loss = test_loss.detach().cpu().numpy()
                    accuracy = accuracy * 100 / total
                    precision = precision * 100 / total
                    recall = recall * 100 / total
                    f1 = f1 * 100 / total

                    iteration_list.append(count)
                    train_losses.append(train_loss)
                    test_losses.append(test_loss)
                    accuracy_scores.append(accuracy)
                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    f1_scores.append(f1)

                    if self.logging:
                        mlflow.log_metric("train_loss", train_loss, step=count)
                        mlflow.log_metric("test loss", test_loss, step=count)
                        mlflow.log_metric("accuracy", accuracy, step=count)
                        mlflow.log_metric("precision", precision, step=count)
                        mlflow.log_metric("recall", recall, step=count)
                        mlflow.log_metric("f1 macro", f1, step=count)

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
