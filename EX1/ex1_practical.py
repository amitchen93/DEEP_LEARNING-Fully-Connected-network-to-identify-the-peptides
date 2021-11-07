from helpers import *
import torch.optim as optim


training_generator, test_generator = get_generators()


# create the nn
net = Net().float()
criterion = nn.BCELoss()
optimizer = optim.Adamax(net.parameters(), lr=learning_rate)


train_loss, train_acc = [], []
test_loss, test_acc = [], []
for epoch in range(max_epochs):
    # Training
    loss_avg, accuracy_avg = run_epoch(net, training_generator, optimizer, criterion)
    train_loss.append(loss_avg)
    train_acc.append(accuracy_avg)

    # Validation
    loss_avg, accuracy_avg = run_epoch(net, test_generator, optimizer, criterion, False)
    test_loss.append(loss_avg)
    test_acc.append(accuracy_avg)

# get matrices
y_pred, y_labels = get_matrices(net, test_generator)
plot_results(train_loss, test_loss, train_acc, test_acc, y_pred, y_labels)

# print the 5 most detectable peptides in the Spike Sars protein
optimize_sequence(net)
