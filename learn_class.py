from nets.classification import LSTMTextClassificationNet
from helper import *

dataset = 'hp'
max_features = 5000

(x_train, y_train), (x_val, y_val), (x_test, y_test) = loadDataset(dataset, max_features=max_features)
print('x_train shape:', x_train.shape)
print('x_val shape:', x_val.shape)
print('x_test shape:', x_test.shape)

embedding_size = 100
class_lstm_hidden_size = 128
class_lstm_layers = 1
class_lin_layers_sizes = [1024,32]
dropout = 0.5
classes = 2

class_steps = 20000
status_every = 100
test_every = 1000
save_every = 1000

class_net = LSTMTextClassificationNet(max_features+1, embedding_size, classes, class_lstm_hidden_size,
                                      class_lstm_layers, class_lin_layers_sizes, dropout)
class_net = class_net.cuda() if use_cuda else class_net
print(class_net)

lr = 0.0001
weight_decay = 0.0001
prevalence = 0.5
batch_size = 1000

class_loss_function = torch.nn.MSELoss()
class_optimizer = torch.optim.Adam(class_net.parameters(), lr=lr, weight_decay=weight_decay)

x_train_pos = x_train[y_train==1]
x_train_neg = x_train[y_train!=1]
x_test_pos = x_test[y_test==1]
x_test_neg = x_test[y_test!=1]

with open('class_net_hist.txt', mode='w', encoding='utf-8') as outputfile, \
        open('class_net_test.txt', mode='w', encoding='utf-8') as testoutputfile:

    class_loss_sum, quant_loss_sum, acc_sum = 0, 0, 0
    t_init = time()
    for step in range(1, class_steps + 1):

        x, y_class, y_quant = sample_data(x_train_pos, x_train_neg, prevalence, batch_size)

        class_optimizer.zero_grad()
        class_net.train()
        y_class_pred = class_net.forward(x)
        class_loss = class_loss_function(y_class_pred, y_class)
        class_loss.backward()
        class_optimizer.step()

        class_loss_sum += class_loss.data[0]
        acc_sum += accuracy(y_class, y_class_pred)

        if step % status_every == 0:
            printtee('step {}\tloss {:.5f}\t acc {:.5f}\t v {:.2f} steps/s'
                  .format(step, class_loss_sum / status_every, acc_sum / status_every, status_every/(time()-t_init)), outputfile)
            class_loss_sum, acc_sum = 0, 0
            t_init = time()

        if step % test_every == 0:
            class_net.eval()
            test_var_x, test_var_y, y_quant = sample_data(x_test_pos, x_test_neg, prevalence, batch_size)
            y_class_pred = class_net.forward(test_var_x)
            test_accuracy = accuracy(test_var_y, y_class_pred)
            printtee('testacc {:.5f}'.format(test_accuracy), testoutputfile)

        if step % save_every == 0:
            filename = get_name(step, dataset)
            print('saving to', filename)
            with open('class_' + filename, mode='bw') as modelfile:
                torch.save(class_net, modelfile)
