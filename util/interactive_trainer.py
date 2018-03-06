#!/usr/bin/env python
import sys, os
import termios
import contextlib
import threading
import time
from queue import Queue
import torch
import torch.nn as nn
from collections import OrderedDict

# TODO: async
# TODO: conditional actions

class InteractiveNeuralTrainer(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.setDaemon(True)
        self._listeners = OrderedDict()
        self._sync_triggers = set()
        self._sync_queue = Queue()

    def add_interaction(self, trigger, action, synchronized=False):
        assert isinstance(trigger, str) and len(trigger) == 1, 'a single char should be specified as the trigger'
        assert hasattr(action, '__call__'), 'the action should be callable'
        assert trigger not in self._listeners, '{} already in use'.format(trigger)

        self._listeners[trigger] = action
        if synchronized:
            self._sync_triggers.add(trigger)

        if trigger=='h':
            print('warning: {} will hidden the "help" interaction'.format(action.__name__))

    def run(self):
        assert len(self._listeners) > 0, 'no interactions available'
        if 'h' not in self._listeners:
            self.add_interaction('h', self.help)
        with raw_mode(sys.stdin):
            print(self.__class__.__name__+' listening!')
            self.help()
            while True:
                trigger = sys.stdin.read(1)
                if trigger in self._listeners:
                    if trigger in self._sync_triggers:
                        self._sync_queue.put(trigger)
                    else:
                        self._listeners[trigger]()

    def synchronize(self):
        while not self._sync_queue.empty():
            trigger = self._sync_queue.get()
            print('\tsync: {}({})'.format(trigger, self._sync_queue.qsize()))
            self._listeners[trigger]()

    def help(self):
        print('\tTrigger\t->\taction')
        for trigger in self._listeners.keys():
            print('\t{}\t->\t{}'.format(trigger, self._listeners[trigger].__name__))


@contextlib.contextmanager
def raw_mode(file):
    #from https://stackoverflow.com/questions/11918999/key-listeners-in-python
    old_attrs = termios.tcgetattr(file.fileno())
    new_attrs = old_attrs[:]
    new_attrs[3] = new_attrs[3] & ~(termios.ECHO | termios.ICANON)
    try:
        termios.tcsetattr(file.fileno(), termios.TCSADRAIN, new_attrs)
        yield
    finally:
        termios.tcsetattr(file.fileno(), termios.TCSADRAIN, old_attrs)

def inspect_param(optimizer, param):
    for param_group in optimizer.state_dict()['param_groups']:
        return param_group[param]


def adapt_optimizer(optimizer, factor, param):
    #assert param in optimizer.state_dict(), 'unknown parameter {} for the optimizer'.format(param)
    def adapt_optimizer_():
        state_dict = optimizer.state_dict()
        for param_group in state_dict['param_groups']:
            if param_group[param] == 0 and factor > 1:
                param_group[param] = 1e-4
            else:
                param_group[param] *= factor
            print('\t%s updated: %.8f' % (param, param_group[param]), end='\n')
        optimizer.load_state_dict(state_dict)
    return adapt_optimizer_

def increase_lr(optimizer, factor=1.1):
    assert factor>1, 'the factor should be >1'
    return adapt_optimizer(optimizer, factor, 'lr')

def decrease_lr(optimizer, factor=0.9):
    assert factor<1, 'the factor should be <1'
    return adapt_optimizer(optimizer, factor, 'lr')

def increase_weight_decay(optimizer, factor=1.1):
    assert factor>1, 'the factor should be >1'
    return adapt_optimizer(optimizer, factor, 'weight_decay')

def decrease_weight_decay(optimizer, factor=0.9):
    assert factor<1, 'the factor should be <1'
    return adapt_optimizer(optimizer, factor, 'weight_decay')

def quick_load(net, save_dir):
    saved_path = os.path.join(save_dir, net.__class__.__name__ + '_quick_save')
    def quick_load_():
        print('\tloading ' + saved_path)
        net.load_state_dict(torch.load(saved_path).state_dict())
    return quick_load_

def quick_save(net, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    def quick_save_():
        save_path = os.path.join(save_dir, net.__class__.__name__+'_quick_save')
        print('\tsaving to ' + save_path)
        with open(save_path, mode='bw') as modelfile:
            torch.save(net, modelfile)
    return quick_save_

def validation(net, X, y, criterion):
    def validation_():
        y_ = net(X)
        eval = criterion(y_,y)
        print('\tValidation: %4f' % eval)
    return validation_

def reboot(net, net_args, optimizer, tracked_optim_params):
    assert isinstance(net,nn.Module), 'cannot reboot on this instance, use a nn.Module'
    def reboot_():
        net.__init__(**net_args)
        optim_params = dict((op_param,inspect_param(optimizer,op_param)) for op_param in tracked_optim_params)
        optimizer.__init__(net.parameters(), **optim_params)
        print('\t%s rebooted' % net.__class__.__name__)
    return reboot_


# ------------------------------------------------------------------------------------------
# EXAMPLE OF USE
# ------------------------------------------------------------------------------------------

def main():
    from torch.autograd import Variable

    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    unknown_transformation = torch.rand(100,10) #what this toy example tries to learn
    X = Variable(torch.rand(10000,100))
    y = Variable(torch.mm(X.data,unknown_transformation))
    Xval = Variable(torch.rand(1000, 100))
    yval = Variable(torch.mm(Xval.data, unknown_transformation))

    innt = InteractiveNeuralTrainer()
    innt.add_interaction('w', increase_lr(optimizer, factor=10.))
    innt.add_interaction('s', decrease_lr(optimizer, factor=.1))
    innt.add_interaction('a', decrease_weight_decay(optimizer=optimizer, factor=.5))
    innt.add_interaction('d', increase_weight_decay(optimizer=optimizer, factor=2.))
    #innt.add_interaction('r', reboot(net, optimizer, tracked_optim_params=['lr','weight_decay']))
    innt.add_interaction('v', validation(net, Xval, yval, criterion))
    innt.add_interaction('q', quick_save(net, 'checkpoint'), synchronized=True)
    innt.add_interaction('e', quick_load(net, 'checkpoint'), synchronized=True)
    innt.start()

    for i in range(10000):
        optimizer.zero_grad()
        y_ = net.forward(X)
        loss = criterion(y_, y)
        loss.backward()
        optimizer.step()
        # innt.process()
        print('\r[step {}]: loss={:.8f}, lr={:.8f}, weight_decay={:.8f}'.format(i, loss.data[0], inspect_param(optimizer, 'lr'), inspect_param(optimizer, 'weight_decay')) , end='\n')
        time.sleep(.5)


if __name__ == '__main__':

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(100, 10)

        def forward(self, x):
            return torch.nn.functional.relu(self.fc1(x))

    net = Net()

    main()