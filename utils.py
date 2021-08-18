
class ScheduledOptimizer(object):

    def __init__(self, optimizer, d_model, warmup_step):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_step = warmup_step
        self.current_step=0

    def step_and_update_lr(self):
        self.update_lr()
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def update_lr(self):
        self.current_step+=1
        lr = self.d_model**(-0.5)*min(self.current_step**(-0.5), self.current_step*self.warmup_step**(-1.5))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        lr=[param_group['lr'] for param_group in self.optimizer.param_groups]
        return lr   




if __name__ == "__main__":
    
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader,TensorDataset
    class M(nn.Module):
        def __init__(self):
            super(M, self).__init__()
            self.fc = nn.Linear(100, 100)

        def forward(self, x):
            return self.fc(x)


    d_model=512
    batch_size=160
    epochs=10
    data=torch.load('./data/train_data_cache.pkl')
    dataset=TensorDataset(data['src'],data['tgt'])
    dataloader=DataLoader(dataset,batch_size=batch_size)
    total_step=len(dataloader)*epochs
    warmup_step=int(total_step*0.1)
    print(total_step,warmup_step)
    model = M()
    optimizer = Adam(model.parameters())
    sch_optim = ScheduledOptimizer(optimizer, d_model, warmup_step)
    lr_list = []
    for step in range(total_step):
        sch_optim.update_lr()
        lr_list.append(sch_optim.get_lr()[0])
        

    plt.plot(range(len(lr_list)), lr_list)
    plt.show()
