# 这里用于实现adam优化器的编写
import torch

class Adam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-7):
        # 读取模型的参数，按照列表的形式存储张量
        self.params = list(params)
        self.lr     = lr
        self.betas  = betas
        self.eps    = eps
    
        # 对于M与V做一个初始化
        self.m      = [torch.zeros_like(p) for p in self.params]
        self.v      = [torch.zeros_like(p) for p in self.params]
        
        # 对于时间步做一个初始化
        self.t      = 0
    
    # 定义step方法，用于每一步的梯度更新
    def step(self):
        self.t     += 1
        for i, param in enumerate(self.params):
        
        # 做一个判断，防止梯度不存在
            if param.grad is None:
                raise ValueError("param:{} in layer:{} does not have grad!!!".format(param,i))
            
            # 拿到当前的梯度信息
            grad          =  param.grad
            # 进行adam的动量更新
            self.m[i]     = self.betas[0]*self.m[i]+(1-self.betas[0])*grad
            self.v[i]     = self.betas[1]*self.v[i]+(1-self.betas[1])*grad
            self.m_hat    = self.m[i]/(1-self.betas[0]**self.t)
            self.v_hat    = self.v[i]/(1-self.betas[1]**self.t)
            param.data   -= self.lr*self.m_hat/(torch.sqrt(self.v_hat)+self.eps)
            
    # 清空当前梯度，方便进行下一步的反向传播
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
         
