# ============ 导入模块 ============
import random  # 用来生成随机数
import math
# ============ 定义感知机类 ============
class Perceptron:
    """感知机类：最简单的神经网络"""

    # -------- 初始化方法 --------
    def __init__(self, input_size,output_size):
        self.layers=3
        self.neuron=4
        """
        当创建感知机对象时，自动调用这个方法
        input_size: 输入数据的维度（比如AND问题有2个输入，所以是2）
        """ 
        # 创建权重列表：每个输入对应一个权重
        # range(input_size) 生成 0, 1, ..., input_size-1
        # 对每个i，生成一个[-1, 1]之间的随机数作为权重
        self.weights = [
        [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(self.neuron)]
        ] + [
        [[random.uniform(-1, 1) for _ in range(self.neuron)] for _ in range(self.neuron)]
        for _ in range(self.layers - 3)
        ] + [
        [[random.uniform(-1, 1) for _ in range(self.neuron)]for _ in range(output_size)]
        ]
        # 偏置：也是一个随机数
        self.bias = [
            [random.uniform(-1, 1) for _ in range(self.neuron)]

        ]+[
            [random.uniform(-1,1) for _ in range(self.neuron)] for _ in range(self.layers-3)
            
        ]+[
            [random.uniform(-1,1) for _ in range(output_size)]
        ]

        # 学习率：控制每次学习的步长（0.1是个比较小的值，比较安全）
        self.lr = 0.5
    def sigmoid(self,x):
        return 1 / (1 + math.exp(-x))

    # -------- 激活函数 --------
    """ def activate(self, x):
        
        激活函数：把计算结果转换成0或1
        x: 加权求和的结果
        返回: 0 或 1
       
        # 如果x > 0，返回1；否则返回0
        # 这是最简单的阶跃函数
        return 1 if x > 0 else 0
 """
    # -------- 前向传播（预测） --------
    def predict(self, inputs):
        """
        前向传播：根据输入计算输出
        inputs: 输入数据，比如 [0, 1]
        返回: 预测结果（0或1）
        """
        a=inputs
        b=[]
        self.h=[inputs]
        # 第一步：加权求和
        # zip(inputs, self.weights) 把输入和权重配对：[(x1,w1), (x2,w2)]
        # x * w for x, w in ... 计算每个配对的乘积
        # sum(...) 把所有乘积加起来
        # 最后加上偏置
        for i in range(len(self.weights)):
            
            for j in range(len(self.weights[i])):
                z= sum(x * w for x, w in zip(a, self.weights[i][j])) + self.bias[i][j]
                output = self.sigmoid(z)
                b.append(output)
            
            a=b
            self.h.append(b)
            b=[]

        # 第二步：通过激活函数
        

        # 返回结果
        return a[0]

    # -------- 训练方法 --------
    def train(self, inputs, target):
        """
        训练：根据错误调整权重
        inputs: 输入数据，比如 [0, 1]
        target: 正确答案（0或1）
        返回: 误差（用来判断学习效果）
        """
        # 第一步：用当前的权重预测一下
        #output = self.predict(inputs)

        # 第二步：计算误差
        # 误差 = 正确答案 - 实际预测
        # 如果预测正确，误差=0，权重不用改
        # 如果预测错误，误差=1或-1，需要调整权重
        #error = target - output

        # 第三步：更新权重
        # 权重更新公式：w = w + 学习率 × 误差 × 输入
        # 对每个权重都更新
        #for i in range(len(self.weights)):
            # self.weights[i] = self.weights[i] + self.lr * error * inputs[i]
            #self.weights[i] += self.lr * error * inputs[i]

        # 第四步：更新偏置
        # 偏置更新公式：bias = bias + 学习率 × 误差
        #self.bias += self.lr * error
        y = self.predict(inputs)
        presigma=None
        # 返回误差，方便外面打印
        #return error
        for i in range(len(self.weights) - 1,-1,-1):#遍历每一层
            cursigma = []
            for j in range(len(self.weights[i])):#遍历神经元
                out = self.h[i+1][j]
                if presigma is None:
                    sigma = (target - out) * out * (1 - out)

                else:
                    duty=sum(presigma[n]*self.weights[i+1][n][j] for n in range(len(self.weights[i+1])))
                    sigma=duty*out * (1 - out)
                cursigma.append(sigma)
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k]=self.weights[i][j][k]+self.lr*sigma*self.h[i][k]
                self.bias[i][j] += self.lr * sigma
            presigma=cursigma
        return target-y


                    
# ============ 主程序 ============

# -------- 准备训练数据 --------
# AND 问题的真值表
# 输入: [x1, x2], 输出: x1 AND x2
training_data = [
    ([0, 0], 0),  # 0 AND 0 = 0
    ([0, 1], 1),  # 0 AND 1 = 0
    ([1, 0], 1),  # 1 AND 0 = 0
    ([1, 1], 0),  # 1 AND 1 = 1
]

# -------- 创建感知机对象 --------
# 输入是2维的（x1和x2），所以 input_size=2
p = Perceptron(input_size=2,output_size=1)

# -------- 训练前的预测（看看初始状态） --------
print("训练前预测：")
# 遍历每个训练样本
for inputs, _ in training_data:
    # 预测
    result = p.predict(inputs)
    # 打印结果
    print(f"{inputs} -> {result}")

# -------- 训练循环 --------
print("\n开始训练...")
# 训练100轮（epoch=100）
for epoch in range(1000):
    # 累计本轮的误差
    total_error = 0

    # 遍历每个训练样本
    for inputs, target in training_data:
        # 训练：根据这个结果调整权重
        error = p.train(inputs, target)
        # 累计误差的绝对值
        total_error += abs(error)

    # 每10轮打印一次
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, 总误差: {total_error}")

# -------- 训练后的预测（看看学习效果） --------
print("\n训练后预测：")
# 遍历每个训练样本
for inputs, target in training_data:
    # 预测
    pred = p.predict(inputs)
    # 打印结果（预测值和正确答案）
    print(f"{inputs} -> {pred} (正确答案: {target})")

# -------- 打印最终的权重和偏置 --------
print(f"\n最终权重: {p.weights}")
print(f"最终偏置: {p.bias}")
