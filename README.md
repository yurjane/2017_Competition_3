# 2017_Competition_3

## 1）难点解决

### 1.1 图像处理

本次竞赛中最为困难的部分是图像处理，其中有 3 个难点：

1. 文字与背景的对比度不大；

2. 横向干扰线的干扰；

3. 字符的分割。

如何解决这 3 个难题：

1. 因为图像的字符位置相对稳定，故可以按照 30×30 的格式对图像进行切割。之后进行高对比度的灰度增强；

2. 进行横向腐蚀；

3. 灰度，二值化后进行连通域判断。

虽然有解决问题的方法，但他们互相牵制，无法做到十全十美：步骤 1 与步骤 2 均可以影响到步骤 3。在进行步骤 3 时重点在于灰度化与连通域判断，但进行整体的高对比灰度化会导致与背景近似的内容丢失，横向腐蚀会丢失横向的细节导致字符断开。综合考虑，对字符的完美分割是不可取的，只能通过 30×30 的裁剪实现字符的分割，这样的丢信息的不可避免的。

对于难点 1，解决方案：

1. 采取了一种动态的灰度化方法，通过尝试多种灰度化组合，找到整体对比度最高的灰度化方案；

2. 进行模糊使字符与背景各自呈现出统一的颜色；

3. 进行灰度分层的处理，促使对比度增强。

对于难点 2，解决方案：

1. 进行横向腐蚀；

2. 膨胀，使字符连在一起。

这样，图像就完成了基本的处理。

### 2）代码实现

数据读取：

```python
class get_verification_code(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        img_original: torch.Tensor = None
        if self.transform is not None:
            img_original = self.transform(img).to(DEVICE)
            # img_original = F.interpolate(img_original, size=size, mode='bilinear')
        if self.target_transform is not None:
            target = self.target_transform(target)
        # targets = torch.zeros(62).to(DEVICE)
        # targets[target] = 1
        return img_original, target
```

判别器：

```python
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.midlevel_resnet = models.resnet18(num_classes=62)

    def forward(self, alphabet) -> torch.Tensor:
        target = self.midlevel_resnet(alphabet)
        return target
```

训练过程：

```python
max_epoch = 100
batch_size = 64

dataset = get_verification_code('./char', transform=transforms.ToTensor())
dataset = DataLoader(dataset, batch_size=batch_size, shuffle=True)

D = Discriminator().to(DEVICE)
if os.path.exists(r'./module.pkl'):
    D.load_state_dict(torch.load(r'./module.pkl'))
    print(True)
else:
    print(False)
    pass

criterion = nn.CrossEntropyLoss()

D_opt = torch.optim.Adam(D.parameters(), lr=0.02, betas=(0.5, 0.999))

step = 0
for epoch in range(max_epoch):
    for idx, (img, target) in enumerate(dataset):
        target = target.to(DEVICE)
        pred = D(img)
        loss = criterion(pred, target)
        D.zero_grad()
        loss.backward()
        D_opt.step()
        if step % 50 == 0:
            print('Epoch: {}/{}, Step: {}, D Loss: {}'.format(epoch, max_epoch, step, loss.item()))
        if step % 5 == 0:
            torch.save(D.state_dict(), r'./module.pkl')
            cur = datetime.datetime.now()
            print(f'now:{cur:%Y-%m-%d (%a) %H:%M:%S} save the modul')
        step += 1

```

判别过程：

```python
if os.path.exists(r'./module.pkl'):
    D.load_state_dict(torch.load(r'./new_module.pkl'))
    targets = list()
    for dir in dirlist:
        print(dir)
        image = Image.open(f'./test/{dir}').resize((224 * 5, 224), resample=Image.BILINEAR)
        image = trans(image).to(DEVICE)
        chars = list()
        for sub in range(5):
            imageArr = image[..., sub * 224:(sub + 1) * 224].view((1, 3, 224, 224))
            labels = list(D(imageArr)[0])
            target = labels.index(max(labels))
            chars.append(number2alphabet[int(target)])
        print(''.join(chars))
        targets.append(''.join(chars))
    data = pd.DataFrame({'y':targets})
    data.to_csv('./test.csv')
else:
    pass
```