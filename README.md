# CNN-Forex-

#  Convolutional Neural Network Summary
#  CNN uses feature extracting method from the given input as RGB (N x N x 3) or Grey scaled images (N x N x 1). It uses filters to find corresponding weights which is to figure out the importance of the pixels in the images that matters.
#  There are some technics such as Padding, Strides, Pooling layers, CNN Backpropagation
#  Third party Nets such as classical Net (LeNet-5, AlexNet, VGG), newer nets ResNet, Inception Net
#  ResNet '2015 (in plain network, increasing # of layers should decrease trainning error but actaully increases because of vanishing gradient. Resblocks allows the gradient to be directly backpropagated to earlier layers and solve problem of "plain network". Resblocks learns identity function easily(the one with less important weights to be gone) -> this harms less on trainning set performance.
#  Inception Network '2014 (1x1 convolution: shrink number of channels, bottleneck layer: uses 1x1 conv before applying big fillters to reduce the computation cost. then uses bottleneck to stack up different channels of the filters. ex) 28x28x1 + 28x28x3 + 28x28x5 +...+ maxpooling layer with same padding. side branch with FCL and softmax helps that we are going in right way)
#  Data Augmentation: It should be implemented on-fly and parallel(saves memory) rotation, random cropping, mirroring, shearing, local warping, color shifting, PCA color augmentation.
#  Ensembling: train several different network independently and uses average of their outputs
#  Transfer Learning: nn + weights from other sources. 1. freeze all and just replace last layers with SM 2.precompute last frozen layer activation and convert X through all fixed layers and save to disk



cnn의 등장배경:
원래는 이미지를 nn에 대입시켜 아웃풋을 얻으려하면 64x64x3 =12228같이 x에 엄청난양의 값을 대입시키고 첫번째 레이어에 1000개의 노드가있을경우 1200만개의 파라메터가 생깁니다. 근데 이건 작은 이미지의 예이고 더 큰 이미지 사용시 엄청난양의 파라메터가 생기므로 학습이 불가능하게됩니다.(계산량이 많아져). 그러므로 이걸 효율적으로 큰 이미지에도 적용시킬법을 찾은것이 convolutional neural network입니다. 
cnn은 큰 이미지를 작은 부분들로 나누고 그 작은 부분에서 더 작은 부분으로 정보를 축약시킬 수 있는 방법을 생각했습니다. 그리하여 만들어진 vertical, horizontal edge filter들이 존재하고, vertical은 작은 이미지의 세로축 정보를 저장해서 큰 이미지의 세로축 정보들을 저장하는 방법이고, horizontal은 작은 이미지의 가로축 정보들을 저장해 큰 이미지를 압축하는 방식이 있었습니다. 또, 이러한 필터에 끝나지않고 필터를 직접 지정하는 방법이 아닌 weight들을 미지정한후 backpropagation을 통해 학습해 나가며 이미지의 정보를 압축시키는 최적의 필터를 만드는 방법또한 나오게 됐습니다. 
convolve: 둘둘말다. 

padding: 이미지를 필터를 통해 압축시키다보면 수많은 layer를 거칠시에 엄청나게 작아지게 되는 것을 방지하기위해서, 또 패팅을 안할시에 모서리끝자락에 존재하는 값들은 중요도가 매우 낮아지는반면에 중앙값들은 중요해지는것을  없애기위해서 이미지의 테두리에 값을 줌으로 크기를 유지하고 중요도도 유지하는 방법. 	

stride formula: output size = [(n+2p-f)/s+1] round-down

cross-correlation vs. convolution 

6x6x3 with 3x3x3(same chanenl for filter)filter => 4x4x1 (bitwise operation gives single numbers)
필터마다 vertical 을 뽑아낼지 horizontal을 뽑아낼지 결정되므로 여러가지 필터로 많은 feature들을 뽑아낼 수 있게된다. 

6x6x3 with 3x3x3 filter with 2diff filter gives 4x4x2 

pooling layer: max pooling: 필터사이즈와 스트라이드 사이즈를 설정하고 그 안의 가장 큰 값들만 추출한다. 원래 pooling을 하기전의 필드의 값들은 얼마나 그 부분이 중요한지를 나타내는 지표이기때문에 굳이 모든 값을 사용할 필요는 없고 가장 중요한값만 사용하자는 취지로 풀링을 하게된다. 만약 좋은 feature를 찾았으면 그것은 보존하고 아니다면 그냥 버려라. 지금까지 매우 잘 이용되고 좋은 성능을 보이고 있지만 완벽히 왜 잘작동하는지는 모른다. f=2, s=2 common f=3, s=2 no parameters to learn 

왜 cnn을 사용할까 그리고 장점과 단점은 무엇일까? 일단 파라미터가 확실히 더 적다. 필터에만 파라메타가 존재하기때문에, fc보다 훨씬 적어진다. feature detector(vertical edge detector) shares hyperparameters. 또한 in each layer, each output value depneds only on a small nu,ber of inputs.: Sparsity of connection(연결성의 분산) 2,4,6,7,10
