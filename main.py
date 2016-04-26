import sys
import operator
import math as math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report

# 行列の各要素にlogistic()を適用
def logistic(U):
    return 1.0/(1.0 + np.exp(-1.0*U))

# 行列の各要素にlogistic_deriv()を適用
def logistic_deriv(U):
    t = logistic(U)
    return t * (1.0-t)

def softmax(U):
    U = U - np.max(U, axis=0) # Uを正規化
    U = np.exp(U)             # 全要素にexp()を適用
    U = U / np.sum(U, axis=0)
    return U

# リストliの中で最も大きな数値に対応するindexを返す
# l2n( [00 11 27 19 18] ) -> 2
def l2n(li):
    return max(range(len(li)), key=(lambda i: li[i]))

# 数値xをサイズ10のリストに変換する
# n2l(3) -> [0 0 0 1 0 0 0 0 0 0]
def n2l(x):
    res = [0,0,0,0,0,0,0,0,0,0]
    res[x] = 1
    return res

# 多層ネットワーク, 多クラス分類器
class FeedforwardNeuralNetwork:

    # コンストラクタ
    # sizes: 入力層，中間層，出力層のユニット数
    # f: 活性化関数．ただし，入力層では不要
    # f_deriv: 中間層のみ
    def __init__(self, sizes, f, f_deriv):

        # 各層のユニット数
        self.sizes = sizes

        # 層の数
        self.L = len(sizes)

        # 活性化関数f，活性化関数の導関数f_deriv
        self.f, self.f_deriv = f, f_deriv

        # 各層の重みW，バイアスb
        self.W, self.b = list(range(self.L)), list(range(self.L))
        for l in range(1,self.L):
            self.W[l]       = np.random.uniform(-1.0,1.0,(sizes[l], sizes[l-1]))
            self.b[l]       = np.random.uniform(-1.0,1.0,(sizes[l], 1))

    # ネットワークにデータを伝搬させる
    #
    # X=[x1,x2,...xn]: 入力データ（ミニバッチ），各データxiは列ベクトル
    # D=[y1,y2....yn]: Xに対応する正解出力
    #
    # D==[]: 順伝播により推測する．返値は，Xに対応する出力．
    # D!=[]: 順伝播と逆伝播により重みを学習する．返値は，誤差．
    def propagate(self, X, D = []):
        # --- 順伝播 ---
        N = len(X[0]) # 列数
        Z, U = list(range(self.L)), list(range(self.L))
        Z[0] = np.array(X,dtype=np.float64)
        for l in range(1,self.L):
            U[l] = np.dot(self.W[l],Z[l-1]) + np.dot(self.b[l],np.ones((1,N)))
            U[l] = U[l].astype(np.float64)
            Z[l] = self.f[l]( U[l] )
        if len(D)==0:
            return Z[self.L-1]

        # --- 逆伝播 ---
        De = list(range(self.L))
        Y  = Z[ self.L -1]
        D  = np.array(D)
        eps = 0.4 # 学習係数

        # De の計算
        De[ self.L-1 ] = Y - D # 注意．P51とは違う．P49を見るとこちらが正しい?
        for l in range(self.L-2,0,-1):
            De[l] = self.f_deriv[l]( U[l] ) * np.dot(self.W[l+1].T , De[l+1] )

        # W,bの更新
        for l in range(self.L-1,0,-1):
            self.W[l] = self.W[l] -  eps/N*np.dot(De[l] , Z[l-1].T)
            self.b[l] = self.b[l] -  eps/N*np.dot(De[l] , np.ones((N, 1)))

        # 誤差を返す
        return np.power(De[self.L-1],2).sum()/N

# 2進から10進への変換によるFeedforwardNeuralNetworkの動作確認
def binary2decimal_test():
    dl = FeedforwardNeuralNetwork(
            [3,8,10], [None,logistic,softmax], [None,logistic_deriv,None])
    for i in range(1000):
        dl.propagate(
                [[0,1,0,1,0,1,0,1],
                 [0,0,1,1,0,0,1,1],
                 [0,0,0,0,1,1,1,1]],
                [[1,0,0,0,0,0,0,0],
                 [0,1,0,0,0,0,0,0],
                 [0,0,1,0,0,0,0,0],
                 [0,0,0,1,0,0,0,0],
                 [0,0,0,0,1,0,0,0],
                 [0,0,0,0,0,1,0,0],
                 [0,0,0,0,0,0,1,0],
                 [0,0,0,0,0,0,0,1],
                 [0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0]])
    X = [[0],
         [1],
         [1]]
    Y = dl.propagate(X)
    print("input:")
    for i in range(2,-1,-1):
        print(X[i][0], end='')
    print("\nanswer:")
    for i in range(10):
        print(str(i) + ": " + str(int(Y[i][0]*100)) + "%")

# MNISTの識別器
class MNIST:

    # コンストラクタ
    def __init__(self):
        # MNISTデータセット
        mnist = fetch_mldata('MNIST original', data_home=".")

        # 画像データX，正解データy
        X = mnist.data
        y = mnist.target

        # 0.0から1.0に正規化
        X = X.astype(np.float64)
        X /= X.max()

        # Xとyを訓練データおよびテストデータに分割
        # test_size: テストデータの割合
        self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(X,y,test_size=0.1)

        # ネットワークを構築
        self.dl = FeedforwardNeuralNetwork([28*28,100,10],
            [None,logistic,softmax], [None,logistic_deriv,None])

    # ネットワークを学習させる
    # num: 学習回数
    def learn(self, num):

        # 訓練データのindex集合
        train_indexes = list(range(0, self.X_train.shape[0]))

        for k in range(num):

            # ミニバッチ（学習の単位）を選択
            minibatch = np.random.choice(train_indexes, 50)

            # 訓練データ，正解データをミニバッチに合わせて抽出
            inputs  = []
            outputs = []
            for ix in minibatch:
                inputs.append(self.X_train[ix])
                outputs.append(n2l(int(self.y_train[ix])))
            inputs = np.array(inputs).T
            outputs = np.array(outputs).T

            # 学習
            self.dl.propagate(inputs, outputs)

            # 進捗状況を出力
            if k% (num//100) ==0:
                if (k*100//num) % 10 ==0:
                    sys.stdout.write(str(k*100//num))
                else:
                    sys.stdout.write(".")
            sys.stdout.flush()
        print()

    # 添字ixのデータをクラス分類
    # ix: テストデータ中の添字
    # 返値: (回答, その確率)
    def predict_one(self, ix):
        inputs = np.array([self.X_test[ix]]).T
        t = self.dl.propagate(inputs)
        return max(enumerate(t.T[0]), key=operator.itemgetter(1))

    # 幾つかのテストデータをテストして，出力
    def predict_some(self):
        # サイズ
        row = 5
        col = 8

        fig = plt.figure(figsize=(12, 9))
        for i in range(1,row*col+1):
            ix = np.random.randint(self.X_test.shape[0])
            img = 256.0 * self.X_test[ix]
            img = img.reshape(28,28)
            a = fig.add_subplot(row,col,i)
            a.axis('off')
            plt.imshow(img,cmap='Greys_r')
            res = self.predict_one(ix)
            a.set_title(str(res[0]) + " [" + str(int(res[1]*100)) + "%]")
        plt.show()


    # 全てのテストデータをテストし，精度・再現率・F値を出力
    def predict_all(self):
        # 多クラス分類を実行
        predictions = self.dl.propagate(np.array(self.X_test).T)

        # 整形
        predictions = np.array(list(map(l2n, predictions.T)))

        # 精度・再現率・F値を出力
        print(classification_report(self.y_test, predictions))

a = MNIST()

print("start learning")
a.learn(2000)
print("start testing")
a.predict_all()
a.predict_some()
