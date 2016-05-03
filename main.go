package main

import (
	"fmt"
	"path/filepath"
	"github.com/lkiversonlk/DeepLearn-Go/core"
)
func main() {

	imagePath, e0 := filepath.Abs("./data/train-images-idx3-ubyte")
	labelPath, e1 := filepath.Abs("./data/train-labels-idx1-ubyte")

	testImagePath, e2 := filepath.Abs("./data/t10k-images-idx3-ubyte")
	testLabelPath, e3 := filepath.Abs("./data/t10k-labels-idx1-ubyte")

	if e0 != nil || e1 != nil || e2 != nil || e3 != nil{
		panic("load data error");
	} else {
		fmt.Println("read image file ", imagePath, " label file", labelPath)
	}

	mnist, err1 := core.ParseMNIST(imagePath, labelPath)
	mninstTest, err2 := core.ParseMNIST(testImagePath, testLabelPath)
	if(err1 != nil || err2 != nil){
		fmt.Println("fail to load MNIST data");
	}else{
		network := core.NetNetwork([]int{ mnist.ImageHeight * mnist.ImageRow, 30, 10})
		network.SGD(mnist, 100, 60, 3.0, mninstTest)

	}
}
