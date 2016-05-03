package src

import (
	"math/rand"
	//"math"
	"fmt"
	"strconv"
)
type Network struct {
	LayerCount int
	Layers     []int
	Biases     []*Rect
	Weights    []*Rect
}

func NetNetwork(sizes []int) *Network {
	ret := &Network{LayerCount:len(sizes), Layers: sizes}
	ret.Biases = make([]*Rect, ret.LayerCount - 1, ret.LayerCount - 1)
	ret.Weights = make([]*Rect, ret.LayerCount - 1, ret.LayerCount - 1)

	for i := 0; i < ret.LayerCount - 1; i ++ {
		ret.Biases[i] = NewRect(ret.Layers[i + 1], 1).Randomize()
		ret.Weights[i] = NewRect(ret.Layers[i + 1], ret.Layers[i]).Randomize()
	}

	return ret
}

func (network *Network)evaluate(mnist *MNIST) int {
	correct := 0
	for _, data := range mnist.Data {
		out := network.shake(data.Image)
		max := out.Get(0, 0)
		result := 0
		for j := 1; j < 10; j ++{
			if out.Get(j, 0) > max {
				max = out.Get(j, 0)
				result = j
			}
		}

		should := -2000
		for j := 0; j < 10; j ++ {
			if int(data.Label.Get(j, 0)) == 1 {
				should = j
				break
			}
		}

		if result == should {
			correct ++;
		}
		//fmt.Printf("recognized result is %d, actually is %d\n", result, should)
	}
	return correct
}

func (network *Network)shake(input *Rect) *Rect {
	if(input.Width != network.Layers[0]){
		panic("shake only accepts input whose length is the same with the network: " + string(network.Layers[0]))
	}else{
		for i := 0; i < network.LayerCount -1; i ++ {
			input = network.Weights[i].Junc(input).Add(network.Biases[i]).Operate(OperateWrapper(Sigmoid))
		}

		return input
	}
}

func  (network *Network)SGD(mnist *MNIST, rounds int, mini_batch_size int, eta float64, test *MNIST) {

	mini_batch_seg := len(mnist.Data) / mini_batch_size
	fmt.Printf("divide the data into %v segs\n", mini_batch_seg)

	/*
	if mini_batch_seg == 0 {
		mini_batch_size = len(mnist.Data)
	}
	*/

	for round := 0; round < rounds; round ++ {
		//first shuffle the data
		shuffle(mnist.Data)
		fmt.Printf("start round %d\n", round)


		for i := 0; i < mini_batch_seg - 1; i ++ {
			start := i * mini_batch_size
			end := (i + 1) * mini_batch_size;
			network.update_mini_batch(mnist.Data[start:end] , eta);
		}

		network.update_mini_batch(mnist.Data[mini_batch_seg * mini_batch_size:], eta)


		//to speed up the calculation
		//network.update_mini_batch(mnist.Data[0:mini_batch_size], eta)
		if test != nil {
			correct := network.evaluate(test)
			fmt.Printf("after round %d, the correct number is %d\n", round, correct)
		}
	}
}

func (net *Network)backprop(input *Rect, output *Rect) (deltaW, deltaB []*Rect){
	zRect := make([]*Rect, net.LayerCount, net.LayerCount)
	activations := make([]*Rect, net.LayerCount, net.LayerCount)

	if input.Width != net.Layers[0] {
		panic("backprop only accepts input with the same length with network: required is " + strconv.Itoa(net.Layers[0]) + " but actually is " + strconv.Itoa(input.Width))
	} else {
		zRect[0] = input
		activations[0] = input
	}

	sigmoided := input

	for i := 1; i < net.LayerCount; i ++{
		zRect[i] = net.Weights[i - 1].Junc(sigmoided).Add(net.Biases[i - 1])
		sigmoided = zRect[i].Copy().Operate(OperateWrapper(Sigmoid))
		activations[i] = sigmoided
	}

	delta := activations[net.LayerCount - 1].Minus(output).Operate(
		func(data float64, x, y int) float64{
			return data * SigmoidPrime(zRect[net.LayerCount - 1].Get(x, y));
		});

	deltaB = make([]*Rect, net.LayerCount - 1, net.LayerCount - 1)
	deltaW = make([]*Rect, net.LayerCount - 1, net.LayerCount - 1)

	deltaB[net.LayerCount - 2] = delta
	deltaW[net.LayerCount - 2] = delta.Junc(activations[net.LayerCount - 2].Transpose())

	for layer := net.LayerCount - 2; layer >= 1; layer -- {
		delta = net.Weights[layer].Transpose().Junc(delta).Operate(
			func(data float64, x, y int) float64{
				//update := data * SigmoidPrime(zRect[layer].Get(x, y));
				//fmt.Printf("delta is %v\n", update)
				//return update
				return data * SigmoidPrime(zRect[layer].Get(x, y));
			});
		deltaB[layer - 1] = delta
		deltaW[layer - 1] = delta.Junc(activations[layer - 1].Transpose())
	}
	return
}

func (net *Network)update_mini_batch(data []*MNISTData, eta float64){
	n := len(data)
	//fmt.Printf("update_mini_batch with data of length %v\n", n)
	s_eta := eta / float64(n);
	for i := 0; i < n ; i ++ {
		deltaW, deltaB := net.backprop(data[i].Image, data[i].Label)
		if len(deltaW) != len(deltaB) || len(deltaW) != (net.LayerCount - 1) {
			panic("update mini batch, backprop returned delta with wrong length")
		}
		for j := 0; j < net.LayerCount - 1; j ++ {
			net.Weights[j].Operate(func(weight float64, x, y int) float64{
				return weight - s_eta * deltaW[j].Get(x, y);
			});
			net.Biases[j].Operate(func(biase float64, x, y int) float64{
				return biase - s_eta * deltaB[j].Get(x, y);
			});
		}
	}
}


func shuffle(data []*MNISTData) {
	n := len(data)
	for i := 0; i < n; i ++ {
		t := int(rand.Float64() * float64(n - i));
		data[i], data[i + t] = data[i + t], data[i];
	}
}