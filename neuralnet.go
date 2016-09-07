// Copyright 2016 Clément Martinez

package neuralnet

import (
	"math"
	"math/rand"
	"time"
)

// Network represents a neural network.
type Network struct {
	Sizes   []int
	Weights [][][]float64
	Biases  [][]float64
}

// Set defines a training or testing database
type Set interface {
	GetVects(int) (input, output []float64)
	Count() int
}

// New creates a neural network given the layer sizes.
// All weights and biases are initialized to 0.
func New(sizes []int) (net *Network) {
	net = &Network{
		Sizes: sizes,
	}

	net.Biases = make([][]float64, len(net.Sizes)-1)
	net.Weights = make([][][]float64, len(net.Sizes)-1)
	for i := 1; i < len(net.Sizes); i++ {
		net.Biases[i-1] = make([]float64, net.Sizes[i])

		net.Weights[i-1] = make([][]float64, net.Sizes[i])
		for j := 0; j < net.Sizes[i]; j++ {
			net.Weights[i-1][j] = make([]float64, net.Sizes[i-1])
		}
	}

	return
}

// Randomize randomizes the weights and biases using a standard normal
// distribution.
func (net *Network) Randomize() {
	for i := range net.Biases {
		for j := range net.Biases[i] {
			net.Biases[i][j] = rand.NormFloat64()
		}
	}

	for i := range net.Weights {
		for j := range net.Weights[i] {
			for k := range net.Weights[i][j] {
				net.Weights[i][j][k] = rand.NormFloat64()
			}
		}
	}
}

// FeedForward computes the neural network output for a given input.
func (net *Network) FeedForward(in []float64) []float64 {
	for i := range net.Biases {
		in = multMatrixVect(net.Weights[i], in)
		applySigmoidVector(in)
	}
	return in
}

// CountLayers returns the number of layers.
func (net *Network) CountLayers() int {
	return len(net.Sizes)
}

// Sigmoid evaluate the simgmoid function with x and returns it.
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// SigmoidDiff is the differential function of the sigmoid
func SigmoidDiff(x float64) float64 {
	sx := Sigmoid(x)
	return sx * (1. - sx)
}

// SeedRand initializes the random generator.
func (net *Network) SeedRand() {
	rand.Seed(int64(time.Now().Nanosecond()))
}
